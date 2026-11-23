"""Training data of the BGE-M3 relation encoder.

The source SubgraphRAG graphs are directed multigraphs represented by three
parallel ID arrays.  This module deliberately does not convert them to a
``networkx.DiGraph``: doing so drops parallel relations between the same pair
of entities and can turn valid shortest-path relations into negatives.

Rich records retain raw relation/entity values for auditability.  Formatting
is applied only when a flat FlagEmbedding record is derived.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .relation_text import format_relation


SCHEMA_VERSION = "phase11.relation_finetune.v1"
EDGE_COMPLETE_LABEL_VERSION = "edge_complete_v1"
LEVEL_PRECEDENCE = (4, 3, 2, 1)
SEMANTIC_RESOURCE_STATUS = "not_checked_resource_unavailable"


@dataclass(frozen=True)
class BuilderConfig:
    """Settings that affect the deterministic derived dataset."""

    relation_format: str = "split"
    seed: int = 42
    dev_ratio: float = 0.1
    split_strategy: str = "record_hash"
    max_negatives: int = 32
    lexical_threshold: float = 0.5
    graph_hard_max_distance: int = 1
    l1_ratio: float = 0.15
    l2_ratio: float = 0.20
    l3_ratio: float = 0.30
    l4_ratio: float = 0.25

    def validate(self) -> None:
        # Validate the formatter without coupling this file to its registry.
        format_relation("domain.type.relation", self.relation_format)
        if not 0.0 <= self.dev_ratio < 1.0:
            raise ValueError("dev_ratio must be in [0, 1)")
        if self.split_strategy not in {"record_hash", "entity_component"}:
            raise ValueError(
                "split_strategy must be 'record_hash' or 'entity_component'"
            )
        if self.max_negatives < 1:
            raise ValueError("max_negatives must be positive")
        if not 0.0 <= self.lexical_threshold <= 1.0:
            raise ValueError("lexical_threshold must be in [0, 1]")
        if self.graph_hard_max_distance < 0:
            raise ValueError("graph_hard_max_distance cannot be negative")
        ratios = self.negative_ratios
        if any(value < 0 for value in ratios.values()) or sum(ratios.values()) <= 0:
            raise ValueError("negative ratios must be non-negative and sum to > 0")

    @property
    def negative_ratios(self) -> dict[int, float]:
        return {
            1: self.l1_ratio,
            2: self.l2_ratio,
            3: self.l3_ratio,
            4: self.l4_ratio,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 representation used by all content hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    _atomic_replace_bytes(Path(path), canonical_json_bytes(value) + b"\n")


def atomic_write_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(canonical_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _as_python_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _entity_values(sample: Mapping[str, Any]) -> list[str]:
    return [*sample.get("text_entity_list", []), *sample.get("non_text_entity_list", [])]


@dataclass
class GraphIndex:
    entity_values: list[str]
    relation_values: list[str]
    heads: list[int]
    relation_ids: list[int]
    tails: list[int]
    pair_to_triples: dict[tuple[int, int], list[int]]
    out_neighbors: list[set[int]]
    in_neighbors: list[set[int]]
    undirected_neighbors: list[set[int]]
    relation_to_triples: dict[str, list[int]]


def index_sample_graph(sample: Mapping[str, Any]) -> GraphIndex:
    """Validate and index one source graph without collapsing parallel edges."""

    required = {
        "id",
        "question",
        "relation_list",
        "h_id_list",
        "r_id_list",
        "t_id_list",
        "q_entity_id_list",
        "a_entity_id_list",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"sample is missing required keys: {missing}")

    entity_values = _entity_values(sample)
    relation_values = [str(value) for value in sample["relation_list"]]
    heads = [int(value) for value in sample["h_id_list"]]
    relation_ids = [int(value) for value in sample["r_id_list"]]
    tails = [int(value) for value in sample["t_id_list"]]
    if not (len(heads) == len(relation_ids) == len(tails)):
        raise ValueError(f"triple ID arrays have different lengths for {sample['id']}")
    if len(set(relation_values)) != len(relation_values):
        raise ValueError(f"relation_list contains duplicate raw IDs for {sample['id']}")

    num_entities = len(entity_values)
    num_relations = len(relation_values)
    pair_to_triples: dict[tuple[int, int], list[int]] = defaultdict(list)
    relation_to_triples: dict[str, list[int]] = defaultdict(list)
    out_neighbors = [set() for _ in range(num_entities)]
    in_neighbors = [set() for _ in range(num_entities)]
    undirected_neighbors = [set() for _ in range(num_entities)]
    for triple_id, (head, relation_id, tail) in enumerate(zip(heads, relation_ids, tails)):
        if not 0 <= head < num_entities or not 0 <= tail < num_entities:
            raise ValueError(
                f"entity index out of range in sample {sample['id']}, triple {triple_id}"
            )
        if not 0 <= relation_id < num_relations:
            raise ValueError(
                f"relation index out of range in sample {sample['id']}, triple {triple_id}"
            )
        pair_to_triples[(head, tail)].append(triple_id)
        relation_to_triples[relation_values[relation_id]].append(triple_id)
        out_neighbors[head].add(tail)
        in_neighbors[tail].add(head)
        undirected_neighbors[head].add(tail)
        undirected_neighbors[tail].add(head)

    for role in ("q_entity_id_list", "a_entity_id_list"):
        for local_id in sample[role]:
            if not 0 <= int(local_id) < num_entities:
                raise ValueError(f"{role} contains an out-of-range ID for {sample['id']}")

    return GraphIndex(
        entity_values=entity_values,
        relation_values=relation_values,
        heads=heads,
        relation_ids=relation_ids,
        tails=tails,
        pair_to_triples=dict(pair_to_triples),
        out_neighbors=out_neighbors,
        in_neighbors=in_neighbors,
        undirected_neighbors=undirected_neighbors,
        relation_to_triples=dict(relation_to_triples),
    )


def _bfs_distances(adjacency: Sequence[set[int]], sources: Iterable[int]) -> list[int | None]:
    distances: list[int | None] = [None] * len(adjacency)
    queue: deque[int] = deque()
    for source in sorted(set(int(value) for value in sources)):
        if distances[source] is None:
            distances[source] = 0
            queue.append(source)
    while queue:
        node = queue.popleft()
        assert distances[node] is not None
        next_distance = distances[node] + 1
        for neighbor in sorted(adjacency[node]):
            if distances[neighbor] is None:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def _shortest_path_dag(
    graph: GraphIndex,
    source: int,
    target: int,
    direction: str,
    distance_from_source: Sequence[int | None] | None = None,
    distance_to_target: Sequence[int | None] | None = None,
) -> tuple[dict[str, Any] | None, set[int], set[tuple[int, int]]]:
    """Create a compressed shortest-path DAG for one directed source/target pair."""

    if distance_from_source is None:
        distance_from_source = _bfs_distances(graph.out_neighbors, [source])
    hop_count = distance_from_source[target]
    if hop_count is None:
        return None, set(), set()
    if distance_to_target is None:
        distance_to_target = _bfs_distances(graph.in_neighbors, [target])

    dag_pairs = {
        (head, tail)
        for head, tail in graph.pair_to_triples
        if distance_from_source[head] is not None
        and distance_to_target[tail] is not None
        and distance_from_source[head] + 1 + distance_to_target[tail] == hop_count
    }
    dag_nodes = {source, target}
    for head, tail in dag_pairs:
        dag_nodes.update((head, tail))

    node_path_counts = {node: 0 for node in dag_nodes}
    edge_path_counts = {node: 0 for node in dag_nodes}
    node_path_counts[source] = 1
    edge_path_counts[source] = 1
    pairs_by_head: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in dag_pairs:
        pairs_by_head[pair[0]].append(pair)
    for node in sorted(dag_nodes, key=lambda item: (distance_from_source[item], item)):
        for head, tail in sorted(pairs_by_head.get(node, [])):
            node_path_counts[tail] += node_path_counts[head]
            edge_path_counts[tail] += edge_path_counts[head] * len(
                graph.pair_to_triples[(head, tail)]
            )

    triple_ids = {
        triple_id for pair in dag_pairs for triple_id in graph.pair_to_triples[pair]
    }
    nodes = [
        {
            "local_id": node,
            "raw_id": graph.entity_values[node],
            "distance_from_source": distance_from_source[node],
            "distance_to_target": distance_to_target[node],
        }
        for node in sorted(dag_nodes, key=lambda item: (distance_from_source[item], item))
    ]
    edges = []
    for head, tail in sorted(dag_pairs, key=lambda pair: (distance_from_source[pair[0]], pair)):
        pair_triples = sorted(graph.pair_to_triples[(head, tail)])
        edges.append(
            {
                "head_local_id": head,
                "tail_local_id": tail,
                "head_raw_id": graph.entity_values[head],
                "tail_raw_id": graph.entity_values[tail],
                "triple_ids": pair_triples,
                "relations": sorted(
                    {graph.relation_values[graph.relation_ids[index]] for index in pair_triples}
                ),
            }
        )
    metadata = {
        "direction": direction,
        "source_local_id": source,
        "source_raw_id": graph.entity_values[source],
        "target_local_id": target,
        "target_raw_id": graph.entity_values[target],
        "hop_count": hop_count,
        "node_path_count": node_path_counts[target],
        "edge_path_count_including_parallel_triples": edge_path_counts[target],
        "nodes": nodes,
        "edges": edges,
    }
    return metadata, triple_ids, dag_pairs


def derive_edge_complete_shortest_paths(
    sample: Mapping[str, Any], graph: GraphIndex | None = None
) -> tuple[list[dict[str, Any]], set[int], set[str], set[tuple[int, int]]]:
    """Derive all shortest-path edges under the legacy direction policy.

    For each topic/answer pair, the source pipeline considers both directed
    orientations.  If both exist, only the shorter orientation(s) are kept;
    a tie keeps both.  Every parallel triple on a selected DAG edge is then a
    positive, which is the edge-complete correction to the old ``DiGraph``
    implementation.
    """

    graph = graph or index_sample_graph(sample)
    entries: list[dict[str, Any]] = []
    positive_triples: set[int] = set()
    positive_pairs: set[tuple[int, int]] = set()
    topics = sorted(set(int(value) for value in sample["q_entity_id_list"]))
    answers = sorted(set(int(value) for value in sample["a_entity_id_list"]))
    out_distance_cache: dict[int, list[int | None]] = {}
    in_distance_cache: dict[int, list[int | None]] = {}

    def out_distances(source: int) -> list[int | None]:
        if source not in out_distance_cache:
            out_distance_cache[source] = _bfs_distances(graph.out_neighbors, [source])
        return out_distance_cache[source]

    def in_distances(target: int) -> list[int | None]:
        if target not in in_distance_cache:
            in_distance_cache[target] = _bfs_distances(graph.in_neighbors, [target])
        return in_distance_cache[target]

    for topic in topics:
        for answer in answers:
            forward, forward_triples, forward_pairs = _shortest_path_dag(
                graph,
                topic,
                answer,
                "topic_to_answer",
                distance_from_source=out_distances(topic),
                distance_to_target=in_distances(answer),
            )
            reverse, reverse_triples, reverse_pairs = _shortest_path_dag(
                graph,
                answer,
                topic,
                "answer_to_topic",
                distance_from_source=out_distances(answer),
                distance_to_target=in_distances(topic),
            )
            reachable = [item for item in (forward, reverse) if item is not None]
            selected: list[dict[str, Any]] = []
            if reachable:
                minimum = min(item["hop_count"] for item in reachable)
                if forward is not None and forward["hop_count"] == minimum:
                    selected.append(forward)
                    positive_triples.update(forward_triples)
                    positive_pairs.update(forward_pairs)
                if reverse is not None and reverse["hop_count"] == minimum:
                    selected.append(reverse)
                    positive_triples.update(reverse_triples)
                    positive_pairs.update(reverse_pairs)
            entries.append(
                {
                    "topic_entity": {
                        "local_id": topic,
                        "raw_id": graph.entity_values[topic],
                    },
                    "answer_entity": {
                        "local_id": answer,
                        "raw_id": graph.entity_values[answer],
                    },
                    "forward_hops": None if forward is None else forward["hop_count"],
                    "reverse_hops": None if reverse is None else reverse["hop_count"],
                    "selected_directions": selected,
                }
            )

    positive_relations = {
        graph.relation_values[graph.relation_ids[triple_id]] for triple_id in positive_triples
    }
    _assert_shortest_path_invariants(
        graph,
        entries,
        positive_triples,
        positive_relations,
        positive_pairs,
    )
    return entries, positive_triples, positive_relations, positive_pairs


def _assert_shortest_path_invariants(
    graph: GraphIndex,
    entries: Sequence[Mapping[str, Any]],
    positive_triples: set[int],
    positive_relations: set[str],
    positive_pairs: set[tuple[int, int]],
) -> None:
    """Fail closed if compressed DAG metadata and authoritative labels diverge."""

    observed_triples: set[int] = set()
    observed_pairs: set[tuple[int, int]] = set()
    for entry in entries:
        available_hops = [
            int(value)
            for value in (entry["forward_hops"], entry["reverse_hops"])
            if value is not None
        ]
        selected = entry["selected_directions"]
        if not available_hops:
            if selected:
                raise AssertionError("unreachable topic/answer pair contains a selected DAG")
            continue
        minimum_hops = min(available_hops)
        if not selected:
            raise AssertionError("reachable topic/answer pair lacks a selected DAG")

        for dag in selected:
            hop_count = int(dag["hop_count"])
            if hop_count != minimum_hops:
                raise AssertionError("selected direction is not shortest across both orientations")
            node_metadata = {int(node["local_id"]): node for node in dag["nodes"]}
            source = int(dag["source_local_id"])
            target = int(dag["target_local_id"])
            if node_metadata[source]["distance_from_source"] != 0:
                raise AssertionError("shortest-path DAG source is not at layer zero")
            if node_metadata[target]["distance_from_source"] != hop_count:
                raise AssertionError("shortest-path DAG target is not at the expected layer")

            for edge in dag["edges"]:
                head = int(edge["head_local_id"])
                tail = int(edge["tail_local_id"])
                pair = (head, tail)
                expected_triples = sorted(graph.pair_to_triples.get(pair, []))
                if edge["triple_ids"] != expected_triples:
                    raise AssertionError("DAG edge does not retain every parallel directed triple")
                expected_relations = sorted(
                    {
                        graph.relation_values[graph.relation_ids[triple_id]]
                        for triple_id in expected_triples
                    }
                )
                if edge["relations"] != expected_relations:
                    raise AssertionError("DAG edge relation metadata does not match its triples")
                head_distance = node_metadata[head]["distance_from_source"]
                tail_distance = node_metadata[tail]["distance_from_source"]
                if head_distance is None or tail_distance != head_distance + 1:
                    raise AssertionError(
                        "shortest-path DAG edge does not advance exactly one layer"
                    )
                observed_pairs.add(pair)
                observed_triples.update(expected_triples)

            if int(dag["node_path_count"]) < 1:
                raise AssertionError("selected DAG must contain at least one node-level path")
            if int(dag["edge_path_count_including_parallel_triples"]) < int(
                dag["node_path_count"]
            ):
                raise AssertionError("parallel-edge path count cannot be below node path count")

    if observed_triples != positive_triples:
        raise AssertionError("positive triple IDs do not equal the selected DAG edge union")
    if observed_pairs != positive_pairs:
        raise AssertionError("positive entity pairs do not equal the selected DAG edge union")
    observed_relations = {
        graph.relation_values[graph.relation_ids[triple_id]] for triple_id in observed_triples
    }
    if observed_relations != positive_relations:
        raise AssertionError("positive raw relations do not equal the selected DAG relation union")


def edge_complete_shortest_path_labels(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return cache-independent shortest-path labels for ``RetrieverDataset``.

    This stable public function intentionally accepts only the processed sample:
    it never reads or mutates the legacy ``triple_scores`` cache.  The returned
    triple IDs are authoritative for ``edge_complete_v1`` and include every
    parallel directed edge on every selected shortest-path DAG.
    """

    graph = index_sample_graph(sample)
    dags, positive_triples, positive_relations, _ = derive_edge_complete_shortest_paths(
        sample, graph
    )
    selected_hops = [
        int(dag["hop_count"])
        for pair_entry in dags
        for dag in pair_entry["selected_directions"]
    ]
    return {
        "shortest_path_label_version": EDGE_COMPLETE_LABEL_VERSION,
        "positive_triple_ids": sorted(positive_triples),
        "positive_relations": sorted(positive_relations),
        "max_path_length": max(selected_hops) if selected_hops else None,
        "shortest_path_dags": dags,
    }


def legacy_cached_positives(
    sample: Mapping[str, Any], label_entry: Mapping[str, Any], graph: GraphIndex | None = None
) -> tuple[set[int], set[str]]:
    graph = graph or index_sample_graph(sample)
    if "triple_scores" not in label_entry:
        raise ValueError(f"legacy label for {sample['id']} lacks triple_scores")
    scores = _as_python_list(label_entry["triple_scores"])
    if len(scores) != len(graph.heads):
        raise ValueError(f"legacy label length mismatch for {sample['id']}")
    positive_triples = {index for index, score in enumerate(scores) if float(score) > 0.0}
    positive_relations = {
        graph.relation_values[graph.relation_ids[triple_id]] for triple_id in positive_triples
    }
    return positive_triples, positive_relations


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
}


def relation_tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower().replace("_", " ").replace(".", " ")))


def query_tokens(value: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(value.lower()) if token not in _QUERY_STOPWORDS}


def _overlap_coefficient(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def lexical_hardness(question: str, relation: str, positives: Iterable[str]) -> float:
    tokens = relation_tokens(relation)
    scores = [_overlap_coefficient(tokens, query_tokens(question))]
    scores.extend(_overlap_coefficient(tokens, relation_tokens(positive)) for positive in positives)
    return max(scores, default=0.0)


def relation_domain(relation: str) -> str:
    return relation.split(".", 1)[0]


def _multi_source_undirected_distances(
    graph: GraphIndex, sources: Iterable[int]
) -> list[int | None]:
    return _bfs_distances(graph.undirected_neighbors, sources)


def _relation_graph_distances(
    graph: GraphIndex, topics: Iterable[int]
) -> dict[str, int | None]:
    distances = _multi_source_undirected_distances(graph, topics)
    output: dict[str, int | None] = {}
    for relation, triple_ids in graph.relation_to_triples.items():
        observed = [
            distance
            for triple_id in triple_ids
            for distance in (distances[graph.heads[triple_id]], distances[graph.tails[triple_id]])
            if distance is not None
        ]
        output[relation] = min(observed) if observed else None
    return output


def _stable_relation_key(relation: str, query_id: str, seed: int, level: int) -> tuple[str, str]:
    payload = f"{seed}\0{query_id}\0{level}\0{relation}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), relation


def _largest_remainder_quotas(total: int, ratios: Mapping[int, float]) -> dict[int, int]:
    denominator = sum(ratios.values())
    exact = {level: total * ratios[level] / denominator for level in ratios}
    quotas = {level: math.floor(value) for level, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(ratios, key=lambda level: (-(exact[level] - quotas[level]), -level))
    for level in order[:remaining]:
        quotas[level] += 1
    return quotas


def _sample_negative_pools(
    pools: Mapping[int, list[dict[str, Any]]],
    query_id: str,
    config: BuilderConfig,
) -> list[dict[str, Any]]:
    total_available = sum(len(values) for values in pools.values())
    target_total = min(config.max_negatives, total_available)
    quotas = _largest_remainder_quotas(target_total, config.negative_ratios)
    ordered = {
        level: sorted(
            values,
            key=lambda item: _stable_relation_key(
                item["relation"], query_id, config.seed, level
            ),
        )
        for level, values in pools.items()
    }
    selected: dict[int, list[dict[str, Any]]] = {
        level: values[: min(quotas.get(level, 0), len(values))]
        for level, values in ordered.items()
    }
    selected_relations = {item["relation"] for values in selected.values() for item in values}
    remaining = target_total - len(selected_relations)
    if remaining:
        # Fill shortages from hardest to easiest while retaining deterministic order.
        for level in LEVEL_PRECEDENCE:
            for item in ordered.get(level, []):
                if remaining == 0:
                    break
                if item["relation"] not in selected_relations:
                    selected.setdefault(level, []).append(item)
                    selected_relations.add(item["relation"])
                    remaining -= 1
            if remaining == 0:
                break
    output = [item for level in LEVEL_PRECEDENCE for item in selected.get(level, [])]
    if len({item["relation"] for item in output}) != len(output):
        raise AssertionError("negative sampling produced duplicate raw relation IDs")
    return output


def _entity_objects(local_ids: Iterable[int], graph: GraphIndex) -> list[dict[str, Any]]:
    return [
        {"local_id": local_id, "raw_id": graph.entity_values[local_id]}
        for local_id in sorted(set(int(value) for value in local_ids))
    ]


def _negative_candidates(
    sample: Mapping[str, Any],
    graph: GraphIndex,
    positives: set[str],
    positive_pairs: set[tuple[int, int]],
    legacy_positives: set[str],
    config: BuilderConfig,
) -> tuple[list[dict[str, Any]], dict[int, int], list[dict[str, Any]]]:
    candidates = set(graph.relation_values)
    candidate_negatives = candidates - positives
    topics = set(int(value) for value in sample["q_entity_id_list"])
    answers = set(int(value) for value in sample["a_entity_id_list"])

    ambiguity: dict[str, set[str]] = defaultdict(set)
    for relation in legacy_positives - positives:
        ambiguity[relation].add("legacy_cached_positive_not_in_edge_complete_gold")
    for (head, tail), triple_ids in graph.pair_to_triples.items():
        if (head in topics and tail in answers) or (head in answers and tail in topics):
            for triple_id in triple_ids:
                relation = graph.relation_values[graph.relation_ids[triple_id]]
                if relation not in positives:
                    ambiguity[relation].add("direct_topic_answer_pair")
    for head, tail in positive_pairs:
        for triple_id in graph.pair_to_triples.get((tail, head), []):
            relation = graph.relation_values[graph.relation_ids[triple_id]]
            if relation not in positives:
                ambiguity[relation].add("reverse_of_positive_entity_pair")

    positive_texts = {format_relation(relation, config.relation_format) for relation in positives}
    formatted_to_relations: dict[str, list[str]] = defaultdict(list)
    for relation in candidate_negatives:
        formatted_to_relations[format_relation(relation, config.relation_format)].append(relation)
    for formatted, relations in formatted_to_relations.items():
        if formatted in positive_texts:
            for relation in relations:
                ambiguity[relation].add("formatted_text_collides_with_positive")
        elif len(relations) > 1:
            keep = min(relations)
            for relation in relations:
                if relation != keep:
                    ambiguity[relation].add("formatted_text_duplicate_negative")

    graph_distances = _relation_graph_distances(graph, topics)
    positive_domains = {relation_domain(relation) for relation in positives}
    pools: dict[int, list[dict[str, Any]]] = {level: [] for level in LEVEL_PRECEDENCE}
    for relation in sorted(candidate_negatives - set(ambiguity)):
        distance = graph_distances[relation]
        lexical_score = lexical_hardness(sample["question"], relation, positives)
        same_domain = relation_domain(relation) in positive_domains
        reasons: list[str] = []
        if distance is not None and distance <= config.graph_hard_max_distance:
            level = 4
            reasons.append("graph_adjacent_not_on_shortest_path")
        elif lexical_score >= config.lexical_threshold:
            level = 3
            reasons.append("lexical_overlap_not_on_shortest_path")
        elif same_domain:
            level = 2
            reasons.append("same_domain_not_on_shortest_path")
        else:
            level = 1
            reasons.append("candidate_unrelated_domain")
        if lexical_score >= config.lexical_threshold and level != 3:
            reasons.append("also_lexically_similar")
        if same_domain and level not in (2, 3):
            reasons.append("also_same_domain")
        item = {
            "relation": relation,
            "level": level,
            "reasons": reasons,
            "lexical_score": round(lexical_score, 8),
            "min_undirected_hop_from_topic": distance,
            "verification": {
                "status": "verified_graph_checks_only",
                "checks": {
                    "edge_complete_positive_membership": "passed",
                    "legacy_cached_positive_membership": "passed",
                    "direct_topic_answer_pair": "passed",
                    "reverse_positive_entity_pair": "passed",
                    "inverse_relation_resource": SEMANTIC_RESOURCE_STATUS,
                    "semantic_alias_resource": SEMANTIC_RESOURCE_STATUS,
                },
            },
        }
        pools[level].append(item)

    selected = _sample_negative_pools(pools, str(sample["id"]), config)
    excluded = [
        {
            "relation": relation,
            "reasons": sorted(reasons),
            "verification_status": "excluded_ambiguous",
        }
        for relation, reasons in sorted(ambiguity.items())
    ]
    return selected, {level: len(pools[level]) for level in sorted(pools)}, excluded


def assign_training_partition(query_id: str, dev_ratio: float, seed: int) -> str:
    payload = f"phase11-fit-dev\0{seed}\0{query_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    unit_interval = integer / 2**64
    return "dev" if unit_interval < dev_ratio else "fit"


def _entity_component_dev_ids(
    rich_records: Sequence[Mapping[str, Any]], dev_ratio: float, seed: int
) -> set[str]:
    """Select whole topic+answer entity components for an entity-disjoint dev set."""

    eligible = [
        record
        for record in rich_records
        if not str(record["training_partition"]).startswith("excluded_")
    ]
    target = round(len(eligible) * dev_ratio)
    if target == 0:
        return set()

    parent = list(range(len(eligible)))
    sizes = [1] * len(eligible)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]

    entity_owner: dict[str, int] = {}
    for index, record in enumerate(eligible):
        entities = {
            str(entity["raw_id"])
            for key in ("topic_entities", "answer_entities")
            for entity in record.get(key, [])
        }
        for entity in entities:
            owner = entity_owner.setdefault(entity, index)
            union(index, owner)

    components: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(eligible):
        components[find(index)].append(str(record["query_id"]))
    ordered = sorted(
        (sorted(query_ids) for query_ids in components.values()),
        key=lambda query_ids: hashlib.sha256(
            f"phase11-entity-component\0{seed}\0{'|'.join(query_ids)}".encode()
        ).hexdigest(),
    )

    # Deterministic subset-sum chooses the requested record count exactly when
    # possible. Giant components larger than the target remain in fit.
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for component_index, query_ids in enumerate(ordered):
        component_size = len(query_ids)
        for subtotal in sorted(tuple(reachable), reverse=True):
            candidate = subtotal + component_size
            if candidate <= target and candidate not in reachable:
                reachable[candidate] = (subtotal, component_index)
    selected_total = max(reachable)
    selected_components: set[int] = set()
    while selected_total:
        previous, component_index = reachable[selected_total]  # type: ignore[misc]
        selected_components.add(component_index)
        selected_total = previous
    return {
        query_id
        for component_index in selected_components
        for query_id in ordered[component_index]
    }


def build_rich_and_flat_record(
    sample: Mapping[str, Any],
    label_entry: Mapping[str, Any],
    dataset: str,
    source_split: str,
    config: BuilderConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Build one auditable rich record and its optional flat train record."""

    config.validate()
    graph = index_sample_graph(sample)
    dags, positive_triples, positives, positive_pairs = derive_edge_complete_shortest_paths(
        sample, graph
    )
    legacy_triples, legacy_positives = legacy_cached_positives(sample, label_entry, graph)

    candidates = set(graph.relation_values)
    if not positives <= candidates or not legacy_positives <= candidates:
        raise AssertionError("positive relation outside the provided candidate graph")

    negatives: list[dict[str, Any]] = []
    negative_pool_counts = {level: 0 for level in sorted(LEVEL_PRECEDENCE)}
    excluded_ambiguous: list[dict[str, Any]] = []
    if positives:
        negatives, negative_pool_counts, excluded_ambiguous = _negative_candidates(
            sample,
            graph,
            positives,
            positive_pairs,
            legacy_positives,
            config,
        )

    negative_relations = {item["relation"] for item in negatives}
    if positives & negative_relations:
        raise AssertionError("positive and negative raw relation sets overlap")
    if len(negative_relations) != len(negatives):
        raise AssertionError("negative levels are not disjoint")
    if any(item["relation"] not in candidates for item in negatives):
        raise AssertionError("negative relation outside candidate graph")

    if not positives:
        training_partition = "excluded_no_shortest_path"
    elif not negatives:
        training_partition = "excluded_no_verified_negative"
    elif config.split_strategy == "entity_component":
        # Component membership requires all eligible records, so build_records
        # assigns fit/dev after this record has been materialized.
        training_partition = "fit"
    else:
        training_partition = assign_training_partition(
            str(sample["id"]), config.dev_ratio, config.seed
        )

    rich: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "query_id": str(sample["id"]),
        "dataset": dataset,
        "source_split": source_split,
        "training_partition": training_partition,
        "query": str(sample["question"]),
        "topic_entities": _entity_objects(sample["q_entity_id_list"], graph),
        "answer_entities": _entity_objects(sample["a_entity_id_list"], graph),
        "declared_topic_entities": [str(value) for value in sample.get("q_entity", [])],
        "declared_answer_entities": [str(value) for value in sample.get("a_entity", [])],
        "shortest_path_dags": dags,
        "positive_triple_ids": sorted(positive_triples),
        "positive_relations": sorted(positives),
        "legacy_positive_triple_ids": sorted(legacy_triples),
        "legacy_positive_relations": sorted(legacy_positives),
        "legacy_delta": {
            "edge_complete_only_triple_ids": sorted(positive_triples - legacy_triples),
            "legacy_only_triple_ids": sorted(legacy_triples - positive_triples),
            "edge_complete_only_relations": sorted(positives - legacy_positives),
            "legacy_only_relations": sorted(legacy_positives - positives),
        },
        "candidate_relations": {
            "source": "provided_sample_graph",
            "all": sorted(candidates),
        },
        "negatives": negatives,
        "negative_pool_counts": {
            str(level): count for level, count in negative_pool_counts.items()
        },
        "verification": {
            "status": (
                "not_applicable_no_shortest_path"
                if not positives
                else "passed_graph_checks_semantic_resources_unavailable"
            ),
            "checks": {
                "positive_relations_subset_of_candidates": "passed",
                "negative_relations_disjoint_from_positives": "passed",
                "negative_level_precedence_disjoint": "passed",
                "direct_topic_answer_pair_exclusion": "passed",
                "reverse_positive_entity_pair_exclusion": "passed",
                "inverse_relation_resource": SEMANTIC_RESOURCE_STATUS,
                "semantic_alias_resource": SEMANTIC_RESOURCE_STATUS,
            },
            "excluded_ambiguous_relations": excluded_ambiguous,
        },
        "relation_format": config.relation_format,
        "builder_config": asdict(config),
    }
    record_hash = object_sha256(rich)
    rich["record_sha256"] = record_hash

    if training_partition.startswith("excluded_"):
        return rich, None

    pos_ids = sorted(positives)
    neg_items = negatives
    neg_ids = [item["relation"] for item in neg_items]
    pos_texts = [format_relation(relation, config.relation_format) for relation in pos_ids]
    neg_texts = [format_relation(relation, config.relation_format) for relation in neg_ids]
    if set(pos_texts) & set(neg_texts):
        raise AssertionError("formatted positive and negative relation texts overlap")
    ambiguous_ids = {item["relation"] for item in excluded_ambiguous}
    # Numerator positives and denominator-only exclusions are deliberately
    # separate.  Treating legacy-only/ambiguous relations as positives would
    # silently change the edge-complete training target.
    known_positive_texts = sorted(set(pos_texts))
    ignored_passage_texts = sorted(
        {
            format_relation(relation, config.relation_format)
            for relation in (legacy_positives - positives) | ambiguous_ids
        }
        - set(known_positive_texts)
    )
    flat = {
        "query_id": str(sample["id"]),
        "query": str(sample["question"]),
        "pos": pos_texts,
        "neg": neg_texts,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "known_positive_texts": known_positive_texts,
        "ignored_passage_texts": ignored_passage_texts,
        "negative_levels": [int(item["level"]) for item in neg_items],
        "relation_format": config.relation_format,
        "source_record_sha256": record_hash,
    }
    if not (len(flat["pos"]) == len(flat["pos_ids"])):
        raise AssertionError("flat positive IDs/texts are not aligned")
    if not (len(flat["neg"]) == len(flat["neg_ids"]) == len(flat["negative_levels"])):
        raise AssertionError("flat negative IDs/texts/levels are not aligned")
    return rich, flat


def build_records(
    samples: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    dataset: str,
    source_split: str,
    config: BuilderConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build and deterministically order rich, fit, and dev records."""

    query_ids = [str(sample["id"]) for sample in samples]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("source split contains duplicate query IDs")
    normalized_labels: dict[str, Mapping[str, Any]] = {}
    for raw_query_id, label_entry in labels.items():
        query_id = str(raw_query_id)
        if query_id in normalized_labels:
            raise ValueError(
                "legacy labels contain colliding query IDs after string conversion: "
                f"{query_id}"
            )
        normalized_labels[query_id] = label_entry
    missing_labels = sorted(set(query_ids) - set(normalized_labels))
    if missing_labels:
        raise ValueError(f"missing legacy labels for query IDs: {missing_labels[:5]}")

    rich_records: list[dict[str, Any]] = []
    built_pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    fit_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: str(item["id"])):
        rich, flat = build_rich_and_flat_record(
            sample,
            normalized_labels[str(sample["id"])],
            dataset=dataset,
            source_split=source_split,
            config=config,
        )
        rich_records.append(rich)
        built_pairs.append((rich, flat))

    if config.split_strategy == "entity_component":
        dev_ids = _entity_component_dev_ids(rich_records, config.dev_ratio, config.seed)
        for rich, flat in built_pairs:
            if flat is None:
                continue
            rich["training_partition"] = (
                "dev" if str(rich["query_id"]) in dev_ids else "fit"
            )
            rich.pop("record_sha256", None)
            record_hash = object_sha256(rich)
            rich["record_sha256"] = record_hash
            flat["source_record_sha256"] = record_hash

    for rich, flat in built_pairs:
        if flat is not None:
            if rich["training_partition"] == "fit":
                fit_records.append(flat)
            elif rich["training_partition"] == "dev":
                dev_records.append(flat)
            else:
                raise AssertionError("flat record has an excluded training partition")
    return rich_records, fit_records, dev_records


def collect_relation_vocabularies(
    samples: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    candidate_vocab: set[str] = set()
    edge_positive_vocab: set[str] = set()
    legacy_positive_vocab: set[str] = set()
    for sample in samples:
        graph = index_sample_graph(sample)
        candidate_vocab.update(graph.relation_values)
        _, _, positives, _ = derive_edge_complete_shortest_paths(sample, graph)
        _, legacy = legacy_cached_positives(sample, labels[str(sample["id"])], graph)
        edge_positive_vocab.update(positives)
        legacy_positive_vocab.update(legacy)
    return {
        "candidate": candidate_vocab,
        "edge_complete_positive": edge_positive_vocab,
        "legacy_positive": legacy_positive_vocab,
    }


def vocabularies_from_rich_records(
    rich_records: Iterable[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Collect source vocabularies without repeating shortest-path traversal."""

    candidate_vocab: set[str] = set()
    edge_positive_vocab: set[str] = set()
    legacy_positive_vocab: set[str] = set()
    for record in rich_records:
        candidate_vocab.update(str(value) for value in record["candidate_relations"]["all"])
        edge_positive_vocab.update(str(value) for value in record["positive_relations"])
        legacy_positive_vocab.update(str(value) for value in record["legacy_positive_relations"])
    return {
        "candidate": candidate_vocab,
        "edge_complete_positive": edge_positive_vocab,
        "legacy_positive": legacy_positive_vocab,
    }


def vocabulary_overlap(
    train_vocabularies: Mapping[str, set[str]],
    comparison_vocabularies: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in sorted(train_vocabularies):
        train = train_vocabularies[name]
        comparison = comparison_vocabularies[name]
        intersection = train & comparison
        unseen = comparison - train
        output[name] = {
            "train_size": len(train),
            "comparison_size": len(comparison),
            "intersection_size": len(intersection),
            "comparison_coverage_by_train": (
                len(intersection) / len(comparison) if comparison else None
            ),
            "comparison_unseen_count": len(unseen),
            "comparison_unseen_relations": sorted(unseen),
        }
    return output


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "median": median,
        "max": max(values),
    }


def summarize_records(
    rich_records: Sequence[Mapping[str, Any]],
    fit_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    partitions = Counter(record["training_partition"] for record in rich_records)
    selected_levels = Counter(
        int(item["level"]) for record in rich_records for item in record["negatives"]
    )
    pool_levels = Counter()
    ambiguity_reasons = Counter()
    for record in rich_records:
        for level, count in record["negative_pool_counts"].items():
            pool_levels[int(level)] += int(count)
        for excluded in record["verification"]["excluded_ambiguous_relations"]:
            ambiguity_reasons.update(excluded["reasons"])
    graph_audit_eligible = sum(bool(record["positive_relations"]) for record in rich_records)
    graph_audit_passed = sum(
        bool(record["positive_relations"])
        and record["verification"]["status"]
        == "passed_graph_checks_semantic_resources_unavailable"
        for record in rich_records
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "rich_records": len(rich_records),
            "flat_fit_records": len(fit_records),
            "flat_dev_records": len(dev_records),
            "training_partitions": dict(sorted(partitions.items())),
        },
        "per_record": {
            "candidate_relations": _summary(
                [len(record["candidate_relations"]["all"]) for record in rich_records]
            ),
            "edge_complete_positive_relations": _summary(
                [len(record["positive_relations"]) for record in rich_records]
            ),
            "legacy_positive_relations": _summary(
                [len(record["legacy_positive_relations"]) for record in rich_records]
            ),
            "selected_negatives": _summary(
                [len(record["negatives"]) for record in rich_records]
            ),
        },
        "negative_levels": {
            "pool_counts": {str(level): pool_levels[level] for level in sorted(pool_levels)},
            "selected_counts": {
                str(level): selected_levels[level] for level in sorted(selected_levels)
            },
        },
        "legacy_delta": {
            "records_with_edge_complete_only_relations": sum(
                bool(record["legacy_delta"]["edge_complete_only_relations"])
                for record in rich_records
            ),
            "records_with_legacy_only_relations": sum(
                bool(record["legacy_delta"]["legacy_only_relations"])
                for record in rich_records
            ),
            "edge_complete_only_relation_mentions": sum(
                len(record["legacy_delta"]["edge_complete_only_relations"])
                for record in rich_records
            ),
            "legacy_only_relation_mentions": sum(
                len(record["legacy_delta"]["legacy_only_relations"])
                for record in rich_records
            ),
        },
        "verification": {
            "status_counts": dict(
                sorted(Counter(record["verification"]["status"] for record in rich_records).items())
            ),
            "excluded_ambiguous_relation_mentions": sum(
                len(record["verification"]["excluded_ambiguous_relations"])
                for record in rich_records
            ),
            "excluded_ambiguity_reason_mentions": dict(sorted(ambiguity_reasons.items())),
            "automated_graph_checks_only": {
                "eligible_records": graph_audit_eligible,
                "passed_records": graph_audit_passed,
                "pass_rate": (
                    graph_audit_passed / graph_audit_eligible
                    if graph_audit_eligible
                    else None
                ),
                "scope": (
                    "edge-complete membership, direct answer pairs, reverse positive pairs, "
                    "and level disjointness; inverse/semantic resources unavailable"
                ),
            },
            "inverse_relation_resource": SEMANTIC_RESOURCE_STATUS,
            "semantic_alias_resource": SEMANTIC_RESOURCE_STATUS,
        },
    }


def builder_config_dict(config: BuilderConfig) -> dict[str, Any]:
    config.validate()
    output = asdict(config)
    output["negative_level_precedence"] = list(LEVEL_PRECEDENCE)
    output["candidate_source"] = "provided_sample_graph"
    output["fit_dev_assignment"] = (
        "topic_answer_entity_connected_components_subset_sum"
        if config.split_strategy == "entity_component"
        else "sha256(query_id, seed)"
    )
    return output
