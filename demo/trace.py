"""Reconstruct the best bounded complete path for a selected RBF triple."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import GraphRecord

NEG = -1e4


@dataclass(frozen=True)
class PathState:
    score: float
    tag: int
    edges: tuple[tuple[int, int], ...]


def _put(states: dict[int, PathState], candidate: PathState) -> None:
    old = states.get(candidate.tag)
    if old is None or candidate.score > old.score:
        states[candidate.tag] = candidate


def _prune(states: dict[int, PathState]) -> dict[int, PathState]:
    return dict(sorted(states.items(), key=lambda item: -item[1].score)[:2])


def _best(states: dict[int, PathState], forbidden_tag: int) -> PathState | None:
    options = [state for tag, state in states.items() if tag != forbidden_tag]
    return max(options, key=lambda state: state.score, default=None)


def _directions(rec: GraphRecord):
    for index in range(rec.n_triples):
        h, t, r = int(rec.h[index]), int(rec.t[index]), int(rec.r[index])
        yield index, 0, h, t, 2 * r, 2 * r + 1
        yield index, 1, t, h, 2 * r + 1, 2 * r


def _prefix(rec: GraphRecord, anchor: int, log_pi: np.ndarray) -> list[dict[int, PathState]]:
    states = [dict() for _ in rec.names]
    states[anchor][-1] = PathState(0.0, -1, ())
    for _ in range(3):
        next_states = [dict(value) for value in states]
        for index, direction, source, target, tag, inverse in _directions(rec):
            previous = _best(states[source], inverse)
            if previous is None:
                continue
            candidate = PathState(
                previous.score + float(log_pi[int(rec.r[index]), direction]),
                tag, previous.edges + ((index, direction),),
            )
            _put(next_states[target], candidate)
        states = [_prune(value) for value in next_states]
    return states


def _continuation(rec: GraphRecord, log_pi: np.ndarray, log_stop: np.ndarray):
    one_step = [dict() for _ in rec.names]
    for index, direction, source, _, tag, _ in _directions(rec):
        relation = int(rec.r[index])
        _put(one_step[source], PathState(
            float(log_pi[relation, direction] + log_stop[relation, direction]),
            tag, ((index, direction),),
        ))
    one_step = [_prune(value) for value in one_step]
    two_steps = [dict(value) for value in one_step]
    for index, direction, source, target, tag, inverse in _directions(rec):
        tail = _best(one_step[target], inverse)
        if tail is None:
            continue
        relation = int(rec.r[index])
        _put(two_steps[source], PathState(
            float(log_pi[relation, direction]) + tail.score,
            tag, ((index, direction),) + tail.edges,
        ))
    return [_prune(value) for value in two_steps]


def trace_path(rec: GraphRecord, result: dict, edge_index: int) -> dict:
    """Trace the union path and report the multi-seed intersection contribution.

    RBF combines a maximum over complete paths (union) with a separate
    all-seeds term (intersection). The highlighted path explains the union.
    """
    if not 0 <= edge_index < rec.n_triples:
        raise IndexError(edge_index)
    log_pi = np.asarray(result["_log_pi"])
    log_stop = np.asarray(result["_log_stop"])
    completion = _continuation(rec, log_pi, log_stop)
    prefix_states = [_prefix(rec, anchor, log_pi) for anchor in result["topics"]]
    node_scores = [
        np.asarray([
            max((state.score for state in states.values()), default=NEG)
            for states in prefix
        ])
        for prefix in prefix_states
    ]
    h, t, relation = int(rec.h[edge_index]), int(rec.t[edge_index]), int(rec.r[edge_index])
    candidates = []
    for anchor_pos, (anchor, prefix) in enumerate(zip(result["topics"], prefix_states)):
        for direction, source, target, tag, inverse in (
            (0, h, t, 2 * relation, 2 * relation + 1),
            (1, t, h, 2 * relation + 1, 2 * relation),
        ):
            before = _best(prefix[source], inverse)
            if before is None:
                continue
            immediate = float(log_stop[relation, direction])
            after = _best(completion[target], inverse)
            if after is not None and after.score > immediate:
                continuation_score, continuation_edges = after.score, after.edges
            else:
                continuation_score, continuation_edges = immediate, ()
            union = before.score + float(log_pi[relation, direction]) + continuation_score
            other_seeds = sum(
                float(scores[target])
                for i, scores in enumerate(node_scores) if i != anchor_pos
            )
            candidates.append({
                "anchor": anchor, "direction": direction,
                "union_log_score": union,
                "intersection_log_score": union + other_seeds,
                "edges": before.edges + ((edge_index, direction),) + continuation_edges,
            })
    if not candidates:
        return {
            "path": [], "union_log_score": NEG,
            "intersection_log_score": NEG,
            "combined_log_score": float(np.logaddexp(NEG, NEG)),
            "note": "No bounded non-backtracking path reaches this triple from the selected topics.",
        }
    representative = max(candidates, key=lambda item: item["union_log_score"])
    union = max(item["union_log_score"] for item in candidates)
    intersection = max(item["intersection_log_score"] for item in candidates)
    path = []
    for index, direction in representative["edges"]:
        head, relation_name, tail = rec.triple(index)
        if direction:
            head, tail = tail, head
        path.append({
            "index": index, "head": head, "relation": relation_name,
            "tail": tail, "direction": "reverse" if direction else "forward",
        })
    return {
        "path": path, "anchor": rec.names[representative["anchor"]],
        "union_log_score": union, "intersection_log_score": intersection,
        "combined_log_score": float(np.logaddexp(union, intersection)),
        "note": "Highlighted path explains the union term; other topic entities affect the intersection term.",
    }
