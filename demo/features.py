"""Question/relation features copied from the GTSQA checkpoint notebook."""

import re
import string
from functools import lru_cache

STOPWORDS = set("""what which who whom where when why how a an the and or but if because as until while
of at by for with about against between into through during before after above below to from up down
in out on off over under again further then once here there all any both each few more most other
some such no nor not only own same so than too very s t can will just don should now is was were are
has have had do does did""".split())

SYNONYM_GROUPS = [
    set(group.split()) for group in (
        "speak spoken speech language languages", "born birth originated origin",
        "died death deceased burial grave", "written write author book writer subjects",
        "acted actor actress starring play plays cast role roles", "produced producer",
        "directed director", "created creator creative", "founded founder founding",
        "located location place containedby contains adjoins adjoin",
        "member membership participant participated", "owned owner proprietor",
        "married marriage spouse wife husband",
        "parent parents children child son daughter father mother",
        "nationality citizen country nation",
    )
]
SUFFIX = re.compile(r"\s*\((?:Q|P)\d+\)\s*$")


def clean_name(value: str) -> str:
    return SUFFIX.sub("", value).strip()


def clean_text(value: str) -> list[str]:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return [word for word in value.split() if word not in STOPWORDS]


@lru_cache(maxsize=500_000)
def token_similarity(question_token: str, relation_token: str) -> float:
    if question_token == relation_token:
        return 1.0
    for group in SYNONYM_GROUPS:
        if question_token in group and relation_token in group:
            return 0.9
    length = min(len(question_token), len(relation_token))
    if length >= 4:
        if question_token[:length] == relation_token[:length]:
            return 0.9
        if question_token[:4] == relation_token[:4]:
            return 0.8
    return 0.0


@lru_cache(maxsize=None)
def relation_tokens(name: str) -> tuple[tuple[str, float], ...]:
    parts = name.split(".")
    result = []
    for i, part in enumerate(parts):
        weight = 1.0 if i == len(parts) - 1 else (0.5 if i == len(parts) - 2 else 0.2)
        for token in part.split("_"):
            token = token.lower()
            if len(token) >= 2 and token not in STOPWORDS:
                result.append((token, weight))
    return tuple(result)


def lexical_score(question_tokens: list[str], relation: str) -> float:
    weighted = relation_tokens(relation)
    total_weight = sum(weight for _, weight in weighted)
    if not total_weight:
        return 0.0
    return sum(
        max((token_similarity(q, token) for q in question_tokens), default=0.0) * weight
        for token, weight in weighted
    ) / total_weight


def lexical_features(question: str, relations: list[str]) -> list[float]:
    tokens = clean_text(question)
    return [lexical_score(tokens, relation) for relation in relations]


def suggest_topics(question: str, names: list[str], limit: int = 5) -> list[int]:
    """Return unambiguous longest entity-label matches in this question's graph."""
    lowered = question.casefold()
    matches = []
    for i, name in enumerate(names):
        label = clean_name(name)
        if len(label) < 3:
            continue
        label = label.casefold()
        if label not in lowered:
            continue
        pattern = rf"(?<!\w){re.escape(label)}(?!\w)"
        for match in re.finditer(pattern, lowered):
            matches.append((match.start(), match.end(), -len(label), i))
    matches.sort(key=lambda item: (item[2], item[0], item[3]))
    accepted: list[tuple[int, int, int]] = []
    for start, end, _, index in matches:
        if all(end <= left or start >= right for left, right, _ in accepted):
            accepted.append((start, end, index))
        if len(accepted) >= limit:
            break
    return [index for _, _, index in sorted(accepted)]
