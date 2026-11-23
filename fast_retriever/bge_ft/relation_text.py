"""Canonical text formatting for Freebase relation identifiers.

Rich/audit artifacts always retain the raw relation identifier.  Only model
inputs use one of these formatters, and the selected formatter is recorded in
every run manifest and score cache.
"""

from collections.abc import Callable


def raw_relation(relation: str) -> str:
    return relation


def split_relation(relation: str) -> str:
    return relation.replace(".", " ").replace("_", " ")


def natural_relation(relation: str) -> str:
    parts = relation.split(".")
    if len(parts) >= 3:
        return parts[-1].replace("_", " ")
    return split_relation(relation)


FORMATTERS: dict[str, Callable[[str], str]] = {
    "raw": raw_relation,
    "split": split_relation,
    "natural": natural_relation,
}


def format_relation(relation: str, format_name: str) -> str:
    try:
        formatter = FORMATTERS[format_name]
    except KeyError as error:
        choices = ", ".join(sorted(FORMATTERS))
        raise ValueError(
            f"Unknown relation format {format_name!r}; choose one of: {choices}"
        ) from error
    return formatter(relation)
