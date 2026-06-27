"""Causal attribution: DAG/confounding sets, backdoor adjustment, EI."""

from .graph import CONFOUNDING_SETS, confounding_set, describe_graph
from .backdoor import (
    backdoor_adjust,
    do_distribution,
    experimental_do,
)
from .ei import (
    effective_information,
    ei_from_do_table,
    EIResult,
)

__all__ = [
    "CONFOUNDING_SETS",
    "confounding_set",
    "describe_graph",
    "backdoor_adjust",
    "do_distribution",
    "experimental_do",
    "effective_information",
    "ei_from_do_table",
    "EIResult",
]
