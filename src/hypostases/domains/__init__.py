"""Domain abstraction module for HYPOSTASES environment integration."""

from hypostases.domains.base import Domain
from hypostases.domains.registry import DomainRegistry
from hypostases.domains.toy_nim import NimState, SimpleNimDomain, nim_evaluator

__all__ = [
    "Domain",
    "DomainRegistry",
    "NimState",
    "SimpleNimDomain",
    "nim_evaluator",
]
