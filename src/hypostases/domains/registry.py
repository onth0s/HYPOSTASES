"""HYPOSTASES Pluggable Domain Registry.

Provides explicit registration and dynamic lookup for Domain Protocol implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from hypostases.domains.base import Domain

T_Domain = TypeVar("T_Domain", bound=Domain)


class DomainRegistry:
    """Central registry for pluggable domain implementations and trainers."""

    _registry: ClassVar[dict[str, type[Domain] | Callable[..., Domain]]] = {}
    # Maps domain name -> training callable (kwargs -> None / return value ignored)
    _trainer_registry: ClassVar[dict[str, Callable[..., None]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[T_Domain]], type[T_Domain]]:
        """Decorator to register a Domain implementation class."""

        def decorator(domain_cls: type[T_Domain]) -> type[T_Domain]:
            cls._registry[name.lower()] = domain_cls
            return domain_cls

        return decorator

    @classmethod
    def register_factory(cls, name: str, factory: Callable[..., Domain]) -> None:
        """Explicitly register a domain factory function."""
        cls._registry[name.lower()] = factory

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Domain:
        """Instantiate and return a registered domain instance by name.

        Args:
            name: The registered domain identifier.
            **kwargs: Initialization keyword arguments passed to domain constructor.

        Returns:
            An instance conforming to the Domain protocol.

        Raises:
            KeyError: If domain name is not registered.
        """
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "none"
            raise KeyError(
                f"Domain '{name}' is not registered. Available registered domains: [{available}]"
            )
        factory = cls._registry[key]
        return factory(**kwargs)

    @classmethod
    def list_domains(cls) -> list[str]:
        """Returns list of registered domain names."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregisters a domain by name if present."""
        cls._registry.pop(name.lower(), None)

    @classmethod
    def clear(cls) -> None:
        """Clears all registered domains (primarily for testing isolation)."""
        cls._registry.clear()

    # ------------------------------------------------------------------
    # Trainer registry
    # ------------------------------------------------------------------

    @classmethod
    def register_trainer(cls, name: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
        """Decorator to register a domain training callable.

        The decorated callable receives arbitrary kwargs forwarded from the CLI
        train dispatcher.  Return values are ignored by the dispatcher.
        """

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            cls._trainer_registry[name.lower()] = fn
            return fn

        return decorator

    @classmethod
    def get_trainer(cls, name: str) -> Callable[..., None]:
        """Returns the registered training callable for *name*.

        Raises:
            KeyError: If no trainer is registered for the domain.
        """
        key = name.lower()
        if key not in cls._trainer_registry:
            available = ", ".join(sorted(cls._trainer_registry.keys())) or "none"
            raise KeyError(
                f"No trainer registered for domain '{name}'. Available trainers: [{available}]"
            )
        return cls._trainer_registry[key]

    @classmethod
    def list_trainers(cls) -> list[str]:
        """Returns sorted list of registered trainer domain names."""
        return sorted(cls._trainer_registry.keys())

    @classmethod
    def clear_trainers(cls) -> None:
        """Clears all registered trainers (primarily for testing isolation)."""
        cls._trainer_registry.clear()
