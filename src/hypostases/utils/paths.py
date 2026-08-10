"""HYPOSTASES Path utilities."""

from __future__ import annotations

from pathlib import Path


def find_project_root() -> Path:
    """Walk parents from the current file to find the project root containing pyproject.toml."""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").is_file() or (parent / "spec").is_dir():
            return parent
    return current_file.parent
