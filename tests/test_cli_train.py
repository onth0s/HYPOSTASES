"""Tests for the 'hypostases train' CLI subcommand.

Validates argument parsing, resume routing, and domain validation
without executing an actual training run.
"""

from __future__ import annotations

import argparse
import sys

import pytest

from hypostases.cli.train import _cli_handler, add_subparser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    subparsers = p.add_subparsers(dest="command")
    add_subparser(subparsers)
    return p


# ---------------------------------------------------------------------------
# Argument parsing correctness
# ---------------------------------------------------------------------------


def test_train_chess_new_defaults(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess"])
    assert args.chess is True
    assert args.resume is None
    assert args.resume_n is None
    assert args.gens == 20
    assert args.games == 15
    assert args.seed == 42
    assert args.workers == 20
    assert args.quiet is False
    assert args.log_games is False


def test_train_chess_custom_gens(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess", "--gens", "50"])
    assert args.gens == 50


def test_train_chess_resume_defaults_n_to_none(parser: argparse.ArgumentParser) -> None:
    # "resume" positional but no N
    args = parser.parse_args(["train", "--chess", "resume"])
    assert args.resume == "resume"
    assert args.resume_n is None


def test_train_chess_resume_with_n(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess", "resume", "2"])
    assert args.resume == "resume"
    assert args.resume_n == 2


def test_train_chess_quiet_flag(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess", "--quiet"])
    assert args.quiet is True


def test_train_chess_log_games_flag(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess", "--log-games"])
    assert args.log_games is True


def test_train_chess_run_dir_flag(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["train", "--chess", "resume", "--run-dir", "exports/runs/run_X"])
    assert args.run_dir == "exports/runs/run_X"


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def test_cli_handler_no_domain_exits(
    parser: argparse.ArgumentParser, capsys: pytest.CaptureFixture
) -> None:
    args = parser.parse_args(["train"])
    # No --chess flag → should sys.exit(1)
    with pytest.raises(SystemExit) as exc_info:
        _cli_handler(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Bad positional argument
# ---------------------------------------------------------------------------


def test_cli_handler_bad_positional_exits(
    parser: argparse.ArgumentParser, capsys: pytest.CaptureFixture
) -> None:
    args = parser.parse_args(["train", "--chess", "foobar"])
    with pytest.raises(SystemExit) as exc_info:
        _cli_handler(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Subparser registration
# ---------------------------------------------------------------------------


def test_train_subcommand_registered() -> None:
    """The main CLI dispatcher must include 'train' as a known subcommand."""
    # Patching sys.argv to request --help so main() exits cleanly without running
    import io
    from contextlib import redirect_stdout

    from hypostases.cli.main import main

    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
        sys.argv = ["hypostases", "--help"]
        main()

    output = buf.getvalue()
    assert "train" in output
