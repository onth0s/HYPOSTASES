"""Compile all formal verification logs (Lean 4 theorem proofs + Pytest formal math tests) into a unified report."""

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def find_elan_path() -> str:
    user_elan = Path.home() / ".elan" / "bin"
    current_path = os.environ.get("PATH", "")
    if user_elan.exists() and str(user_elan) not in current_path:
        return f"{user_elan};{current_path}"
    return current_path


def run_lean_verification(repo_root: Path) -> tuple[int, str]:
    formal_dir = repo_root / "formal"
    if not formal_dir.exists():
        return 1, "Error: formal/ directory not found."

    env = os.environ.copy()
    env["PATH"] = find_elan_path()

    lake_bin = shutil.which("lake", path=env["PATH"])
    if not lake_bin:
        return (
            1,
            "Error: 'lake' executable not found on PATH. Please ensure Lean 4 (elan) is installed.",
        )

    cmd = [lake_bin, "build"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(formal_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + "\n" + proc.stderr
        return proc.returncode, output.strip()
    except Exception as e:
        return 1, f"Execution failed: {e}"


def run_pytest_formal_math(repo_root: Path) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "tests/formal_math/", "-v", "--tb=short"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        output = proc.stdout + "\n" + proc.stderr
        return proc.returncode, output.strip()
    except Exception as e:
        return 1, f"Execution failed: {e}"


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_filename = logs_dir / f"formal_verification_{timestamp}.log"
    latest_symlink = logs_dir / "formal_verification_latest.log"

    console.print(
        Panel(
            "[bold cyan]HYPOSTASES — Dual Formal Verification Collector[/bold cyan]", expand=False
        )
    )

    console.print(
        "[yellow]1/2 Running Lean 4 / Mathlib4 theorem compilation (`lake build`)...[/yellow]"
    )
    lean_code, lean_out = run_lean_verification(repo_root)

    console.print(
        "[yellow]2/2 Running Pytest formal mathematical test suite (`tests/formal_math/`)...[/yellow]"
    )
    pytest_code, pytest_out = run_pytest_formal_math(repo_root)

    lean_status = "PASSED" if lean_code == 0 else "FAILED"
    pytest_status = "PASSED" if pytest_code == 0 else "FAILED"

    # Build compiled log content
    report = []
    report.append("=" * 80)
    report.append("HYPOSTASES FORMAL VERIFICATION CONSOLIDATED REPORT")
    report.append(f"Generated at (UTC): {timestamp}")
    report.append(f"Lean 4 Status: {lean_status}")
    report.append(f"Pytest Formal Math Status: {pytest_status}")
    report.append("=" * 80)
    report.append("\n" + "#" * 40)
    report.append("# SECTION 1: LEAN 4 INTERACTIVE THEOREM PROOFS")
    report.append("#" * 40 + "\n")
    report.append(lean_out)
    report.append("\n" + "#" * 40)
    report.append("# SECTION 2: PYTEST MODULAR FORMAL MATH TEST SUITE")
    report.append("#" * 40 + "\n")
    report.append(pytest_out)
    report.append("\n" + "=" * 80)
    report.append("END OF FORMAL VERIFICATION REPORT")
    report.append("=" * 80 + "\n")

    full_text = "\n".join(report)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(latest_symlink, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Format user table
    table = Table(
        title="Formal Verification Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Verification Tier", style="dim")
    table.add_column("Engine / Toolchain")
    table.add_column("Exit Code")
    table.add_column("Status", justify="center")

    table.add_row(
        "Tier 1: Mechanized Theorems",
        "Lean 4 (v4.13.0) + Mathlib4",
        str(lean_code),
        "[green]PASSED[/green]" if lean_code == 0 else "[red]FAILED[/red]",
    )
    table.add_row(
        "Tier 2: Asymptotic / Empirical Math",
        "Pytest (tests/formal_math/)",
        str(pytest_code),
        "[green]PASSED[/green]" if pytest_code == 0 else "[red]FAILED[/red]",
    )

    console.print(table)
    console.print(f"[bold green]Report successfully written to:[/bold green] {output_filename}")
    console.print(f"[bold green]Latest snapshot written to:[/bold green] {latest_symlink}")


if __name__ == "__main__":
    main()
