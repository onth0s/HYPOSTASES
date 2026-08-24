"""Compile all formal verification logs (Lean 4 theorem proofs + Pytest formal math tests) with rigorous provenance and sanitized paths."""

import datetime
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def sanitize_text(text: str, repo_root: Path) -> str:
    """Obscure user directory paths, machine usernames, and environment-specific prefixes from strings."""
    if not text:
        return text

    user_home = str(Path.home())
    user_name = Path.home().name
    repo_root_str = str(repo_root)

    # Replace repository absolute path with relative <HYPOSTASES_ROOT>
    text = text.replace(repo_root_str, "<HYPOSTASES_ROOT>")
    text = text.replace(repo_root_str.replace("\\", "/"), "<HYPOSTASES_ROOT>")

    # Replace user profile / home directory with <USER_HOME>
    text = text.replace(user_home, "<USER_HOME>")
    text = text.replace(user_home.replace("\\", "/"), "<USER_HOME>")

    # Catch any remaining Windows User directory patterns (e.g. C:\Users\<Username>\...)
    text = re.sub(
        r"[A-Za-z]:\\[Uu]sers\\[^\\]+",
        r"<USER_HOME>",
        text,
    )
    text = re.sub(
        r"[A-Za-z]:/[Uu]sers/[^/]+",
        r"<USER_HOME>",
        text,
    )

    # Obscure machine username if still present
    if user_name:
        text = re.sub(re.escape(user_name), "<USER>", text, flags=re.IGNORECASE)

    return text


def get_git_provenance(repo_root: Path) -> dict[str, str]:
    """Retrieve git provenance metadata for the exact repository state under test."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit_hash = "UNKNOWN_OR_NOT_A_GIT_REPO"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        branch = "UNKNOWN"

    try:
        dirty_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        is_dirty = "DIRTY (uncommitted changes present)" if dirty_status else "CLEAN"
    except Exception:
        is_dirty = "UNKNOWN"

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "working_tree": is_dirty,
    }


def find_elan_path() -> str:
    user_elan = Path.home() / ".elan" / "bin"
    current_path = os.environ.get("PATH", "")
    if user_elan.exists() and str(user_elan) not in current_path:
        return f"{user_elan};{current_path}"
    return current_path


def run_lean_verification(repo_root: Path) -> dict:
    """Execute Lean 4 theorem compilation live with exact invocation provenance."""
    formal_dir = repo_root / "formal"
    if not formal_dir.exists():
        raise RuntimeError(f"Formal verification directory '{formal_dir}' does not exist.")

    env = os.environ.copy()
    env["PATH"] = find_elan_path()

    lake_bin = shutil.which("lake", path=env["PATH"])
    if not lake_bin:
        raise RuntimeError(
            "Executable 'lake' not found on PATH. Please ensure Lean 4 (elan) is installed."
        )

    cmd = [lake_bin, "build"]
    cmd_str = " ".join(cmd)
    start_time = datetime.datetime.now(datetime.timezone.utc)

    proc = subprocess.run(
        cmd,
        cwd=str(formal_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration_s = (end_time - start_time).total_seconds()
    output = (proc.stdout + "\n" + proc.stderr).strip()

    # Discover and inspect Hypostases project-specific formal source files
    hypostases_formal_files = [f for f in formal_dir.glob("**/*.lean") if ".lake" not in f.parts]
    hypostases_formal_files.sort()

    file_metadata = []
    for f in hypostases_formal_files:
        rel_path = f.relative_to(repo_root)
        mtime = datetime.datetime.fromtimestamp(
            f.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat()
        file_metadata.append(
            {
                "file": str(rel_path),
                "last_modified": mtime,
                "size_bytes": f.stat().st_size,
            }
        )

    return {
        "command": sanitize_text(cmd_str, repo_root),
        "cwd": sanitize_text(str(formal_dir), repo_root),
        "exit_code": proc.returncode,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_s,
        "output": sanitize_text(output, repo_root),
        "files": file_metadata,
    }


def run_pytest_formal_math(repo_root: Path, junit_xml_path: Path) -> dict:
    """Execute pytest formal math test suite with structured JUnit XML telemetry."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/formal_math/",
        "-v",
        "--tb=short",
        f"--junitxml={junit_xml_path}",
    ]
    cmd_str = " ".join(cmd)
    start_time = datetime.datetime.now(datetime.timezone.utc)

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    end_time = datetime.datetime.now(datetime.timezone.utc)
    duration_s = (end_time - start_time).total_seconds()
    output = (proc.stdout + "\n" + proc.stderr).strip()

    if not junit_xml_path.exists():
        raise RuntimeError(
            f"Pytest did not produce expected JUnit XML report at '{junit_xml_path}'. "
            f"Process exited with code {proc.returncode}."
        )

    # Parse per-test-suite / per-test timestamps & durations from JUnit XML
    tree = ET.parse(junit_xml_path)
    root = tree.getroot()

    testsuites_data = []
    # root can be <testsuites> or <testsuite>
    suite_elements = root.findall("testsuite") if root.tag == "testsuites" else [root]

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    for suite in suite_elements:
        suite_name = suite.attrib.get("name", "pytest")
        suite_tests = int(suite.attrib.get("tests", 0))
        suite_failures = int(suite.attrib.get("failures", 0))
        suite_errors = int(suite.attrib.get("errors", 0))
        suite_skipped = int(suite.attrib.get("skipped", 0))
        suite_time = float(suite.attrib.get("time", 0.0))
        suite_timestamp = suite.attrib.get("timestamp", start_time.isoformat())

        total_tests += suite_tests
        total_failures += suite_failures
        total_errors += suite_errors
        total_skipped += suite_skipped

        testcases = []
        for case in suite.findall("testcase"):
            tc_name = case.attrib.get("name", "")
            tc_classname = case.attrib.get("classname", "")
            tc_file = sanitize_text(case.attrib.get("file", ""), repo_root)
            tc_time = float(case.attrib.get("time", 0.0))
            is_failed = case.find("failure") is not None
            is_error = case.find("error") is not None
            is_skipped = case.find("skipped") is not None

            status = "PASSED"
            if is_failed:
                status = "FAILED"
            elif is_error:
                status = "ERROR"
            elif is_skipped:
                status = "SKIPPED"

            testcases.append(
                {
                    "classname": tc_classname,
                    "name": tc_name,
                    "file": tc_file,
                    "time_seconds": tc_time,
                    "status": status,
                }
            )

        testsuites_data.append(
            {
                "name": suite_name,
                "tests": suite_tests,
                "failures": suite_failures,
                "errors": suite_errors,
                "skipped": suite_skipped,
                "duration_seconds": suite_time,
                "timestamp": suite_timestamp,
                "testcases": testcases,
            }
        )

    return {
        "command": sanitize_text(cmd_str, repo_root),
        "cwd": sanitize_text(str(repo_root), repo_root),
        "exit_code": proc.returncode,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_s,
        "output": sanitize_text(output, repo_root),
        "summary": {
            "total_tests": total_tests,
            "failures": total_failures,
            "errors": total_errors,
            "skipped": total_skipped,
        },
        "suites": testsuites_data,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    timestamp_str = utc_now.strftime("%Y-%m-%dT%H-%M-%SZ")
    output_filename = logs_dir / f"formal_verification_{timestamp_str}.log"
    latest_symlink = logs_dir / "formal_verification_latest.log"
    junit_xml_path = logs_dir / f"pytest_formal_math_{timestamp_str}.xml"

    console.print(
        Panel(
            "[bold cyan]HYPOSTASES — Formal Verification Collector with Full Live Provenance (Sanitized)[/bold cyan]",
            expand=False,
        )
    )

    console.print("[dim]Fetching repository state & git provenance...[/dim]")
    git_info = get_git_provenance(repo_root)

    console.print("[yellow]1/2 Running Lean 4 theorem compilation live (`lake build`)...[/yellow]")
    lean_results = run_lean_verification(repo_root)

    console.print(
        "[yellow]2/2 Running Pytest formal math test suite live (`tests/formal_math/`)...[/yellow]"
    )
    pytest_results = run_pytest_formal_math(repo_root, junit_xml_path)

    # Validate execution completion
    if lean_results["exit_code"] != 0 and not lean_results["output"]:
        raise RuntimeError("Lean 4 build failed without producing log output.")

    # Build exhaustive structured provenance report
    report = []
    sep_major = "=" * 90
    sep_minor = "-" * 90

    report.append(sep_major)
    report.append("HYPOSTASES CONSOLIDATED FORMAL VERIFICATION & PROVENANCE REPORT")
    report.append(sep_major)
    report.append(f"Report Generated (UTC):     {utc_now.isoformat()}")
    report.append(f"Git Commit Hash:            {git_info['commit_hash']}")
    report.append(f"Git Active Branch:          {git_info['branch']}")
    report.append(f"Working Tree Status:        {git_info['working_tree']}")
    report.append(f"Platform / Python:          {sys.platform} | Python {sys.version.split()[0]}")
    report.append(sep_minor)
    report.append("EXECUTIVE VERIFICATION SUMMARY")
    report.append(sep_minor)
    report.append(
        f"• Tier 1 [Lean 4 / Mathlib4]  Exit Code: {lean_results['exit_code']}  Status: {'PASSED' if lean_results['exit_code'] == 0 else 'FAILED'}  (Duration: {lean_results['duration_seconds']:.2f}s)"
    )
    report.append(
        f"• Tier 2 [Pytest Formal Math] Exit Code: {pytest_results['exit_code']}  Status: {'PASSED' if pytest_results['exit_code'] == 0 else 'FAILED'}  (Duration: {pytest_results['duration_seconds']:.2f}s, {pytest_results['summary']['total_tests']} tests)"
    )
    report.append(sep_major)

    report.append("\n" + "#" * 45)
    report.append("# SECTION 1: LEAN 4 THEOREM PROVER PROVENANCE")
    report.append("#" * 45)
    report.append(f"Invocation Command:    {lean_results['command']}")
    report.append(f"Working Directory:     {lean_results['cwd']}")
    report.append(f"Execution Start (UTC): {lean_results['start_time']}")
    report.append(f"Execution End (UTC):   {lean_results['end_time']}")
    report.append(f"Execution Duration:    {lean_results['duration_seconds']:.3f} s")
    report.append(f"Process Exit Code:     {lean_results['exit_code']}")
    report.append("\nFormal Source Files & Timestamps under Test:")
    for fmeta in lean_results["files"]:
        report.append(
            f"  - {fmeta['file']} (Modified: {fmeta['last_modified']}, {fmeta['size_bytes']} bytes)"
        )
    report.append("\nRaw Lean 4 Lake Build Output:")
    report.append(
        lean_results["output"] if lean_results["output"] else "(Clean build with 0 warnings/errors)"
    )

    report.append("\n" + "#" * 45)
    report.append("# SECTION 2: PYTEST FORMAL MATH SUITE PROVENANCE")
    report.append("#" * 45)
    report.append(f"Invocation Command:    {pytest_results['command']}")
    report.append(f"Working Directory:     {pytest_results['cwd']}")
    report.append(f"Execution Start (UTC): {pytest_results['start_time']}")
    report.append(f"Execution End (UTC):   {pytest_results['end_time']}")
    report.append(f"Execution Duration:    {pytest_results['duration_seconds']:.3f} s")
    report.append(f"Process Exit Code:     {pytest_results['exit_code']}")
    report.append(f"Total Tests Executed:  {pytest_results['summary']['total_tests']}")
    report.append(
        f"Failures / Errors:     {pytest_results['summary']['failures']} / {pytest_results['summary']['errors']}"
    )
    report.append(f"Skipped Tests:         {pytest_results['summary']['skipped']}")
    report.append("\nPer-Suite Execution Telemetry & Timestamps:")
    for suite in pytest_results["suites"]:
        report.append(f"  • Suite: {suite['name']}")
        report.append(
            f"    - Timestamp: {suite['timestamp']} | Duration: {suite['duration_seconds']:.3f}s | Tests: {suite['tests']} | Failures: {suite['failures']}"
        )
        for tc in suite["testcases"]:
            report.append(f"      [{tc['status']}] {tc['name']} ({tc['time_seconds']:.4f}s)")
    report.append("\nRaw Pytest Console Output:")
    report.append(pytest_results["output"])

    report.append("\n" + sep_major)
    report.append("END OF FORMAL VERIFICATION AND PROVENANCE REPORT")
    report.append(sep_major + "\n")

    full_text = "\n".join(report)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(latest_symlink, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Console Presentation Table
    table = Table(
        title="Live Verification & Provenance Summary (Sanitized)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Tier", style="cyan")
    table.add_column("Exact Invocation")
    table.add_column("Commit (HEAD)", style="dim")
    table.add_column("Exit Code", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        "Tier 1 (Lean 4)",
        lean_results["command"],
        git_info["commit_hash"][:8],
        str(lean_results["exit_code"]),
        f"{lean_results['duration_seconds']:.2f}s",
        "[green]PASSED[/green]" if lean_results["exit_code"] == 0 else "[red]FAILED[/red]",
    )
    table.add_row(
        "Tier 2 (Pytest)",
        pytest_results["command"],
        git_info["commit_hash"][:8],
        str(pytest_results["exit_code"]),
        f"{pytest_results['duration_seconds']:.2f}s",
        "[green]PASSED[/green]" if pytest_results["exit_code"] == 0 else "[red]FAILED[/red]",
    )

    console.print(table)
    console.print(
        f"[bold green]Consolidated provenance log written to:[/bold green] {sanitize_text(str(output_filename), repo_root)}"
    )
    console.print(
        f"[bold green]Latest report snapshot updated at:[/bold green] {sanitize_text(str(latest_symlink), repo_root)}"
    )

    # Fail loudly if any test or compilation step failed
    if lean_results["exit_code"] != 0 or pytest_results["exit_code"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
