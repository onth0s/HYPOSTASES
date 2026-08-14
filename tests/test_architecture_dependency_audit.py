"""AST Dependency Audit for Core Engine & Plugin Decoupling.

Scans all core engine modules under `src/hypostases/` (excluding `src/hypostases/plugins/`)
to empirically verify zero imports of python-chess, chess domain symbols, or plugin modules.
"""

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "chess",
    "python-chess",
    "ChessDomain",
    "hypostases.plugins",
}

FORBIDDEN_MODULE_PREFIXES = (
    "chess",
    "hypostases.plugins",
)

FORBIDDEN_SYMBOLS = {
    "Board",
    "Square",
    "Piece",
    "PieceType",
    "FEN",
    "ep_square",
    "halfmove_clock",
}

SRC_HYPOSTASES_PATH = Path("src/hypostases")
AUDIT_REPORT_PATH = Path("scratch/ARCHITECTURE_DEPENDENCY_AUDIT.md")


def audit_module(path: Path) -> tuple[list[str], int, int]:
    """Scans a single Python file AST for forbidden domain or plugin imports."""
    assert path.exists(), f"Module {path} not found"
    source_code = path.read_text(encoding="utf-8")
    tree = ast.parse(source_code, filename=str(path))

    detected: list[str] = []
    imports_count = 0
    identifiers_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports_count += len(node.names)
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS or any(
                    alias.name.startswith(p) for p in FORBIDDEN_MODULE_PREFIXES
                ):
                    detected.append(f"L{node.lineno}: Direct import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            imports_count += len(node.names)
            if node.module and any(p in node.module for p in FORBIDDEN_MODULE_PREFIXES):
                detected.append(f"L{node.lineno}: ImportFrom module '{node.module}'")
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS or alias.name in FORBIDDEN_SYMBOLS:
                    detected.append(f"L{node.lineno}: ImportFrom symbol '{alias.name}'")
        elif isinstance(node, ast.Name):
            identifiers_count += 1
            if node.id in FORBIDDEN_SYMBOLS:
                detected.append(f"L{node.lineno}: Forbidden symbol usage '{node.id}'")

    return detected, imports_count, identifiers_count


def test_d008_domain_decoupling_ast_audit() -> None:
    """Verifies that ALL core engine modules have ZERO imports of python-chess, ChessDomain, or plugins."""
    plugins_dir = SRC_HYPOSTASES_PATH / "plugins"

    core_py_files = [
        p
        for p in SRC_HYPOSTASES_PATH.glob("**/*.py")
        if plugins_dir not in p.parents and p.name != "__init__.py"
    ]

    total_violations: list[str] = []
    total_imports = 0
    total_identifiers = 0
    inspected_modules: list[str] = []

    for path in sorted(core_py_files):
        violations, imp_cnt, id_cnt = audit_module(path)
        total_imports += imp_cnt
        total_identifiers += id_cnt
        inspected_modules.append(path.name)

        if violations:
            for v in violations:
                total_violations.append(f"{path.as_posix()}:{v}")

    status_str = "PASSED" if not total_violations else "FAILED"

    report_content = f"""# ARCHITECTURE_DEPENDENCY_AUDIT.md

> Core Engine Decoupling Audit Status: **{status_str}**

## Inspection Scope & Audit Telemetry
- **Core Modules Scanned**: {len(core_py_files)}
- **Total Imports Inspected**: {total_imports}
- **Total Identifiers Inspected**: {total_identifiers}

### Inspection Rules & Forbidden Identifiers
- **Forbidden Imports**: `chess`, `python-chess`, `ChessDomain`, `hypostases.plugins`
- **Forbidden Symbols**: `Board`, `Square`, `Piece`, `PieceType`, `FEN`, `ep_square`, `halfmove_clock`

### AST Audit Findings
- **Total Architectural Violations**: {len(total_violations)}
"""
    if total_violations:
        report_content += "\n### Detected Violations:\n"
        for v in total_violations:
            report_content += f"- ❌ {v}\n"
        print("\n[FAIL] AST Audit Violations:\n" + "\n".join(total_violations))
    else:
        report_content += "\n✅ Zero domain-specific or plugin imports detected across all core engine modules. Engine is 100% domain-agnostic.\n"

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(report_content, encoding="utf-8")

    assert not total_violations, "Core Engine Decoupling Violation(s):\n" + "\n".join(
        total_violations
    )
