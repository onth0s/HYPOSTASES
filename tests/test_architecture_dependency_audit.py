"""AST Dependency Audit for Generic Search Engine Decoupling.

Spec Ref: scratch/DISSONANCES.md D-008 & D-013.
Generates scratch/ARCHITECTURE_DEPENDENCY_AUDIT.md artifact.
"""

import ast
from pathlib import Path
import pytest

FORBIDDEN_IMPORTS = {
    "chess",
    "python-chess",
    "ChessDomain",
}

FORBIDDEN_SYMBOLS = {
    "Board",
    "Move",
    "Square",
    "Piece",
    "PieceType",
    "FEN",
    "SAN",
    "ep_square",
    "halfmove_clock",
}

SEARCH_MODULE_PATH = Path("src/hypostases/world_model/alphabeta_search.py")
TELEMETRY_MODULE_PATH = Path("src/hypostases/world_model/telemetry.py")
AUDIT_REPORT_PATH = Path("scratch/HYPOSTASES_NNUE_CONVERGENCE_AUDIT.md")


def audit_module(path: Path) -> tuple[list[str], int, int]:
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
                if alias.name in FORBIDDEN_IMPORTS or alias.name in FORBIDDEN_SYMBOLS:
                    detected.append(f"L{node.lineno}: Direct import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            imports_count += len(node.names)
            if node.module and any(f in node.module for f in ("chess", "ChessDomain")):
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
    """Verifies that AlphaBetaSearch and Telemetry have ZERO imports of python-chess or ChessDomain."""
    search_violations, s_imp, s_id = audit_module(SEARCH_MODULE_PATH)
    telemetry_violations, t_imp, t_id = audit_module(TELEMETRY_MODULE_PATH)

    total_violations = search_violations + telemetry_violations
    status_str = "PASSED" if not total_violations else "FAILED"

    report_content = f"""# ARCHITECTURE_DEPENDENCY_AUDIT.md

> Domain Decoupling Audit Status: **{status_str}**
> Spec Ref: DISSONANCES.md D-008 & D-013

## Inspection Scope & Audit Telemetry
- **Modules Scanned**: 2 (`alphabeta_search.py`, `telemetry.py`)
- **Total Imports Inspected**: {s_imp + t_imp}
- **Total Identifiers Inspected**: {s_id + t_id}

### Target Modules Inspected
- [`alphabeta_search.py`](file:///{SEARCH_MODULE_PATH.resolve()}) ({s_imp} imports, {s_id} identifiers)
- [`telemetry.py`](file:///{TELEMETRY_MODULE_PATH.resolve()}) ({t_imp} imports, {t_id} identifiers)

### Inspection Rules & Forbidden Identifiers
- **Forbidden Imports**: `chess`, `python-chess`, `ChessDomain`
- **Forbidden Symbols**: `Board`, `Move`, `Square`, `Piece`, `PieceType`, `FEN`, `SAN`, `ep_square`, `halfmove_clock`

### AST Audit Findings
- **`alphabeta_search.py` Violations**: {len(search_violations)}
- **`telemetry.py` Violations**: {len(telemetry_violations)}
"""
    if total_violations:
        report_content += "\n### Detected Violations:\n"
        for v in total_violations:
            report_content += f"- ❌ {v}\n"
    else:
        report_content += "\n✅ Zero domain-specific imports or symbols detected. Search & Telemetry engines are 100% domain-agnostic.\n"

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(report_content, encoding="utf-8")

    assert not total_violations, f"D-008 Domain Decoupling Violation(s): {total_violations}"
