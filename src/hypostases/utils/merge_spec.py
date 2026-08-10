"""HYPOSTASES CLI Utils — Merges specification markdown parts into a single document."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hypostases.utils.paths import find_project_root


def strip_yaml_frontmatter(content: str) -> str:
    """Strips YAML frontmatter from a markdown string, ignoring leading empty lines."""
    content = content.replace("\r\n", "\n")
    lines = content.split("\n")
    start_idx = 0
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1

    if start_idx < len(lines) and lines[start_idx].strip() == "---":
        try:
            end_idx = lines.index("---", start_idx + 1)
            # Skip the trailing newline after the frontmatter if present
            stripped = "\n".join(lines[end_idx + 1 :])
            # If the original content had a newline right after ---, avoid leading newline
            if stripped.startswith("\n"):
                stripped = stripped[1:]
            return stripped
        except ValueError:
            pass
    return content


def main(args_list: list[str] | None = None):
    """Main function to merge HYPOSTASES specification parts."""
    parser = argparse.ArgumentParser(description="Merge HYPOSTASES specification parts.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be merged without writing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to output the merged file. Defaults to project root.",
    )

    args = parser.parse_args(args_list)

    # Paths — resolve repository root using paths utility
    project_root = find_project_root()
    spec_dir = project_root / "spec"
    output_dir = Path(args.output_dir) if args.output_dir else project_root

    if not spec_dir.exists():
        print(f"Error: Spec directory not found at {spec_dir}")
        return

    md_files = sorted(
        [f for f in spec_dir.iterdir() if f.suffix == ".md" and f.is_file()], key=lambda x: x.name
    )

    if not md_files:
        print(f"No markdown files found in {spec_dir}")
        return

    merged_content = ""
    total_lines = 0

    for fpath in md_files:
        content = fpath.read_text(encoding="utf-8")
        stripped_content = strip_yaml_frontmatter(content)

        # Ensure proper separation
        if merged_content and not merged_content.endswith("\n"):
            merged_content += "\n"

        merged_content += stripped_content

    total_lines = merged_content.count("\n") + (1 if merged_content else 0)

    epoch_seconds = int(time.time())
    output_filename = f"HYPOSTASES_{epoch_seconds}.md"
    output_path = output_dir / output_filename

    if args.dry_run:
        print(f"[DRY RUN] Would merge {len(md_files)} files into {output_path}")
        print(f"[DRY RUN] Output filename: {output_filename}")
        print(f"[DRY RUN] Total line count: {total_lines}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(merged_content, encoding="utf-8")
        print(f"Merged {len(md_files)} parts.")
        print(f"Output filename: {output_filename}")
        print(f"Total line count: {total_lines}")


if __name__ == "__main__":
    main()
