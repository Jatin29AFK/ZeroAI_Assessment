"""Validate rectified article output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rectification_system import load_article_records


def validate_outputs(expected_count: int | None) -> dict:
    records = load_article_records()
    expected_ids = {record.rectified_file.name for record in records}
    output_dir = Path("rectified_articles")
    output_files = sorted(output_dir.glob("article_*.txt")) if output_dir.exists() else []
    output_ids = {path.name for path in output_files}

    empty: list[str] = []
    bad_start: list[str] = []
    for path in output_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        stripped = text.lstrip()
        if not stripped.strip():
            empty.append(path.name)
        if stripped.startswith("```") or stripped.startswith("[") or stripped.startswith("{"):
            bad_start.append(path.name)

    missing = sorted(expected_ids - output_ids)
    extra = sorted(output_ids - expected_ids)
    ok = not missing and not empty and not bad_start
    if expected_count is not None and len(output_files) != expected_count:
        ok = False

    return {
        "ok": ok,
        "expected_articles": len(expected_ids),
        "output_files": len(output_files),
        "expected_count": expected_count,
        "missing": missing,
        "extra": extra,
        "empty": empty,
        "bad_start": bad_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rectified article outputs")
    parser.add_argument("--expected-count", type=int, default=104)
    args = parser.parse_args()

    report = validate_outputs(args.expected_count)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
