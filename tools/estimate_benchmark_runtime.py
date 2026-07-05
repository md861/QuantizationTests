#!/usr/bin/env python3
"""Estimate benchmark runtimes from existing result CSV/metadata artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize historical benchmark timing rows for planning."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark result subdirectories.",
    )
    parser.add_argument("--model", default=None, help="Optional model-name substring.")
    parser.add_argument("--method", default=None, help="Optional method substring.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(_iter_timing_rows(args.results_root))
    if args.model:
        rows = [r for r in rows if args.model in r.get("model_name", "")]
    if args.method:
        rows = [r for r in rows if args.method in r.get("method", "")]

    method_rows = [r for r in rows if r.get("method_elapsed_seconds")]
    job_rows = [r for r in rows if r.get("job_elapsed_seconds")]

    print(f"Timing rows found: {len(rows)}")
    print(f"Rows with per-method timing: {len(method_rows)}")
    if method_rows:
        by_method: dict[str, list[float]] = {}
        for row in method_rows:
            by_method.setdefault(row["method"], []).append(
                float(row["method_elapsed_seconds"])
            )
        print("\nPer-method timing medians:")
        for method, values in sorted(by_method.items()):
            print(f"  {method}: {median(values):.1f}s from {len(values)} row(s)")

    if job_rows:
        print("\nJob-level timing artifacts:")
        for row in sorted(job_rows, key=lambda r: r["path"]):
            print(
                "  {path}: {elapsed:.1f}s ({method})".format(
                    path=row["path"],
                    elapsed=float(row["job_elapsed_seconds"]),
                    method=row.get("method") or "job",
                )
            )
    if not rows:
        print("No timing artifacts found.")
    return 0


def _iter_timing_rows(results_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not results_root.exists():
        return rows

    for path in sorted(results_root.rglob("*logit_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["path"] = str(path)
                if row.get("method_elapsed_seconds"):
                    rows.append(row)

    for path in sorted(results_root.rglob("*metadata.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        elapsed = metadata.get("elapsed_seconds")
        if elapsed is None:
            continue
        rows.append(
            {
                "path": str(path),
                "model_name": str(metadata.get("model_name", "")),
                "method": str(metadata.get("method", "")),
                "job_elapsed_seconds": str(elapsed),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
