#!/usr/bin/env python3
"""Combine beta-only sweep outputs into a paper-friendly CSV/TSV table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine beta-only sweep CSV files.")
    parser.add_argument(
        "--input",
        action="append",
        nargs=4,
        metavar=("DATASET", "SPLIT", "CONDITION", "CSV_PATH"),
        required=True,
        help="Dataset label, split/sample label, condition label, and beta_only_sweep_results.csv.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-tsv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for dataset, split, condition_label, csv_path in args.input:
        frame = pd.read_csv(csv_path)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "split", split)
        frame["condition_label"] = condition_label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    ordered = [
        "dataset",
        "split",
        "embedding_model",
        "condition_label",
        "condition",
        "candidate_top_n",
        "beta",
        "samples",
        "recall@3",
        "recall@5",
        "recall@7",
        "recall@9",
        "violation@3",
        "violation@5",
        "violation@7",
        "violation@9",
        "avg_recall",
        "avg_violation",
        "avg_recall_delta",
        "avg_violation_delta",
        "strict_recall_kept_all_k",
        "avg_recall_kept",
        "violation_lower_all_k",
        "avg_violation_lower",
        "mean_answer_rank",
        "mean_trap_rank",
    ]
    existing = [column for column in ordered if column in combined.columns]
    remaining = [column for column in combined.columns if column not in existing]
    combined = combined[existing + remaining]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    if args.output_tsv:
        output_tsv = Path(args.output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_tsv, index=False, sep="\t")
    print(f"Rows: {len(combined)}")
    print(f"CSV: {output_csv}")
    if args.output_tsv:
        print(f"TSV: {args.output_tsv}")


if __name__ == "__main__":
    main()
