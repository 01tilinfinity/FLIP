#!/usr/bin/env python3
"""Build a small NevIR evaluation sample for K-FLIP experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "orionweller/NevIR"
REQUIRED_COLUMNS = ("id", "q1", "q2", "doc1", "doc2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load NevIR from Hugging Face and export a mini K-FLIP sample."
    )
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-column", default="q1", choices=("q1", "q2"))
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"NevIR schema changed; missing columns: {missing}")


def load_nevir_sample(
    *,
    split: str = "train",
    sample_size: int = 50,
    seed: int = 42,
    query_column: str = "q1",
) -> pd.DataFrame:
    """Return a deterministic mini sample with explicit target/trap fields."""
    dataset = load_dataset(DATASET_NAME, split=split)
    df = dataset.to_pandas()
    validate_columns(df)

    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if sample_size > len(df):
        raise ValueError(f"--sample-size={sample_size} exceeds split size {len(df)}")

    sample = (
        df.loc[:, list(REQUIRED_COLUMNS)]
        .sample(n=sample_size, random_state=seed)
        .reset_index(drop=True)
    )
    sample.insert(1, "query_column", query_column)
    sample.insert(2, "query", sample[query_column])
    answer_doc = "doc1" if query_column == "q1" else "doc2"
    trap_doc = "doc2" if query_column == "q1" else "doc1"
    sample["answer_doc"] = answer_doc
    sample["trap_doc"] = trap_doc
    sample["answer_text"] = sample[answer_doc]
    sample["trap_text"] = sample[trap_doc]
    return sample


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_nevir_sample(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        query_column=args.query_column,
    )

    stem = (
        f"nevir_mini_{args.split}_{args.query_column}_"
        f"{args.sample_size}_seed{args.seed}"
    )
    csv_path = output_dir / f"{stem}.csv"
    jsonl_path = output_dir / f"{stem}.jsonl"

    sample.to_csv(csv_path, index=False)
    write_jsonl(sample.to_dict(orient="records"), jsonl_path)

    print(f"Loaded dataset: {DATASET_NAME}")
    print(f"Split: {args.split}")
    print(f"Rows exported: {len(sample)}")
    print(f"Query column: {args.query_column}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print()
    print(sample.loc[:, ["id", "query", "doc1", "doc2"]].head(3).to_string())


if __name__ == "__main__":
    main()
