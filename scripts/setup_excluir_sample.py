#!/usr/bin/env python3
"""Build an ExcluIR sample in the local FLIP doc1/doc2 format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ExcluIR to FLIP CSV/JSONL.")
    parser.add_argument("--raw-dir", default="data/excluir_raw")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional explicit output filename stem. Defaults to excluir_manual_<N>_seed<seed>.",
    )
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Use the first N benchmark rows instead of a deterministic shuffle.",
    )
    return parser.parse_args()


def title_of(document: str) -> str:
    return document.splitlines()[0].strip()


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (raw_dir / "corpus.json").open(encoding="utf-8") as handle:
        corpus = json.load(handle)
    with (raw_dir / "test_manual_final.json").open(encoding="utf-8") as handle:
        benchmark = json.load(handle)

    row_indices = list(range(len(benchmark)))
    if not args.preserve_order:
        random.Random(args.seed).shuffle(row_indices)
    row_indices = row_indices[: args.sample_size]

    records = []
    decomposition_rows = []
    for output_index, benchmark_index in enumerate(row_indices):
        row = benchmark[benchmark_index]
        negative_index, positive_index = row["corpus_sub_index"]
        negative_doc = corpus[negative_index]
        positive_doc = corpus[positive_index]
        sample_id = f"excluir_{benchmark_index}"
        query = row["RQ_rewrite"].strip()
        original_query = row["question0"].strip()
        negative_title = title_of(negative_doc)

        records.append(
            {
                "id": sample_id,
                "benchmark_index": benchmark_index,
                "query_column": "q1",
                "query": query,
                "q1": query,
                "q2": negative_title,
                "question0": original_query,
                "positive_corpus_index": positive_index,
                "negative_corpus_index": negative_index,
                "doc1": positive_doc,
                "doc2": negative_doc,
                "answer_doc": "doc1",
                "trap_doc": "doc2",
                "answer_text": positive_doc,
                "trap_text": negative_doc,
            }
        )
        decomposition_rows.append(
            {
                "id": sample_id,
                "query": query,
                "Q_target": original_query,
                "Q_trap": negative_title,
            }
        )

    if args.output_stem:
        stem = args.output_stem
    else:
        stem = f"excluir_manual_1000_seed{args.seed}"
        if args.sample_size != 1000:
            stem = f"excluir_manual_{args.sample_size}_seed{args.seed}"
        if args.preserve_order:
            stem += "_ordered"

    csv_path = output_dir / f"{stem}.csv"
    jsonl_path = output_dir / f"{stem}.jsonl"
    decompositions_path = output_dir / f"{stem}_decompositions.jsonl"

    sample = pd.DataFrame(records)
    sample.to_csv(csv_path, index=False)
    write_jsonl(records, jsonl_path)
    write_jsonl(decomposition_rows, decompositions_path)

    print(f"Corpus documents: {len(corpus)}")
    print(f"Benchmark rows: {len(benchmark)}")
    print(f"Rows exported: {len(sample)}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Decompositions: {decompositions_path}")
    print()
    print(sample.loc[:, ["id", "query", "q2"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
