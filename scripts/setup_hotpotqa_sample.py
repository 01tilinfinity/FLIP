#!/usr/bin/env python3
"""Build a HotpotQA mini sample in the K-FLIP doc1/doc2 format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "hotpotqa/hotpot_qa"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HotpotQA to the local K-FLIP doc1/doc2 CSV format."
    )
    parser.add_argument("--config", default="distractor", choices=("distractor", "fullwiki"))
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def context_items(row: dict) -> list[tuple[str, list[str]]]:
    context = row["context"]
    return list(zip(context["title"], context["sentences"], strict=True))


def supporting_titles(row: dict) -> set[str]:
    return set(row["supporting_facts"]["title"])


def supporting_doc(row: dict) -> str:
    support_titles = supporting_titles(row)
    parts = []
    for title, sentences in context_items(row):
        if title in support_titles:
            text = " ".join(sentence.strip() for sentence in sentences).strip()
            if text:
                parts.append(f"{title}: {text}")
    return "\n".join(parts)


def hard_negative_doc(row: dict) -> tuple[str, str]:
    support_titles = supporting_titles(row)
    question_tokens = tokenize(row["question"])
    candidates = []
    for title, sentences in context_items(row):
        if title in support_titles:
            continue
        text = " ".join(sentence.strip() for sentence in sentences).strip()
        if not text:
            continue
        overlap = len(question_tokens & tokenize(f"{title} {text}"))
        candidates.append((overlap, title, text))
    if not candidates:
        return "", ""
    _, title, text = max(candidates, key=lambda item: (item[0], len(item[2])))
    return title, f"{title}: {text}"


def row_to_record(row: dict) -> dict[str, str] | None:
    doc1 = supporting_doc(row)
    trap_title, doc2 = hard_negative_doc(row)
    if not doc1 or not doc2:
        return None
    question = row["question"]
    return {
        "id": row["id"],
        "query_column": "q1",
        "query": question,
        "q1": question,
        "q2": trap_title,
        "answer": row.get("answer", ""),
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "doc1": doc1,
        "doc2": doc2,
        "answer_doc": "doc1",
        "trap_doc": "doc2",
        "answer_text": doc1,
        "trap_text": doc2,
    }


def build_sample(config: str, split: str, sample_size: int, seed: int) -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, config, split=split)
    shuffled = dataset.shuffle(seed=seed)
    records = []
    for row in shuffled:
        record = row_to_record(row)
        if record is None:
            continue
        records.append(record)
        if len(records) >= sample_size:
            break
    if len(records) < sample_size:
        raise ValueError(f"Only built {len(records)} usable records; requested {sample_size}")
    return pd.DataFrame(records)


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.config, args.split, args.sample_size, args.seed)

    stem = f"hotpotqa_{args.config}_{args.split}_{args.sample_size}_seed{args.seed}"
    csv_path = output_dir / f"{stem}.csv"
    jsonl_path = output_dir / f"{stem}.jsonl"
    sample.to_csv(csv_path, index=False)
    write_jsonl(sample.to_dict(orient="records"), jsonl_path)

    print(f"Loaded dataset: {DATASET_NAME}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Rows exported: {len(sample)}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print()
    print(sample.loc[:, ["id", "query", "answer", "q2"]].head(5).to_string())


if __name__ == "__main__":
    main()
