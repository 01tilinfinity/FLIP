#!/usr/bin/env python3
"""Fill BoolQuestions NaturalQuestions referenced doc cache from corpus.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


DATASET = "ustc-zhangzm/BoolQuestions"
BASE_URL = "https://datasets-server.huggingface.co"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract referenced NQ docs into cache JSON.")
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument(
        "--cache-json",
        default="data/boolquestions_not_323/referenced_corpus_doc_cache.json",
    )
    return parser.parse_args()


def fetch_eval_rows(config: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(
            f"{BASE_URL}/rows",
            params={
                "dataset": DATASET,
                "config": config,
                "split": "eval",
                "offset": offset,
                "length": 100,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        page = [item["row"] for item in data["rows"]]
        rows.extend(page)
        offset += len(page)
        if offset >= int(data["num_rows_total"]) or not page:
            break
    return rows


def first_passage_id(ctxs: list[dict[str, Any]]) -> int | None:
    for ctx in ctxs:
        passage_id = ctx.get("passage_id")
        if passage_id is not None:
            return int(passage_id)
    return None


def needed_nq_docids() -> set[int]:
    docids: set[int] = set()
    for row in fetch_eval_rows("NaturalQuestions"):
        if row.get("question_type") != "not":
            continue
        positive = first_passage_id(row.get("positive_ctxs") or [])
        negative = first_passage_id(row.get("negative_ctxs") or [])
        if positive is not None and negative is not None:
            docids.update([positive, negative])
    return docids


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_json)
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    missing = {
        docid for docid in needed_nq_docids() if f"NaturalQuestions:{docid}" not in cache
    }
    print(f"Need NaturalQuestions docs: {len(missing)}", flush=True)
    if not missing:
        return

    with Path(args.corpus_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            docid = int(row["docid"])
            if docid not in missing:
                continue
            doc = str(row.get("doc", "")).strip()
            title = str(row.get("title", "")).strip()
            text = f"{title}\n{doc}" if title and not doc.startswith(title) else doc
            cache[f"NaturalQuestions:{docid}"] = text
            missing.remove(docid)
            print(f"found NaturalQuestions:{docid}; remaining {len(missing)}", flush=True)
            if not missing:
                break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise RuntimeError(f"Missing after scan: {sorted(missing)[:10]}")
    print(f"Updated cache: {cache_path}", flush=True)


if __name__ == "__main__":
    main()
