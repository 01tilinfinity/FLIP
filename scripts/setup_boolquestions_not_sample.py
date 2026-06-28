#!/usr/bin/env python3
"""Build the BoolQuestions explicit-not sample with resolvable negative documents.

The output matches the ExcluIR CSV/corpus format used by
``fast_excluir_score_anti_rrf_sweep.py``:

- ``query`` is the original BoolQuestions query.
- ``positive_corpus_index`` points to the first positive passage.
- ``negative_corpus_index`` points to the first negative passage with a passage id.
- ``corpus.json`` contains only the referenced positive/negative documents.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


DATASET = "ustc-zhangzm/BoolQuestions"
EVAL_CONFIGS = ("MSMARCO", "NaturalQuestions")
BASE_URL = "https://datasets-server.huggingface.co"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BoolQuestions not-only rows.")
    parser.add_argument("--output-dir", default="data/boolquestions_not_323")
    parser.add_argument("--output-stem", default="boolquestions_not_323")
    parser.add_argument(
        "--doc-source",
        choices=("parquet", "api"),
        default="parquet",
        help="How to resolve corpus documents. Parquet is slower once but avoids API rate limits.",
    )
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="Directory for cached BoolQuestions corpus parquet shards.",
    )
    parser.add_argument(
        "--allow-original-jsonl-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the original corpus.jsonl from the HF repo if Dataset Viewer parquet is partial.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def hf_get(path: str, params: dict[str, Any], *, max_retries: int) -> dict[str, Any]:
    url = f"{BASE_URL}{path}?{urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                wait_seconds = float(retry_after) if retry_after else min(10 * 2**attempt, 120)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            return data
        except Exception as error:
            last_error = error
            if attempt == max_retries - 1:
                break
            time.sleep(min(5 * 2**attempt, 120))
    raise RuntimeError(f"HF request failed: {url}") from last_error


def fetch_all_eval_rows(config: str, *, max_retries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = hf_get(
            "/rows",
            {
                "dataset": DATASET,
                "config": config,
                "split": "eval",
                "offset": offset,
                "length": 100,
            },
            max_retries=max_retries,
        )
        page = [item["row"] for item in data["rows"]]
        rows.extend(page)
        offset += len(page)
        total = int(data["num_rows_total"])
        if offset >= total or not page:
            break
    return rows


def fetch_corpus_doc(
    *,
    config: str,
    docid: int,
    max_retries: int,
    cache: dict[str, str],
    sleep_seconds: float,
) -> str:
    key = f"{config}:{docid}"
    if key in cache:
        return cache[key]
    data = hf_get(
        "/rows",
        {
            "dataset": DATASET,
            "config": f"{config}-corpus",
            "split": "corpus",
            "offset": docid,
            "length": 1,
        },
        max_retries=max_retries,
    )
    if not data["rows"]:
        raise ValueError(f"Missing corpus row for {key}")
    row = data["rows"][0]["row"]
    if int(row["docid"]) != docid:
        raise ValueError(f"Expected docid={docid} for {key}, got {row['docid']}")
    doc = str(row.get("doc", "")).strip()
    title = str(row.get("title", "")).strip()
    text = f"{title}\n{doc}" if title and not doc.startswith(title) else doc
    cache[key] = text
    time.sleep(sleep_seconds)
    return text


def fetch_parquet_files(config: str, *, max_retries: int) -> list[dict[str, Any]]:
    data = hf_get(
        "/parquet",
        {"dataset": DATASET},
        max_retries=max_retries,
    )
    return [
        item
        for item in data["parquet_files"]
        if item["config"] == f"{config}-corpus" and item["split"] == "corpus"
    ]


def download_file(url: str, path: Path, *, max_retries: int) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    wait_seconds = float(retry_after) if retry_after else min(10 * 2**attempt, 120)
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(path)
            return
        except Exception as error:
            last_error = error
            if attempt == max_retries - 1:
                break
            time.sleep(min(5 * 2**attempt, 120))
    raise RuntimeError(f"Failed to download {url}") from last_error


def load_corpus_docs_from_parquet(
    *,
    config: str,
    docids: set[int],
    parquet_dir: Path,
    cache: dict[str, str],
    max_retries: int,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for --doc-source parquet") from error

    missing = {docid for docid in docids if f"{config}:{docid}" not in cache}
    if not missing:
        return
    parquet_files = fetch_parquet_files(config, max_retries=max_retries)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for {config}-corpus")

    config_dir = parquet_dir / f"{config}-corpus"
    for item in parquet_files:
        local_path = config_dir / item["filename"]
        print(f"parquet cache {config}: {item['filename']}", flush=True)
        download_file(item["url"], local_path, max_retries=max_retries)
        columns = ["docid", "doc", "title"] if config == "NaturalQuestions" else ["docid", "doc"]
        table = pq.read_table(
            local_path,
            columns=columns,
            filters=[("docid", "in", sorted(missing))],
        )
        for row in table.to_pylist():
            docid = int(row["docid"])
            doc = str(row.get("doc", "")).strip()
            title = str(row.get("title", "")).strip()
            text = f"{title}\n{doc}" if title and not doc.startswith(title) else doc
            cache[f"{config}:{docid}"] = text
        missing = {docid for docid in missing if f"{config}:{docid}" not in cache}
        if not missing:
            break
    if missing:
        raise ValueError(f"Missing {len(missing)} {config} docs after parquet scan: {sorted(missing)[:10]}")


def load_corpus_docs_from_original_jsonl(
    *,
    config: str,
    docids: set[int],
    cache: dict[str, str],
    max_retries: int,
) -> None:
    missing = {docid for docid in docids if f"{config}:{docid}" not in cache}
    if not missing:
        return
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{config}/corpus.jsonl"
    max_needed = max(missing)
    print(
        f"stream original corpus {config}: need {len(missing)} docs, max docid {max_needed}",
        flush=True,
    )
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    wait_seconds = float(retry_after) if retry_after else min(10 * 2**attempt, 120)
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    row = json.loads(raw_line)
                    docid = int(row["docid"])
                    if docid in missing:
                        doc = str(row.get("doc", "")).strip()
                        title = str(row.get("title", "")).strip()
                        text = f"{title}\n{doc}" if title and not doc.startswith(title) else doc
                        cache[f"{config}:{docid}"] = text
                        missing.remove(docid)
                        print(f"found {config}:{docid}; remaining {len(missing)}", flush=True)
                        if not missing:
                            return
                    if docid > max_needed and config != "NaturalQuestions":
                        break
            if not missing:
                return
        except Exception as error:
            last_error = error
            if attempt == max_retries - 1:
                break
            time.sleep(min(5 * 2**attempt, 120))
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} {config} docs after original JSONL scan: {sorted(missing)[:10]}"
        ) from last_error


def first_passage_id(ctxs: list[dict[str, Any]]) -> int | None:
    for ctx in ctxs:
        passage_id = ctx.get("passage_id")
        if passage_id is not None:
            return int(passage_id)
    return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else output_dir / "parquet_cache"

    doc_cache_path = output_dir / "referenced_corpus_doc_cache.json"
    doc_cache: dict[str, str] = {}
    if doc_cache_path.exists():
        doc_cache = json.loads(doc_cache_path.read_text(encoding="utf-8"))

    candidates: list[dict[str, Any]] = []
    needed_docids: dict[str, set[int]] = {config: set() for config in EVAL_CONFIGS}
    skipped: list[dict[str, Any]] = []
    for config in EVAL_CONFIGS:
        eval_rows = fetch_all_eval_rows(config, max_retries=args.max_retries)
        for row in eval_rows:
            if row.get("question_type") != "not":
                continue
            positive_ctxs = row.get("positive_ctxs") or []
            negative_ctxs = row.get("negative_ctxs") or []
            positive_docid = first_passage_id(positive_ctxs)
            negative_docid = first_passage_id(negative_ctxs)
            if positive_docid is None or negative_docid is None:
                skipped.append(
                    {
                        "config": config,
                        "qid": row.get("qid"),
                        "reason": "missing_positive_or_negative_passage_id",
                    }
                )
                continue
            needed_docids[config].update([positive_docid, negative_docid])
            candidates.append(
                {
                    "config": config,
                    "row": row,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": negative_ctxs,
                    "positive_docid": positive_docid,
                    "negative_docid": negative_docid,
                }
            )

    for config, docids in needed_docids.items():
        missing = {docid for docid in docids if f"{config}:{docid}" not in doc_cache}
        if not missing:
            continue
        if args.doc_source == "parquet":
            try:
                load_corpus_docs_from_parquet(
                    config=config,
                    docids=docids,
                    parquet_dir=parquet_dir,
                    cache=doc_cache,
                    max_retries=args.max_retries,
                )
            except ValueError:
                if not args.allow_original_jsonl_fallback:
                    raise
                load_corpus_docs_from_original_jsonl(
                    config=config,
                    docids=docids,
                    cache=doc_cache,
                    max_retries=args.max_retries,
                )
            write_json(doc_cache_path, doc_cache)
        else:
            for docid in sorted(missing):
                fetch_corpus_doc(
                    config=config,
                    docid=docid,
                    max_retries=args.max_retries,
                    cache=doc_cache,
                    sleep_seconds=args.sleep_seconds,
                )
                write_json(doc_cache_path, doc_cache)

    corpus: list[str] = []
    corpus_index_by_key: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    for candidate in candidates:
        config = candidate["config"]
        row = candidate["row"]
        positive_ctxs = candidate["positive_ctxs"]
        negative_ctxs = candidate["negative_ctxs"]
        positive_docid = candidate["positive_docid"]
        negative_docid = candidate["negative_docid"]
        indices: dict[str, int] = {}
        docs: dict[str, str] = {}
        for label, docid in (("positive", positive_docid), ("negative", negative_docid)):
            key = f"{config}:{docid}"
            if key not in doc_cache:
                raise ValueError(f"Missing resolved document text for {key}")
            if key not in corpus_index_by_key:
                corpus_index_by_key[key] = len(corpus)
                corpus.append(doc_cache[key])
            indices[label] = corpus_index_by_key[key]
            docs[label] = corpus[indices[label]]

        sample_id = f"boolq_{config}_{row['qid']}"
        positive_answer = positive_ctxs[0].get("answer", "") if positive_ctxs else ""
        negative_answer = negative_ctxs[0].get("answer", "") if negative_ctxs else ""
        records.append(
            {
                "id": sample_id,
                "dataset_config": config,
                "qid": row["qid"],
                "query_column": "q1",
                "query": str(row["question"]).strip(),
                "q1": str(row["question"]).strip(),
                "q2": "",
                "question_type": row["question_type"],
                "positive_passage_id": positive_docid,
                "negative_passage_id": negative_docid,
                "positive_answer": positive_answer,
                "negative_answer": negative_answer,
                "positive_corpus_index": indices["positive"],
                "negative_corpus_index": indices["negative"],
                "doc1": docs["positive"],
                "doc2": docs["negative"],
                "answer_doc": "doc1",
                "trap_doc": "doc2",
                "answer_text": docs["positive"],
                "trap_text": docs["negative"],
            }
        )

    if not records:
        raise RuntimeError("No BoolQuestions not rows with positive and negative passage ids found.")

    stem = args.output_stem
    write_csv(output_dir / f"{stem}.csv", records)
    write_jsonl(output_dir / f"{stem}.jsonl", records)
    write_json(output_dir / "corpus.json", corpus)
    write_json(output_dir / "corpus_index_by_source_docid.json", corpus_index_by_key)
    write_json(output_dir / "skipped_rows.json", skipped)

    print(f"Rows exported: {len(records)}")
    print(f"Referenced corpus docs: {len(corpus)}")
    print(f"Skipped not rows: {len(skipped)}")
    print(f"CSV: {output_dir / f'{stem}.csv'}")
    print(f"JSONL: {output_dir / f'{stem}.jsonl'}")
    print(f"Corpus: {output_dir / 'corpus.json'}")


if __name__ == "__main__":
    main()
