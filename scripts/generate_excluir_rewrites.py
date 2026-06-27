#!/usr/bin/env python3
"""Generate non-oracle ExcluIR query decompositions with an OpenAI model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts/excluir_rewriter_gpt4o_mini/v1_base_system.txt")


USER_TEMPLATE = """Now decompose the following query.

Original query:
{original_query}

Return only valid JSON. Do not include markdown, comments, explanations, or extra fields.

{{
"q_target": "...",
"q_trap": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate inferred ExcluIR decompositions.")
    parser.add_argument("--sample-csv", default="data/excluir_manual_1000_seed42.csv")
    parser.add_argument(
        "--output-jsonl",
        default="outputs/excluir_rewriter_gpt4o_mini_v1_base/decompositions.jsonl",
    )
    parser.add_argument("--system-prompt-path", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--allow-mixed-prompt-output",
        action="store_true",
        help="Allow appending rows generated with a different recorded prompt hash.",
    )
    return parser.parse_args()


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_rows(path: Path, query_column: str, max_samples: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if query_column not in row:
                raise KeyError(f"Missing query column: {query_column}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "query": str(row[query_column]).strip(),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except Exception:
                continue
    return done


def load_recorded_prompt_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    hashes = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                prompt_hash = json.loads(line).get("system_prompt_sha256")
            except Exception:
                continue
            if isinstance(prompt_hash, str) and prompt_hash:
                hashes.add(prompt_hash)
    return hashes


def parse_json_response(content: str) -> dict[str, str]:
    data = json.loads(content)
    q_target = str(data.get("q_target", "")).strip()
    q_trap = str(data.get("q_trap", "")).strip()
    return {"q_target": q_target, "q_trap": q_trap}


def decompose_query(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    query: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(original_query=query)},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = parse_json_response(content)
            return {
                **parsed,
                "raw_response": content,
                "usage": response.usage.model_dump() if response.usage else {},
            }
        except Exception as error:
            last_error = error
            wait_seconds = min(2**attempt, 30)
            print(f"retry {attempt + 1}/{max_retries}: {type(error).__name__}: {error}", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to decompose query after {max_retries} attempts") from last_error


def main() -> None:
    args = parse_args()
    load_dotenv()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.sample_csv), args.query_column, args.max_samples)
    system_prompt_path = Path(args.system_prompt_path)
    system_prompt = load_system_prompt(system_prompt_path)
    system_prompt_sha256 = file_sha256(system_prompt_path)
    recorded_prompt_hashes = load_recorded_prompt_hashes(output_path)
    if (
        recorded_prompt_hashes
        and system_prompt_sha256 not in recorded_prompt_hashes
        and not args.allow_mixed_prompt_output
    ):
        raise ValueError(
            f"{output_path} already contains rows from another prompt hash. "
            "Use a version-specific --output-jsonl, or pass --allow-mixed-prompt-output."
        )
    done_ids = load_done_ids(output_path)
    client = OpenAI()

    pending_rows = [row for row in rows if row["id"] not in done_ids]
    if not pending_rows:
        print(f"All rows already complete: {output_path}")
        return

    def run_one(row: dict[str, str]) -> dict[str, Any]:
        result = decompose_query(
            client,
            model=args.model,
            system_prompt=system_prompt,
            query=row["query"],
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
        return {
            "id": row["id"],
            "query": row["query"],
            "Q_target": result["q_target"],
            "Q_trap": result["q_trap"],
            "rewriter_model": args.model,
            "system_prompt_path": str(system_prompt_path),
            "system_prompt_sha256": system_prompt_sha256,
            "usage": result["usage"],
        }

    completed = len(done_ids)
    total = len(rows)
    with output_path.open("a", encoding="utf-8") as handle:
        if args.workers <= 1:
            for row in pending_rows:
                record = run_one(row)
                completed += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"{completed}/{total} {record['id']} | target={record['Q_target']!r} | trap={record['Q_trap']!r}",
                    flush=True,
                )
                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)
            print(f"Output: {output_path}")
            return

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, row): row for row in pending_rows}
            for future in as_completed(futures):
                record = future.result()
                completed += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"{completed}/{total} {record['id']} | target={record['Q_target']!r} | trap={record['Q_trap']!r}",
                    flush=True,
                )

    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
