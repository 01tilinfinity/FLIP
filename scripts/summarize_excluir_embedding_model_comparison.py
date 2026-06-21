#!/usr/bin/env python3
"""Summarize ExcluIR rewriter experiments across embedding models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "embedding_model",
    "method",
    "gamma/alpha",
    "beta",
    "candidate_top_n",
    "Recall@3",
    "Violation@3",
    "Gap@3",
    "Recall@5",
    "Violation@5",
    "Gap@5",
    "Avg Gap",
    "Right Rank",
]


EXPERIMENTS = [
    {
        "embedding_model": "BAAI/bge-m3",
        "no_target_dir": "results/excluir_1000_bge_m3_rewriter_gpt4o_mini_no_target_top3579",
        "target_minus_trap_dir": "results/excluir_1000_bge_m3_rewriter_gpt4o_mini_target_minus_trap_top3579",
    },
    {
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "no_target_dir": "results/excluir_1000_qwen3_embedding_0_6b_rewriter_gpt4o_mini_no_target_top3579",
        "target_minus_trap_dir": "results/excluir_1000_qwen3_embedding_0_6b_rewriter_gpt4o_mini_target_minus_trap_top3579",
    },
    {
        "embedding_model": "Qwen/Qwen3-Embedding-4B",
        "no_target_dir": "results/excluir_1000_qwen3_embedding_4b_rewriter_gpt4o_mini_no_target_top3579",
        "target_minus_trap_dir": "results/excluir_1000_qwen3_embedding_4b_rewriter_gpt4o_mini_target_minus_trap_top3579",
    },
    {
        "embedding_model": "text-embedding-3-small",
        "no_target_dir": "results/excluir_1000_text_embedding_3_small_rewriter_gpt4o_mini_no_target_top3579",
        "target_minus_trap_dir": "results/excluir_1000_text_embedding_3_small_rewriter_gpt4o_mini_target_minus_trap_top3579",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/excluir_embedding_model_comparison_rewriter_gpt4o_mini",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def baseline_row(path: Path) -> dict[str, str]:
    for row in read_rows(path):
        if row.get("method") == "baseline":
            return row
    raise ValueError(f"No baseline row found in {path}")


def metric(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    if value == "":
        return ""
    return f"{float(value):.3f}"


def format_row(
    *,
    embedding_model: str,
    method: str,
    row: dict[str, str],
) -> dict[str, str]:
    gamma = row.get("gamma", "")
    alpha = row.get("alpha", "")
    weight = gamma if gamma not in {"", None} else alpha
    return {
        "embedding_model": embedding_model,
        "method": method,
        "gamma/alpha": weight,
        "beta": row.get("beta", ""),
        "candidate_top_n": row.get("candidate_top_n", ""),
        "Recall@3": metric(row, "recall@3"),
        "Violation@3": metric(row, "violation_rate@3"),
        "Gap@3": metric(row, "recall_minus_violation@3"),
        "Recall@5": metric(row, "recall@5"),
        "Violation@5": metric(row, "violation_rate@5"),
        "Gap@5": metric(row, "recall_minus_violation@5"),
        "Avg Gap": metric(row, "avg_recall_minus_violation"),
        "Right Rank": metric(row, "right_rank"),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for experiment in EXPERIMENTS:
        embedding_model = experiment["embedding_model"]
        no_target_dir = Path(experiment["no_target_dir"])
        target_minus_trap_dir = Path(experiment["target_minus_trap_dir"])

        summary_rows.append(
            format_row(
                embedding_model=embedding_model,
                method="baseline",
                row=baseline_row(no_target_dir / "score_anti_rrf_results.csv"),
            )
        )
        summary_rows.append(
            format_row(
                embedding_model=embedding_model,
                method="baseline_minus_trap",
                row=first_row(no_target_dir / "score_anti_rrf_best_configs.csv"),
            )
        )
        summary_rows.append(
            format_row(
                embedding_model=embedding_model,
                method="target_minus_trap",
                row=first_row(target_minus_trap_dir / "score_anti_rrf_best_configs.csv"),
            )
        )

    csv_path = output_dir / "embedding_model_comparison_summary.csv"
    md_path = output_dir / "embedding_model_comparison_summary.md"
    write_csv(csv_path, summary_rows)
    write_markdown(md_path, summary_rows)
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
