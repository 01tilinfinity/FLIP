#!/usr/bin/env python3
"""Beta-only baseline-minus-trap sweep for ExcluIR score matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


TOP_KS = (3, 5, 7, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final_score = baseline_score - beta * q_trap_score."
    )
    parser.add_argument(
        "--matrix",
        action="append",
        nargs=2,
        metavar=("MODEL", "NPZ_PATH"),
        required=True,
        help="Model label and score matrix path. Can be repeated.",
    )
    parser.add_argument(
        "--betas",
        default="0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10",
        help="Comma-separated beta values.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/excluir_beta_only_v3_oracle_style_trap",
    )
    parser.add_argument(
        "--score-source",
        choices=("baseline", "target"),
        default="baseline",
        help="Positive score source in final_score = source_score - beta * q_trap_score.",
    )
    return parser.parse_args()


def parse_betas(value: str) -> list[float]:
    betas = [float(item.strip()) for item in value.split(",") if item.strip()]
    return sorted(set(betas))


def minmax_rows(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=1, keepdims=True)
    maxs = values.max(axis=1, keepdims=True)
    denom = np.where(maxs == mins, 1.0, maxs - mins)
    return (values - mins) / denom


def rank_positions(scores: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
    doc_scores = scores[np.arange(scores.shape[0]), doc_indices]
    return 1 + (scores > doc_scores[:, None]).sum(axis=1)


def metric_row(
    *,
    model: str,
    condition: str,
    beta: float,
    answer_ranks: np.ndarray,
    trap_ranks: np.ndarray,
    baseline_recall: dict[int, float],
    baseline_violation: dict[int, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "embedding_model": model,
        "condition": condition,
        "candidate_top_n": "all",
        "beta": f"{beta:.2f}",
        "samples": int(answer_ranks.shape[0]),
        "mean_answer_rank": float(answer_ranks.mean()),
        "mean_trap_rank": float(trap_ranks.mean()),
    }
    recalls = []
    violations = []
    recall_kept_all = True
    violation_lower_all = True
    for top_k in TOP_KS:
        recall = float((answer_ranks <= top_k).mean())
        violation = float((trap_ranks <= top_k).mean())
        recalls.append(recall)
        violations.append(violation)
        row[f"recall@{top_k}"] = recall
        row[f"violation@{top_k}"] = violation
        row[f"recall_delta@{top_k}"] = recall - baseline_recall[top_k]
        row[f"violation_delta@{top_k}"] = violation - baseline_violation[top_k]
        recall_kept_all = recall_kept_all and recall >= baseline_recall[top_k]
        violation_lower_all = violation_lower_all and violation < baseline_violation[top_k]
    avg_recall = float(np.mean(recalls))
    avg_violation = float(np.mean(violations))
    baseline_avg_recall = float(np.mean([baseline_recall[k] for k in TOP_KS]))
    baseline_avg_violation = float(np.mean([baseline_violation[k] for k in TOP_KS]))
    row["avg_recall"] = avg_recall
    row["avg_violation"] = avg_violation
    row["avg_recall_delta"] = avg_recall - baseline_avg_recall
    row["avg_violation_delta"] = avg_violation - baseline_avg_violation
    row["strict_recall_kept_all_k"] = recall_kept_all
    row["avg_recall_kept"] = avg_recall >= baseline_avg_recall
    row["violation_lower_all_k"] = violation_lower_all
    row["avg_violation_lower"] = avg_violation < baseline_avg_violation
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_beta(rows: list[dict[str, object]], predicate) -> str:
    for row in rows:
        if float(row["beta"]) == 0.0:
            continue
        if predicate(row):
            return str(row["beta"])
    return ""


def best_row(rows: list[dict[str, object]], predicate) -> dict[str, object] | None:
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["avg_violation"]))


def main() -> None:
    args = parse_args()
    betas = parse_betas(args.betas)
    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, object]] = []
    tradeoff_rows: list[dict[str, object]] = []
    condition = f"{args.score_source} - beta*q_trap"

    for model, matrix_path in args.matrix:
        matrix = np.load(matrix_path, allow_pickle=True)
        baseline_scores = matrix["baseline_scores"]
        source_scores = (
            matrix["baseline_scores"]
            if args.score_source == "baseline"
            else matrix["target_scores"]
        )
        trap_scores = matrix["trap_scores"]
        answer_indices = matrix["answer_indices"]
        trap_indices = matrix["trap_indices"]

        source_norm = minmax_rows(source_scores)
        trap_norm = minmax_rows(trap_scores)
        baseline_answer_ranks = rank_positions(baseline_scores, answer_indices)
        baseline_trap_ranks = rank_positions(baseline_scores, trap_indices)
        baseline_recall = {
            top_k: float((baseline_answer_ranks <= top_k).mean()) for top_k in TOP_KS
        }
        baseline_violation = {
            top_k: float((baseline_trap_ranks <= top_k).mean()) for top_k in TOP_KS
        }

        model_rows = []
        for beta in betas:
            if beta == 0.0:
                if args.score_source == "baseline":
                    answer_ranks = baseline_answer_ranks
                    trap_ranks = baseline_trap_ranks
                else:
                    answer_ranks = rank_positions(source_scores, answer_indices)
                    trap_ranks = rank_positions(source_scores, trap_indices)
            else:
                final_scores = source_norm - beta * trap_norm
                answer_ranks = rank_positions(final_scores, answer_indices)
                trap_ranks = rank_positions(final_scores, trap_indices)
            row = metric_row(
                model=model,
                condition=condition,
                beta=beta,
                answer_ranks=answer_ranks,
                trap_ranks=trap_ranks,
                baseline_recall=baseline_recall,
                baseline_violation=baseline_violation,
            )
            model_rows.append(row)
            all_rows.append(row)

        best_strict = best_row(model_rows, lambda row: bool(row["strict_recall_kept_all_k"]))
        best_avg_kept = best_row(model_rows, lambda row: bool(row["avg_recall_kept"]))
        tradeoff_rows.append(
            {
                "embedding_model": model,
                "condition": condition,
                "candidate_top_n": "all",
                "strict_tradeoff_start_beta": first_beta(
                    model_rows, lambda row: not row["strict_recall_kept_all_k"]
                ),
                "avg_tradeoff_start_beta": first_beta(
                    model_rows, lambda row: not row["avg_recall_kept"]
                ),
                "first_avg_violation_lower_beta": first_beta(
                    model_rows, lambda row: row["avg_violation_lower"]
                ),
                "best_strict_beta": best_strict["beta"] if best_strict else "",
                "best_strict_avg_recall": best_strict["avg_recall"] if best_strict else "",
                "best_strict_avg_violation": best_strict["avg_violation"] if best_strict else "",
                "best_avg_kept_beta": best_avg_kept["beta"] if best_avg_kept else "",
                "best_avg_kept_avg_recall": best_avg_kept["avg_recall"] if best_avg_kept else "",
                "best_avg_kept_avg_violation": best_avg_kept["avg_violation"] if best_avg_kept else "",
            }
        )

    write_csv(output_dir / "beta_only_sweep_results.csv", all_rows)
    write_csv(output_dir / "beta_only_tradeoff_summary.csv", tradeoff_rows)
    print(output_dir / "beta_only_sweep_results.csv")
    print(output_dir / "beta_only_tradeoff_summary.csv")


if __name__ == "__main__":
    main()
