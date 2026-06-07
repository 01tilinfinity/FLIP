#!/usr/bin/env python3
"""Fast summary-only score Anti-RRF sweep for larger samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_kflip_experiment import (
    BM25Retriever,
    CorpusPool,
    DenseRetriever,
    load_decompositions,
    load_sample,
    tokenize,
)
from sweep_score_anti_rrf import parse_candidate_list, parse_float_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast summary-only score Anti-RRF sweep.")
    parser.add_argument("--sample-csv", default="data/hotpotqa_distractor_train_1000_seed42.csv")
    parser.add_argument(
        "--decompositions-jsonl",
        default="outputs/hotpotqa_1000_bm25_heuristic/query_decompositions.jsonl",
    )
    parser.add_argument("--output-dir", default="results/hotpotqa_1000_score_anti_rrf")
    parser.add_argument("--retriever", choices=("bm25", "dense", "both"), default="both")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--betas", default="0.1,0.2,0.3,0.5,0.75,1")
    parser.add_argument("--candidate-top-ns", default="5,10,20,all")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def minmax_rows(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mins = values.min(axis=1, keepdims=True)
        maxs = values.max(axis=1, keepdims=True)
        denom = np.where(maxs == mins, 1.0, maxs - mins)
        return (values - mins) / denom

    masked = np.where(mask, values, np.nan)
    mins = np.nanmin(masked, axis=1, keepdims=True)
    maxs = np.nanmax(masked, axis=1, keepdims=True)
    denom = np.where(maxs == mins, 1.0, maxs - mins)
    return np.where(mask, (values - mins) / denom, np.nan)


def build_score_matrices(
    *,
    sample: pd.DataFrame,
    corpus: CorpusPool,
    decompositions: list[dict[str, str]],
    retriever: BM25Retriever | DenseRetriever,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = [str(row["query"]) for _, row in sample.iterrows()]
    targets = [decomposition["Q_target"] for decomposition in decompositions]
    traps = [decomposition["Q_trap"] for decomposition in decompositions]
    if isinstance(retriever, DenseRetriever):
        all_queries = queries + targets + traps
        embeddings = retriever.model.encode(
            all_queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.dot(embeddings, retriever.doc_embeddings.T)
        size = len(sample)
        return scores[:size], scores[size : 2 * size], scores[2 * size :]

    matrices = []
    for query_list in (queries, targets, traps):
        rows = [retriever.bm25.get_scores(tokenize(query)) for query in query_list]
        matrices.append(np.array(rows, dtype=float))
    return matrices[0], matrices[1], matrices[2]


def rank_positions(scores: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
    doc_scores = scores[np.arange(scores.shape[0]), doc_indices]
    return 1 + (scores > doc_scores[:, None]).sum(axis=1)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row_indices = np.arange(scores.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def candidate_ranks(
    *,
    final_scores: np.ndarray,
    baseline_ranks: np.ndarray,
    candidate_mask: np.ndarray,
    doc_indices: np.ndarray,
) -> np.ndarray:
    in_candidate = candidate_mask[np.arange(candidate_mask.shape[0]), doc_indices]
    ranks = baseline_ranks[np.arange(baseline_ranks.shape[0]), doc_indices].copy()
    candidate_scores = np.where(candidate_mask, final_scores, np.nan)
    doc_scores = final_scores[np.arange(final_scores.shape[0]), doc_indices]
    better = candidate_scores > doc_scores[:, None]
    ranks[in_candidate] = 1 + better[in_candidate].sum(axis=1)
    return ranks


def evaluate_summary(
    *,
    sample: pd.DataFrame,
    retriever_name: str,
    method: str,
    answer_ranks: np.ndarray,
    trap_ranks: np.ndarray,
    top_k: int,
    extra: dict,
) -> dict:
    row = {
        "retriever": retriever_name,
        "method": method,
        "samples": len(sample),
        f"recall@{top_k}": float((answer_ranks <= top_k).mean()),
        f"violation_rate@{top_k}": float((trap_ranks <= top_k).mean()),
        "mean_answer_rank": float(answer_ranks.mean()),
        "mean_trap_rank": float(trap_ranks.mean()),
    }
    row.update(extra)
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(Path(args.sample_csv), None)
    corpus = CorpusPool(sample)
    decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    candidate_top_ns = parse_candidate_list(args.candidate_top_ns)

    retrievers = []
    if args.retriever in ("bm25", "both"):
        retrievers.append(BM25Retriever(corpus))
    if args.retriever in ("dense", "both"):
        retrievers.append(DenseRetriever(corpus, model_name=args.dense_model))

    doc_index = {doc_id: index for index, doc_id in enumerate(corpus.doc_ids)}
    answer_indices = np.array([doc_index[f"{row['id']}::{row['answer_doc']}"] for _, row in sample.iterrows()])
    trap_indices = np.array([doc_index[f"{row['id']}::{row['trap_doc']}"] for _, row in sample.iterrows()])

    summary_rows = []
    for retriever in retrievers:
        print(f"Scoring {retriever.name}", flush=True)
        baseline_scores, target_scores, trap_scores = build_score_matrices(
            sample=sample,
            corpus=corpus,
            decompositions=decompositions,
            retriever=retriever,
        )
        baseline_ranks = ranks_from_scores(baseline_scores)
        baseline_answer_ranks = baseline_ranks[np.arange(len(sample)), answer_indices]
        baseline_trap_ranks = baseline_ranks[np.arange(len(sample)), trap_indices]
        summary_rows.append(
            evaluate_summary(
                sample=sample,
                retriever_name=retriever.name,
                method="baseline",
                answer_ranks=baseline_answer_ranks,
                trap_ranks=baseline_trap_ranks,
                top_k=args.top_k,
                extra={"alpha": "", "beta": "", "candidate_top_n": ""},
            )
        )

        baseline_norm_all = minmax_rows(baseline_scores)
        target_norm_all = minmax_rows(target_scores)
        trap_norm_all = minmax_rows(trap_scores)
        baseline_order = np.argsort(-baseline_scores, axis=1)

        for candidate_top_n in candidate_top_ns:
            candidate_label = "all" if candidate_top_n is None else str(candidate_top_n)
            if candidate_top_n is None:
                candidate_mask = None
                baseline_norm = baseline_norm_all
                target_norm = target_norm_all
                trap_norm = trap_norm_all
            else:
                candidate_mask = np.zeros_like(baseline_scores, dtype=bool)
                row_indices = np.arange(len(sample))[:, None]
                candidate_mask[row_indices, baseline_order[:, :candidate_top_n]] = True
                baseline_norm = minmax_rows(baseline_scores, candidate_mask)
                target_norm = minmax_rows(target_scores, candidate_mask)
                trap_norm = minmax_rows(trap_scores, candidate_mask)

            for alpha in alphas:
                for beta in betas:
                    final_scores = alpha * baseline_norm + target_norm - beta * trap_norm
                    if candidate_top_n is None:
                        answer_ranks = rank_positions(final_scores, answer_indices)
                        trap_ranks = rank_positions(final_scores, trap_indices)
                    else:
                        answer_ranks = candidate_ranks(
                            final_scores=final_scores,
                            baseline_ranks=baseline_ranks,
                            candidate_mask=candidate_mask,
                            doc_indices=answer_indices,
                        )
                        trap_ranks = candidate_ranks(
                            final_scores=final_scores,
                            baseline_ranks=baseline_ranks,
                            candidate_mask=candidate_mask,
                            doc_indices=trap_indices,
                        )
                    method = f"score_anti_rrf_c{candidate_label}_a{alpha:g}_b{beta:g}"
                    summary_rows.append(
                        evaluate_summary(
                            sample=sample,
                            retriever_name=retriever.name,
                            method=method,
                            answer_ranks=answer_ranks,
                            trap_ranks=trap_ranks,
                            top_k=args.top_k,
                            extra={
                                "alpha": alpha,
                                "beta": beta,
                                "candidate_top_n": candidate_label,
                            },
                        )
                    )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "score_anti_rrf_sweep.csv"
    summary.to_csv(summary_path, index=False)
    variants = summary[summary["method"] != "baseline"].copy()
    variants["recall_minus_violation"] = variants["recall@3"] - variants["violation_rate@3"]
    top = variants.sort_values(
        ["recall_minus_violation", "recall@3", "violation_rate@3"],
        ascending=[False, False, True],
    )
    top_path = output_dir / "score_anti_rrf_top_configs.csv"
    top.head(50).to_csv(top_path, index=False)

    print(f"Summary: {summary_path}")
    print(f"Top configs: {top_path}")
    print()
    print(top.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
