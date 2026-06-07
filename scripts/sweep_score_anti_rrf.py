#!/usr/bin/env python3
"""Sweep score-based Anti-RRF variants over cached query decompositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_kflip_experiment import (
    BM25Retriever,
    CorpusPool,
    DenseRetriever,
    SearchResult,
    evaluate_rankings,
    load_decompositions,
    load_sample,
    ranked_results,
    to_rank_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run score-level Anti-RRF sweeps.")
    parser.add_argument("--sample-csv", default="data/nevir_mini_train_q1_50_seed42.csv")
    parser.add_argument(
        "--decompositions-jsonl",
        default="outputs/openai_full_v3/query_decompositions.jsonl",
    )
    parser.add_argument("--output-dir", default="results/score_anti_rrf")
    parser.add_argument("--retriever", choices=("bm25", "dense", "both"), default="both")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--betas", default="0.1,0.2,0.3,0.5,0.75,1")
    parser.add_argument(
        "--candidate-top-ns",
        default="5,10,20,all",
        help="Comma-separated candidate guards. Use 'all' for no guard.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Write per-sample detail rows for every configuration.",
    )
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_candidate_list(value: str) -> list[int | None]:
    candidates: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        candidates.append(None if item in {"all", "none"} else int(item))
    return candidates


def score_dict(results: list[SearchResult]) -> dict[str, float]:
    return {result.doc_id: result.score for result in results}


def minmax_normalize(scores: dict[str, float], doc_ids: list[str]) -> dict[str, float]:
    values = np.array([scores[doc_id] for doc_id in doc_ids], dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return {doc_id: 0.0 for doc_id in doc_ids}
    return {doc_id: (scores[doc_id] - min_value) / (max_value - min_value) for doc_id in doc_ids}


def ranked_from_scores(
    *,
    corpus_doc_ids: list[str],
    baseline_results: list[SearchResult],
    final_scores: dict[str, float],
    candidate_doc_ids: list[str] | None,
) -> list[SearchResult]:
    if candidate_doc_ids is None:
        ordered_doc_ids = sorted(corpus_doc_ids, key=lambda doc_id: (-final_scores[doc_id], doc_id))
    else:
        baseline_ranks = to_rank_dict(baseline_results)
        candidate_set = set(candidate_doc_ids)
        ordered_candidates = sorted(
            candidate_doc_ids,
            key=lambda doc_id: (-final_scores[doc_id], baseline_ranks[doc_id]),
        )
        remainder = [result.doc_id for result in baseline_results if result.doc_id not in candidate_set]
        ordered_doc_ids = ordered_candidates + remainder

    return [
        SearchResult(
            doc_id=doc_id,
            score=final_scores.get(doc_id, float("-inf")),
            rank=rank,
        )
        for rank, doc_id in enumerate(ordered_doc_ids, start=1)
    ]


def precompute_searches(
    sample: pd.DataFrame,
    retriever: BM25Retriever | DenseRetriever,
    decompositions: list[dict[str, str]],
) -> dict[str, dict[str, list[SearchResult]]]:
    if isinstance(retriever, DenseRetriever):
        sample_ids = [str(row["id"]) for _, row in sample.iterrows()]
        baseline_queries = [str(row["query"]) for _, row in sample.iterrows()]
        target_queries = [decomposition["Q_target"] for decomposition in decompositions]
        trap_queries = [decomposition["Q_trap"] for decomposition in decompositions]
        all_queries = baseline_queries + target_queries + trap_queries
        query_embeddings = retriever.model.encode(
            all_queries,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.dot(query_embeddings, retriever.doc_embeddings.T)
        searches: dict[str, dict[str, list[SearchResult]]] = {}
        size = len(sample_ids)
        for index, sample_id in enumerate(sample_ids):
            searches[sample_id] = {
                "baseline": ranked_results(retriever.corpus.doc_ids, scores[index]),
                "target": ranked_results(retriever.corpus.doc_ids, scores[size + index]),
                "trap": ranked_results(retriever.corpus.doc_ids, scores[2 * size + index]),
            }
        return searches

    searches: dict[str, dict[str, list[SearchResult]]] = {}
    for decomposition, (_, row) in zip(decompositions, sample.iterrows(), strict=True):
        sample_id = str(row["id"])
        searches[sample_id] = {
            "baseline": retriever.search(str(row["query"])),
            "target": retriever.search(decomposition["Q_target"]),
            "trap": retriever.search(decomposition["Q_trap"]),
        }
    return searches


def run_score_anti_rrf(
    *,
    sample: pd.DataFrame,
    corpus: CorpusPool,
    retriever_name: str,
    searches: dict[str, dict[str, list[SearchResult]]],
    alpha: float,
    beta: float,
    candidate_top_n: int | None,
    top_k: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rankings: dict[str, list[SearchResult]] = {}
    for _, row in sample.iterrows():
        sample_id = str(row["id"])
        baseline_results = searches[sample_id]["baseline"]
        target_results = searches[sample_id]["target"]
        trap_results = searches[sample_id]["trap"]
        candidate_doc_ids = (
            None
            if candidate_top_n is None
            else [result.doc_id for result in baseline_results[:candidate_top_n]]
        )
        scoring_doc_ids = candidate_doc_ids if candidate_doc_ids is not None else corpus.doc_ids

        baseline_norm = minmax_normalize(score_dict(baseline_results), scoring_doc_ids)
        target_norm = minmax_normalize(score_dict(target_results), scoring_doc_ids)
        trap_norm = minmax_normalize(score_dict(trap_results), scoring_doc_ids)

        final_scores = {
            doc_id: alpha * baseline_norm[doc_id] + target_norm[doc_id] - beta * trap_norm[doc_id]
            for doc_id in scoring_doc_ids
        }
        rankings[sample_id] = ranked_from_scores(
            corpus_doc_ids=corpus.doc_ids,
            baseline_results=baseline_results,
            final_scores=final_scores,
            candidate_doc_ids=candidate_doc_ids,
        )

    candidate_label = "all" if candidate_top_n is None else str(candidate_top_n)
    method = f"score_anti_rrf_c{candidate_label}_a{alpha:g}_b{beta:g}"
    summary, detail_rows = evaluate_rankings(
        sample,
        rankings,
        method=method,
        retriever=retriever_name,
        top_k=top_k,
    )
    summary["alpha"] = alpha
    summary["beta"] = beta
    summary["candidate_top_n"] = candidate_label
    return summary, detail_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(Path(args.sample_csv), args.max_samples)
    corpus = CorpusPool(sample)
    decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    candidate_top_ns = parse_candidate_list(args.candidate_top_ns)

    retrievers: list[BM25Retriever | DenseRetriever] = []
    if args.retriever in ("bm25", "both"):
        retrievers.append(BM25Retriever(corpus))
    if args.retriever in ("dense", "both"):
        retrievers.append(DenseRetriever(corpus, model_name=args.dense_model))

    summary_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[dict[str, float | int | str]] = []
    for retriever in retrievers:
        print(f"Precomputing searches for {retriever.name}", flush=True)
        searches = precompute_searches(sample, retriever, decompositions)
        baseline_rankings = {
            str(row["id"]): searches[str(row["id"])]["baseline"]
            for _, row in sample.iterrows()
        }
        baseline_summary, baseline_details = evaluate_rankings(
            sample,
            baseline_rankings,
            method="baseline",
            retriever=retriever.name,
            top_k=args.top_k,
        )
        baseline_summary["alpha"] = ""
        baseline_summary["beta"] = ""
        baseline_summary["candidate_top_n"] = ""
        summary_rows.append(baseline_summary)
        if args.write_details:
            detail_rows.extend(baseline_details)

        for candidate_top_n in candidate_top_ns:
            for alpha in alphas:
                for beta in betas:
                    summary, details = run_score_anti_rrf(
                        sample=sample,
                        corpus=corpus,
                        retriever_name=retriever.name,
                        searches=searches,
                        alpha=alpha,
                        beta=beta,
                        candidate_top_n=candidate_top_n,
                        top_k=args.top_k,
                    )
                    summary_rows.append(summary)
                    if args.write_details:
                        detail_rows.extend(details)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "score_anti_rrf_sweep.csv"
    detail_path = output_dir / "score_anti_rrf_details.csv"
    summary_df.to_csv(summary_path, index=False)
    if args.write_details:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(detail_path, index=False)

    top_df = summary_df[summary_df["method"] != "baseline"].copy()
    top_df["recall_minus_violation"] = top_df["recall@3"] - top_df["violation_rate@3"]
    top_df = top_df.sort_values(
        by=["recall_minus_violation", "recall@3", "violation_rate@3"],
        ascending=[False, False, True],
    )
    top_path = output_dir / "score_anti_rrf_top_configs.csv"
    top_df.head(25).to_csv(top_path, index=False)

    print(f"Summary: {summary_path}")
    if args.write_details:
        print(f"Details: {detail_path}")
    print(f"Top configs: {top_path}")
    print()
    print(top_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
