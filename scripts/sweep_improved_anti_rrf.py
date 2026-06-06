#!/usr/bin/env python3
"""Sweep softer score Anti-RRF variants with hinge and confidence gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_kflip_experiment import (
    BM25Retriever,
    CorpusPool,
    DenseRetriever,
    SearchResult,
    evaluate_rankings,
    load_decompositions,
    load_sample,
)
from sweep_score_anti_rrf import (
    minmax_normalize,
    parse_candidate_list,
    parse_float_list,
    precompute_searches,
    ranked_from_scores,
    score_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved score Anti-RRF sweeps.")
    parser.add_argument("--sample-csv", default="data/nevir_mini_train_q1_50_seed42.csv")
    parser.add_argument(
        "--decompositions-jsonl",
        default="outputs/openai_full_v3/query_decompositions.jsonl",
    )
    parser.add_argument("--output-dir", default="results/improved_anti_rrf")
    parser.add_argument("--retriever", choices=("bm25", "dense", "both"), default="both")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--variants", default="linear,hinge,gate,hinge_gate")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--betas", default="0.25,0.5,0.75,1,1.5,2,3")
    parser.add_argument("--margins", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--thresholds", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--candidate-top-ns", default="5,10,20,all")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Write per-sample detail rows for every configuration.",
    )
    return parser.parse_args()


def parse_variants(value: str) -> list[str]:
    allowed = {"linear", "hinge", "gate", "hinge_gate"}
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - allowed)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return variants


def penalty_value(
    *,
    variant: str,
    target_score: float,
    trap_score: float,
    beta: float,
    margin: float,
    threshold: float,
) -> float:
    delta = trap_score - target_score
    if variant == "linear":
        return beta * trap_score
    if variant == "hinge":
        return beta * max(0.0, delta - margin)
    if variant == "gate":
        return beta * trap_score if delta >= threshold else 0.0
    if variant == "hinge_gate":
        return beta * max(0.0, delta - margin) if delta >= threshold else 0.0
    raise ValueError(f"Unhandled variant: {variant}")


def run_variant(
    *,
    sample: pd.DataFrame,
    corpus: CorpusPool,
    retriever_name: str,
    searches: dict[str, dict[str, list[SearchResult]]],
    variant: str,
    alpha: float,
    beta: float,
    margin: float,
    threshold: float,
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

        final_scores = {}
        for doc_id in scoring_doc_ids:
            penalty = penalty_value(
                variant=variant,
                target_score=target_norm[doc_id],
                trap_score=trap_norm[doc_id],
                beta=beta,
                margin=margin,
                threshold=threshold,
            )
            final_scores[doc_id] = baseline_norm[doc_id] + alpha * target_norm[doc_id] - penalty

        rankings[sample_id] = ranked_from_scores(
            corpus_doc_ids=corpus.doc_ids,
            baseline_results=baseline_results,
            final_scores=final_scores,
            candidate_doc_ids=candidate_doc_ids,
        )

    candidate_label = "all" if candidate_top_n is None else str(candidate_top_n)
    method = (
        f"{variant}_score_c{candidate_label}_a{alpha:g}_b{beta:g}_"
        f"m{margin:g}_t{threshold:g}"
    )
    summary, detail_rows = evaluate_rankings(
        sample,
        rankings,
        method=method,
        retriever=retriever_name,
        top_k=top_k,
    )
    summary.update(
        {
            "variant": variant,
            "alpha": alpha,
            "beta": beta,
            "margin": margin,
            "threshold": threshold,
            "candidate_top_n": candidate_label,
        }
    )
    return summary, detail_rows


def compact_grid(
    *,
    variants: list[str],
    alphas: list[float],
    betas: list[float],
    margins: list[float],
    thresholds: list[float],
    candidate_top_ns: list[int | None],
) -> list[tuple[str, float, float, float, float, int | None]]:
    configs = []
    for variant in variants:
        variant_margins = margins if variant in {"hinge", "hinge_gate"} else [0.0]
        variant_thresholds = thresholds if variant in {"gate", "hinge_gate"} else [0.0]
        for candidate_top_n in candidate_top_ns:
            for alpha in alphas:
                for beta in betas:
                    for margin in variant_margins:
                        for threshold in variant_thresholds:
                            configs.append(
                                (variant, alpha, beta, margin, threshold, candidate_top_n)
                            )
    return configs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(Path(args.sample_csv), args.max_samples)
    corpus = CorpusPool(sample)
    decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)
    variants = parse_variants(args.variants)
    configs = compact_grid(
        variants=variants,
        alphas=parse_float_list(args.alphas),
        betas=parse_float_list(args.betas),
        margins=parse_float_list(args.margins),
        thresholds=parse_float_list(args.thresholds),
        candidate_top_ns=parse_candidate_list(args.candidate_top_ns),
    )

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
        baseline_summary.update(
            {
                "variant": "baseline",
                "alpha": "",
                "beta": "",
                "margin": "",
                "threshold": "",
                "candidate_top_n": "",
            }
        )
        summary_rows.append(baseline_summary)
        if args.write_details:
            detail_rows.extend(baseline_details)

        total = len(configs)
        for index, (variant, alpha, beta, margin, threshold, candidate_top_n) in enumerate(
            configs,
            start=1,
        ):
            if index % 250 == 0:
                print(f"{retriever.name}: {index}/{total} configs", flush=True)
            summary, details = run_variant(
                sample=sample,
                corpus=corpus,
                retriever_name=retriever.name,
                searches=searches,
                variant=variant,
                alpha=alpha,
                beta=beta,
                margin=margin,
                threshold=threshold,
                candidate_top_n=candidate_top_n,
                top_k=args.top_k,
            )
            summary_rows.append(summary)
            if args.write_details:
                detail_rows.extend(details)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "improved_anti_rrf_sweep.csv"
    detail_path = output_dir / "improved_anti_rrf_details.csv"
    summary_df.to_csv(summary_path, index=False)
    if args.write_details:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(detail_path, index=False)

    variants_df = summary_df[summary_df["variant"] != "baseline"].copy()
    variants_df["recall_minus_violation"] = (
        variants_df["recall@3"] - variants_df["violation_rate@3"]
    )
    variants_df["balanced_score"] = (
        variants_df["recall@3"] - 0.5 * variants_df["violation_rate@3"]
    )

    top_gap = variants_df.sort_values(
        ["recall_minus_violation", "recall@3", "violation_rate@3"],
        ascending=[False, False, True],
    ).head(50)
    top_balanced = variants_df.sort_values(
        ["balanced_score", "recall@3", "violation_rate@3"],
        ascending=[False, False, True],
    ).head(50)
    high_recall_low_violation = variants_df[variants_df["recall@3"] >= 0.8].sort_values(
        ["violation_rate@3", "recall@3"],
        ascending=[True, False],
    ).head(50)
    pareto_rows = []
    for _, row in variants_df.iterrows():
        dominated = variants_df[
            (variants_df["recall@3"] >= row["recall@3"])
            & (variants_df["violation_rate@3"] <= row["violation_rate@3"])
            & (
                (variants_df["recall@3"] > row["recall@3"])
                | (variants_df["violation_rate@3"] < row["violation_rate@3"])
            )
        ]
        if dominated.empty:
            pareto_rows.append(row)
    pareto_df = pd.DataFrame(pareto_rows).sort_values(
        ["recall@3", "violation_rate@3"],
        ascending=[False, True],
    )

    top_gap.to_csv(output_dir / "top_by_gap.csv", index=False)
    top_balanced.to_csv(output_dir / "top_by_balanced_score.csv", index=False)
    high_recall_low_violation.to_csv(output_dir / "top_high_recall_low_violation.csv", index=False)
    pareto_df.to_csv(output_dir / "pareto_frontier.csv", index=False)

    print(f"Summary: {summary_path}")
    if args.write_details:
        print(f"Details: {detail_path}")
    print(f"Top by gap: {output_dir / 'top_by_gap.csv'}")
    print(f"Top high-recall/low-violation: {output_dir / 'top_high_recall_low_violation.csv'}")
    print()
    print("Best by recall - violation")
    print(top_gap.head(12).to_string(index=False))
    print()
    print("Best with recall >= 0.8")
    print(high_recall_low_violation.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
