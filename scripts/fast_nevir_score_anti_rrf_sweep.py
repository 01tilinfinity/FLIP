#!/usr/bin/env python3
"""Fast score Anti-RRF comparison for NevIR doc-pair samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fast_excluir_score_anti_rrf_sweep import (
    candidate_ranks,
    evaluate_summary,
    load_dotenv_if_available,
    minmax_rows,
    parse_candidate_list,
    parse_decomposition_json,
    parse_float_list,
    parse_int_list,
    rank_positions,
    score_matrices,
    topn_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NevIR score Anti-RRF comparisons.")
    parser.add_argument("--sample-csv", default="data/nevir_mini_train_q2_1000_seed42.csv")
    parser.add_argument(
        "--decompositions-jsonl",
        default="outputs/nevir_rewriter_gpt4o_mini_v2_recall_preserving_q2/decompositions.jsonl",
    )
    parser.add_argument("--output-dir", default="results/nevir_1000_score_anti_rrf")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "sentence_transformers", "openai"),
        default="auto",
    )
    parser.add_argument("--retriever-label", default=None)
    parser.add_argument("--dense-doc-batch-size", type=int, default=64)
    parser.add_argument("--dense-query-batch-size", type=int, default=64)
    parser.add_argument("--openai-max-batch-tokens", type=int, default=250000)
    parser.add_argument("--openai-max-input-tokens", type=int, default=8191)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--query-prompt-name", default=None)
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-score-matrix-path", default=None)
    parser.add_argument("--load-score-matrix-path", default=None)
    parser.add_argument("--cache-dir", default="data/nevir_cache")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--gammas",
        default=None,
        help="Target weights for --score-mode target_minus_trap. Defaults to --alphas.",
    )
    parser.add_argument("--betas", default="0.3,0.5,0.75,1")
    parser.add_argument("--candidate-top-ns", default="5,10,20,all")
    parser.add_argument(
        "--score-mode",
        choices=("no_target", "target_minus_trap"),
        default="no_target",
    )
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def load_decompositions(path: Path, sample: pd.DataFrame) -> list[dict[str, str]]:
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_id[str(row["id"])] = parse_decomposition_json(json.dumps(row, ensure_ascii=False))

    missing = [str(row["id"]) for _, row in sample.iterrows() if str(row["id"]) not in by_id]
    if missing:
        raise ValueError(f"Cached decomposition file is missing ids: {missing[:5]}")
    return [by_id[str(row["id"])] for _, row in sample.iterrows()]


def build_nevir_corpus(sample: pd.DataFrame) -> tuple[list[str], list[str], dict[str, int]]:
    corpus: list[str] = []
    doc_ids: list[str] = []
    for _, row in sample.iterrows():
        row_id = str(row["id"])
        for column in ("doc1", "doc2"):
            doc_ids.append(f"{row_id}::{column}")
            corpus.append(str(row[column]))
    return corpus, doc_ids, {doc_id: index for index, doc_id in enumerate(doc_ids)}


def run_sweep(
    *,
    sample: pd.DataFrame,
    decompositions: list[dict[str, str]],
    corpus: list[str],
    doc_ids: list[str],
    doc_index: dict[str, int],
    args: argparse.Namespace,
) -> list[dict]:
    queries = [str(row["query"]) for _, row in sample.iterrows()]
    targets = [decomposition["Q_target"] for decomposition in decompositions]
    traps = [decomposition["Q_trap"] for decomposition in decompositions]
    answer_indices = np.array(
        [doc_index[f"{row['id']}::{row['answer_doc']}"] for _, row in sample.iterrows()],
        dtype=int,
    )
    trap_indices = np.array(
        [doc_index[f"{row['id']}::{row['trap_doc']}"] for _, row in sample.iterrows()],
        dtype=int,
    )

    load_path = Path(args.load_score_matrix_path) if args.load_score_matrix_path else None
    save_path = Path(args.save_score_matrix_path) if args.save_score_matrix_path else None
    if load_path is not None:
        print(f"Loading score matrices: {load_path}", flush=True)
        matrices = np.load(load_path, allow_pickle=True)
        baseline_scores = matrices["baseline_scores"]
        target_scores = matrices["target_scores"]
        trap_scores = matrices["trap_scores"]
    else:
        baseline_scores, target_scores, trap_scores = score_matrices(
            retriever="dense",
            corpus=corpus,
            queries=queries,
            targets=targets,
            traps=traps,
            dense_model=args.dense_model,
            cache_dir=Path(args.cache_dir),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            query_prompt_name=args.query_prompt_name,
            model_dtype=args.model_dtype,
            doc_batch_size=args.dense_doc_batch_size,
            query_batch_size=args.dense_query_batch_size,
            dense_backend=args.dense_backend,
            openai_max_batch_tokens=args.openai_max_batch_tokens,
            openai_max_input_tokens=args.openai_max_input_tokens,
        )

    if save_path is not None:
        print(f"Saving score matrices: {save_path}", flush=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            baseline_scores=baseline_scores,
            target_scores=target_scores,
            trap_scores=trap_scores,
            answer_indices=answer_indices,
            trap_indices=trap_indices,
            doc_ids=np.array(doc_ids, dtype=object),
            queries=np.array(queries, dtype=object),
            targets=np.array(targets, dtype=object),
            traps=np.array(traps, dtype=object),
            answer_docs=sample["answer_doc"].astype(str).to_numpy(dtype=object),
            trap_docs=sample["trap_doc"].astype(str).to_numpy(dtype=object),
        )

    top_ks = [args.top_k] if args.top_k is not None else parse_int_list(args.top_ks)
    alphas = parse_float_list(args.gammas if args.gammas else args.alphas)
    betas = parse_float_list(args.betas)
    candidate_top_ns = parse_candidate_list(args.candidate_top_ns)
    retriever_label = args.retriever_label or args.dense_model.replace("/", "__")

    baseline_answer_ranks = rank_positions(baseline_scores, answer_indices)
    baseline_trap_ranks = rank_positions(baseline_scores, trap_indices)
    rows = [
        evaluate_summary(
            retriever=retriever_label,
            method="baseline",
            answer_ranks=baseline_answer_ranks,
            trap_ranks=baseline_trap_ranks,
            top_ks=top_ks,
            extra={"alpha": "", "gamma": "", "beta": "", "candidate_top_n": ""},
        )
    ]

    baseline_norm_all = minmax_rows(baseline_scores)
    target_norm_all = minmax_rows(target_scores)
    trap_norm_all = minmax_rows(trap_scores)

    for candidate_top_n in candidate_top_ns:
        candidate_label = "all" if candidate_top_n is None else str(candidate_top_n)
        if candidate_top_n is None:
            candidate_mask = None
            baseline_norm = baseline_norm_all
            target_norm = target_norm_all
            trap_norm = trap_norm_all
        else:
            candidate_mask = topn_mask(baseline_scores, candidate_top_n)
            baseline_norm = minmax_rows(baseline_scores, candidate_mask)
            target_norm = minmax_rows(target_scores, candidate_mask)
            trap_norm = minmax_rows(trap_scores, candidate_mask)

        for alpha in alphas:
            for beta in betas:
                if args.score_mode == "target_minus_trap":
                    final_scores = alpha * target_norm - beta * trap_norm
                    method_prefix = "score_target_minus_trap"
                    weight_label = "g"
                    gamma = alpha
                else:
                    final_scores = alpha * baseline_norm - beta * trap_norm
                    method_prefix = "score_anti_rrf_no_target"
                    weight_label = "a"
                    gamma = ""

                if candidate_mask is None:
                    answer_ranks = rank_positions(final_scores, answer_indices)
                    trap_ranks = rank_positions(final_scores, trap_indices)
                else:
                    answer_ranks = candidate_ranks(
                        final_scores=final_scores,
                        baseline_scores=baseline_scores,
                        candidate_mask=candidate_mask,
                        doc_indices=answer_indices,
                    )
                    trap_ranks = candidate_ranks(
                        final_scores=final_scores,
                        baseline_scores=baseline_scores,
                        candidate_mask=candidate_mask,
                        doc_indices=trap_indices,
                    )

                rows.append(
                    evaluate_summary(
                        retriever=retriever_label,
                        method=(
                            f"{method_prefix}_c{candidate_label}_"
                            f"{weight_label}{alpha:g}_b{beta:g}"
                        ),
                        answer_ranks=answer_ranks,
                        trap_ranks=trap_ranks,
                        top_ks=top_ks,
                        extra={
                            "alpha": alpha,
                            "gamma": gamma,
                            "beta": beta,
                            "candidate_top_n": candidate_label,
                        },
                    )
                )
    return rows


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(args.sample_csv)
    corpus, doc_ids, doc_index = build_nevir_corpus(sample)
    decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)

    summary = pd.DataFrame(
        run_sweep(
            sample=sample,
            decompositions=decompositions,
            corpus=corpus,
            doc_ids=doc_ids,
            doc_index=doc_index,
            args=args,
        )
    )
    summary_path = output_dir / "score_anti_rrf_results.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(output_dir / "score_anti_rrf_sweep.csv", index=False)

    variants = summary[summary["method"] != "baseline"].copy()
    primary_k = min([args.top_k] if args.top_k is not None else parse_int_list(args.top_ks))
    primary_gap = f"recall_minus_violation@{primary_k}"
    primary_recall = f"recall@{primary_k}"
    primary_violation = f"violation_rate@{primary_k}"
    top = variants.sort_values(
        ["avg_recall_minus_violation", primary_gap, "right_rank", primary_recall, primary_violation],
        ascending=[False, False, False, False, True],
    )
    top_path = output_dir / "score_anti_rrf_best_configs.csv"
    top.head(50).to_csv(top_path, index=False)
    top.head(50).to_csv(output_dir / "score_anti_rrf_top_configs.csv", index=False)

    print(f"Summary: {summary_path}")
    print(f"Best configs: {top_path}")
    print()
    print(top.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
