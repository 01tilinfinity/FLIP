#!/usr/bin/env python3
"""Fast score Anti-RRF comparison for ExcluIR against the full corpus."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=Path(".env"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ExcluIR score Anti-RRF comparisons.")
    parser.add_argument("--sample-csv", default="data/excluir_manual_1000_seed42.csv")
    parser.add_argument(
        "--decompositions-jsonl",
        default="data/excluir_manual_1000_seed42_decompositions.jsonl",
    )
    parser.add_argument("--corpus-json", default="data/excluir_raw/corpus.json")
    parser.add_argument("--output-dir", default="results/excluir_1000_score_anti_rrf")
    parser.add_argument("--retriever", choices=("bm25", "dense", "both"), default="both")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "sentence_transformers", "openai"),
        default="auto",
        help="Embedding backend for dense retrieval. Auto uses OpenAI for text-embedding-* models.",
    )
    parser.add_argument(
        "--retriever-label",
        default=None,
        help="Label to write in result CSVs for a dense model.",
    )
    parser.add_argument("--dense-doc-batch-size", type=int, default=64)
    parser.add_argument("--dense-query-batch-size", type=int, default=64)
    parser.add_argument(
        "--openai-max-batch-tokens",
        type=int,
        default=250000,
        help="Approximate max tokens per OpenAI embeddings request batch.",
    )
    parser.add_argument(
        "--openai-max-input-tokens",
        type=int,
        default=8191,
        help="Truncate individual OpenAI embedding inputs to this many tokens.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face custom model code for dense models that require it.",
    )
    parser.add_argument(
        "--query-prompt-name",
        default=None,
        help="SentenceTransformer prompt_name for query-like inputs.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype for dense model loading when supported.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the dense model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--save-score-matrix-path",
        default=None,
        help="Optional .npz path to save baseline, target, and trap score matrices.",
    )
    parser.add_argument(
        "--load-score-matrix-path",
        default=None,
        help="Optional .npz path to reuse previously saved score matrices.",
    )
    parser.add_argument("--cache-dir", default="data/excluir_cache")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--gammas",
        default=None,
        help=(
            "Comma-separated target weights for --score-mode target_minus_trap. "
            "Defaults to --alphas for backward compatibility."
        ),
    )
    parser.add_argument("--betas", default="0.3,0.5,0.75,1")
    parser.add_argument("--candidate-top-ns", default="5,10,20,all")
    parser.add_argument(
        "--score-mode",
        choices=("full", "no_target", "target_minus_trap"),
        default="full",
        help=(
            "full: alpha * baseline_score + target_score - beta * trap_score; "
            "no_target: alpha * baseline_score - beta * trap_score; "
            "target_minus_trap: gamma * target_score - beta * trap_score"
        ),
    )
    parser.add_argument(
        "--top-ks",
        default="3,5,7,9",
        help="Comma-separated cutoffs to evaluate, e.g. 3,5,7,9.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Deprecated single cutoff. Use --top-ks instead.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one top-k cutoff is required.")
    return sorted(set(items))


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


def parse_decomposition_json(content: str) -> dict[str, str]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Decomposition response must be a JSON object")
    target = parsed.get("Q_target", parsed.get("q_target"))
    trap = parsed.get("Q_trap", parsed.get("q_trap"))
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Decomposition response is missing Q_target")
    if not isinstance(trap, str):
        raise ValueError("Decomposition response is missing Q_trap")
    return {"Q_target": target.strip(), "Q_trap": trap.strip()}


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


def rank_positions(scores: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
    doc_scores = scores[np.arange(scores.shape[0]), doc_indices]
    return 1 + (scores > doc_scores[:, None]).sum(axis=1)


def candidate_ranks(
    *,
    final_scores: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_mask: np.ndarray,
    doc_indices: np.ndarray,
) -> np.ndarray:
    baseline_ranks = rank_positions(baseline_scores, doc_indices)
    in_candidate = candidate_mask[np.arange(candidate_mask.shape[0]), doc_indices]
    ranks = baseline_ranks.copy()
    candidate_scores = np.where(candidate_mask, final_scores, np.nan)
    doc_scores = final_scores[np.arange(final_scores.shape[0]), doc_indices]
    better = candidate_scores > doc_scores[:, None]
    ranks[in_candidate] = 1 + better[in_candidate].sum(axis=1)
    return ranks


def topn_mask(scores: np.ndarray, top_n: int) -> np.ndarray:
    partition = np.argpartition(-scores, kth=top_n - 1, axis=1)[:, :top_n]
    mask = np.zeros(scores.shape, dtype=bool)
    mask[np.arange(scores.shape[0])[:, None], partition] = True
    return mask


def evaluate_summary(
    *,
    retriever: str,
    method: str,
    answer_ranks: np.ndarray,
    trap_ranks: np.ndarray,
    top_ks: list[int],
    extra: dict,
) -> dict:
    row = {
        "retriever": retriever,
        "method": method,
        "samples": int(answer_ranks.shape[0]),
        "right_rank": float((answer_ranks < trap_ranks).mean()),
        "mean_answer_rank": float(answer_ranks.mean()),
        "mean_trap_rank": float(trap_ranks.mean()),
    }
    gaps = []
    for top_k in top_ks:
        recall = float((answer_ranks <= top_k).mean())
        violation = float((trap_ranks <= top_k).mean())
        gap = recall - violation
        row[f"recall@{top_k}"] = recall
        row[f"violation_rate@{top_k}"] = violation
        row[f"recall_minus_violation@{top_k}"] = gap
        gaps.append(gap)
    row["avg_recall_minus_violation"] = float(np.mean(gaps))
    row.update(extra)
    return row


def bm25_scores(corpus: list[str], queries: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(document) for document in corpus])
    rows = []
    for index, query in enumerate(queries, start=1):
        if index % 100 == 0:
            print(f"BM25 scoring {index}/{len(queries)}", flush=True)
        rows.append(bm25.get_scores(tokenize(query)).astype(np.float32))
    return np.vstack(rows)


def model_kwargs_for_dtype(dtype: str) -> dict:
    if dtype == "auto":
        return {}
    import torch

    return {"torch_dtype": getattr(torch, dtype)}


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (embeddings / norms).astype(np.float32)


def dense_backend_for_model(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    return "openai" if model_name.startswith("text-embedding-") else "sentence_transformers"


def load_dense_model(
    *,
    model_name: str,
    local_files_only: bool,
    trust_remote_code: bool,
    model_dtype: str,
):
    from sentence_transformers import SentenceTransformer

    kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    model_kwargs = model_kwargs_for_dtype(model_dtype)
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    try:
        return SentenceTransformer(model_name, **kwargs)
    except TypeError:
        # Older sentence-transformers versions may not expose every keyword.
        kwargs.pop("trust_remote_code", None)
        if "model_kwargs" in kwargs:
            kwargs.pop("model_kwargs", None)
        return SentenceTransformer(model_name, **kwargs)
    except Exception as error:
        print(f"Remote model load failed; retrying local cache only: {error}", flush=True)
        kwargs["local_files_only"] = True
        return SentenceTransformer(model_name, **kwargs)


def encode_dense(
    model,
    texts: list[str],
    *,
    batch_size: int,
    prompt_name: str | None = None,
) -> np.ndarray:
    kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    return model.encode(texts, **kwargs).astype(np.float32)


def openai_encoding_for_model(model_name: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def truncate_openai_text(text: str, *, encoding, max_tokens: int) -> str:
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def openai_batches(
    indexed_texts: list[tuple[int, str]],
    *,
    model_name: str,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
):
    encoding = openai_encoding_for_model(model_name)
    batch_indices: list[int] = []
    batch_texts: list[str] = []
    batch_tokens = 0
    for index, text in indexed_texts:
        truncated = truncate_openai_text(
            str(text),
            encoding=encoding,
            max_tokens=max_input_tokens,
        )
        token_count = len(encoding.encode(truncated))
        if batch_texts and (
            len(batch_texts) >= batch_size
            or batch_tokens + token_count > max_batch_tokens
        ):
            yield batch_indices, batch_texts
            batch_indices = []
            batch_texts = []
            batch_tokens = 0
        batch_indices.append(index)
        batch_texts.append(truncated)
        batch_tokens += token_count
    if batch_texts:
        yield batch_indices, batch_texts


def encode_openai(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    non_empty_texts = [
        (index, str(text))
        for index, text in enumerate(texts)
        if str(text).strip()
    ]
    empty_count = len(texts) - len(non_empty_texts)
    if empty_count:
        print(f"OpenAI embedding: using zero vectors for {empty_count} empty inputs", flush=True)
    embeddings_by_index: dict[int, list[float]] = {}
    total = len(texts)
    processed = 0
    for batch_indices, batch in openai_batches(
        non_empty_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_input_tokens=max_input_tokens,
    ):
        for attempt in range(6):
            try:
                response = client.embeddings.create(model=model_name, input=batch)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        ordered = sorted(response.data, key=lambda item: item.index)
        for original_index, item in zip(batch_indices, ordered):
            embeddings_by_index[original_index] = item.embedding
        processed += len(batch)
        print(f"OpenAI embedding {processed + empty_count}/{total}", flush=True)

    if embeddings_by_index:
        embedding_dim = len(next(iter(embeddings_by_index.values())))
    else:
        raise ValueError("OpenAI embedding input contains no non-empty texts.")
    embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
    for index, embedding in embeddings_by_index.items():
        embeddings[index] = np.asarray(embedding, dtype=np.float32)
    return l2_normalize(embeddings)


def openai_score_matrices(
    *,
    corpus: list[str],
    queries: list[str],
    targets: list[str],
    traps: list[str],
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_model = model_name.replace("/", "__")
    embedding_path = cache_dir / f"corpus_{safe_model}.npy"
    if embedding_path.exists():
        doc_embeddings = np.load(embedding_path)
    else:
        doc_embeddings = encode_openai(
            corpus,
            model_name=model_name,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            max_input_tokens=max_input_tokens,
        )
        np.save(embedding_path, doc_embeddings)

    all_queries = queries + targets + traps
    query_embeddings = encode_openai(
        all_queries,
        model_name=model_name,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_input_tokens=max_input_tokens,
    )
    scores = np.dot(query_embeddings, doc_embeddings.T).astype(np.float32)
    size = len(queries)
    return scores[:size], scores[size : 2 * size], scores[2 * size :]


def dense_score_matrices(
    *,
    corpus: list[str],
    queries: list[str],
    targets: list[str],
    traps: list[str],
    model_name: str,
    cache_dir: Path,
    local_files_only: bool,
    trust_remote_code: bool,
    query_prompt_name: str | None,
    model_dtype: str,
    doc_batch_size: int,
    query_batch_size: int,
    dense_backend: str,
    openai_max_batch_tokens: int,
    openai_max_input_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dense_backend_for_model(model_name, dense_backend) == "openai":
        return openai_score_matrices(
            corpus=corpus,
            queries=queries,
            targets=targets,
            traps=traps,
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=doc_batch_size,
            max_batch_tokens=openai_max_batch_tokens,
            max_input_tokens=openai_max_input_tokens,
        )

    model = load_dense_model(
        model_name=model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        model_dtype=model_dtype,
    )
    safe_model = model_name.replace("/", "__")
    embedding_path = cache_dir / f"corpus_{safe_model}.npy"
    if embedding_path.exists():
        doc_embeddings = np.load(embedding_path)
    else:
        doc_embeddings = encode_dense(
            model,
            corpus,
            batch_size=doc_batch_size,
        )
        np.save(embedding_path, doc_embeddings)

    all_queries = queries + targets + traps
    query_embeddings = encode_dense(
        model,
        all_queries,
        batch_size=query_batch_size,
        prompt_name=query_prompt_name,
    )
    scores = np.dot(query_embeddings, doc_embeddings.T).astype(np.float32)
    size = len(queries)
    return scores[:size], scores[size : 2 * size], scores[2 * size :]


def score_matrices(
    *,
    retriever: str,
    corpus: list[str],
    queries: list[str],
    targets: list[str],
    traps: list[str],
    dense_model: str,
    cache_dir: Path,
    local_files_only: bool,
    trust_remote_code: bool,
    query_prompt_name: str | None,
    model_dtype: str,
    doc_batch_size: int,
    query_batch_size: int,
    dense_backend: str,
    openai_max_batch_tokens: int,
    openai_max_input_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if retriever == "bm25":
        all_scores = bm25_scores(corpus, queries + targets + traps)
        size = len(queries)
        return all_scores[:size], all_scores[size : 2 * size], all_scores[2 * size :]
    return dense_score_matrices(
        corpus=corpus,
        queries=queries,
        targets=targets,
        traps=traps,
        model_name=dense_model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        query_prompt_name=query_prompt_name,
        model_dtype=model_dtype,
        doc_batch_size=doc_batch_size,
        query_batch_size=query_batch_size,
        dense_backend=dense_backend,
        openai_max_batch_tokens=openai_max_batch_tokens,
        openai_max_input_tokens=openai_max_input_tokens,
    )


def run_retriever(
    *,
    retriever: str,
    corpus: list[str],
    sample: pd.DataFrame,
    decompositions: list[dict[str, str]],
    alphas: list[float],
    betas: list[float],
    candidate_top_ns: list[int | None],
    dense_model: str,
    cache_dir: Path,
    local_files_only: bool,
    trust_remote_code: bool,
    query_prompt_name: str | None,
    model_dtype: str,
    doc_batch_size: int,
    query_batch_size: int,
    dense_backend: str,
    openai_max_batch_tokens: int,
    openai_max_input_tokens: int,
    retriever_label: str,
    top_ks: list[int],
    score_mode: str,
    save_score_matrix_path: Path | None,
    load_score_matrix_path: Path | None,
) -> list[dict]:
    queries = [str(row["query"]) for _, row in sample.iterrows()]
    targets = [decomposition["Q_target"] for decomposition in decompositions]
    traps = [decomposition["Q_trap"] for decomposition in decompositions]
    answer_indices = sample["positive_corpus_index"].to_numpy(dtype=int)
    trap_indices = sample["negative_corpus_index"].to_numpy(dtype=int)

    if load_score_matrix_path is not None:
        print(f"Loading score matrices: {load_score_matrix_path}", flush=True)
        matrices = np.load(load_score_matrix_path)
        baseline_scores = matrices["baseline_scores"]
        target_scores = matrices["target_scores"]
        trap_scores = matrices["trap_scores"]
    else:
        print(f"Scoring {retriever}", flush=True)
        baseline_scores, target_scores, trap_scores = score_matrices(
            retriever=retriever,
            corpus=corpus,
            queries=queries,
            targets=targets,
            traps=traps,
            dense_model=dense_model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            query_prompt_name=query_prompt_name,
            model_dtype=model_dtype,
            doc_batch_size=doc_batch_size,
            query_batch_size=query_batch_size,
            dense_backend=dense_backend,
            openai_max_batch_tokens=openai_max_batch_tokens,
            openai_max_input_tokens=openai_max_input_tokens,
        )

    if save_score_matrix_path is not None:
        print(f"Saving score matrices: {save_score_matrix_path}", flush=True)
        save_score_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_score_matrix_path,
            baseline_scores=baseline_scores,
            target_scores=target_scores,
            trap_scores=trap_scores,
            answer_indices=answer_indices,
            trap_indices=trap_indices,
            queries=np.array(queries, dtype=object),
            targets=np.array(targets, dtype=object),
            traps=np.array(traps, dtype=object),
        )

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
                if score_mode == "no_target":
                    final_scores = alpha * baseline_norm - beta * trap_norm
                    method_prefix = "score_anti_rrf_no_target"
                    weight_label = "a"
                    gamma = ""
                elif score_mode == "target_minus_trap":
                    final_scores = alpha * target_norm - beta * trap_norm
                    method_prefix = "score_target_minus_trap"
                    weight_label = "g"
                    gamma = alpha
                else:
                    final_scores = alpha * baseline_norm + target_norm - beta * trap_norm
                    method_prefix = "score_anti_rrf"
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
    with Path(args.corpus_json).open(encoding="utf-8") as handle:
        corpus = json.load(handle)
    decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)
    alphas = parse_float_list(args.gammas if args.gammas else args.alphas)
    betas = parse_float_list(args.betas)
    candidate_top_ns = parse_candidate_list(args.candidate_top_ns)
    top_ks = [args.top_k] if args.top_k is not None else parse_int_list(args.top_ks)

    retrievers = ["bm25", "dense"] if args.retriever == "both" else [args.retriever]
    summary_rows = []
    for retriever in retrievers:
        retriever_label = (
            "bm25"
            if retriever == "bm25"
            else args.retriever_label
            or args.dense_model.replace("/", "__")
        )
        summary_rows.extend(
            run_retriever(
                retriever=retriever,
                corpus=corpus,
                sample=sample,
                decompositions=decompositions,
                alphas=alphas,
                betas=betas,
                candidate_top_ns=candidate_top_ns,
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
                retriever_label=retriever_label,
                top_ks=top_ks,
                score_mode=args.score_mode,
                save_score_matrix_path=(
                    Path(args.save_score_matrix_path)
                    if args.save_score_matrix_path
                    else None
                ),
                load_score_matrix_path=(
                    Path(args.load_score_matrix_path)
                    if args.load_score_matrix_path
                    else None
                ),
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "score_anti_rrf_results.csv"
    summary.to_csv(summary_path, index=False)
    compatibility_summary_path = output_dir / "score_anti_rrf_sweep.csv"
    summary.to_csv(compatibility_summary_path, index=False)

    variants = summary[summary["method"] != "baseline"].copy()
    primary_k = min(top_ks)
    primary_gap = f"recall_minus_violation@{primary_k}"
    primary_recall = f"recall@{primary_k}"
    primary_violation = f"violation_rate@{primary_k}"
    top = variants.sort_values(
        ["avg_recall_minus_violation", primary_gap, "right_rank", primary_recall, primary_violation],
        ascending=[False, False, False, False, True],
    )
    top_path = output_dir / "score_anti_rrf_best_configs.csv"
    top.head(50).to_csv(top_path, index=False)
    compatibility_top_path = output_dir / "score_anti_rrf_top_configs.csv"
    top.head(50).to_csv(compatibility_top_path, index=False)

    print(f"Summary: {summary_path}")
    print(f"Best configs: {top_path}")
    print()
    print(top.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
