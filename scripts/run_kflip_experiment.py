#!/usr/bin/env python3
"""Run K-FLIP local retrieval, query decomposition, reranking, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DECOMPOSITION_PROMPT = """You are an expert Query Decomposition Module for an advanced Information Retrieval system. 
Your task is to analyze the user's input query ($Q_{user}$), which contains a negative constraint or a contrastive negation, and decompose it into two distinct search queries: $Q_{target}$ and $Q_{trap}$.

Strictly follow these rules to generate the output:

1. **Identify the Core Intent ($Q_{target}$):**
   - Extract the positive target concept that the user actually wants to retrieve.
   - Expand this query using synonyms, technical terms, or alternative phrasing that represents the desired search space.
   - Do NOT include any terms or concepts that the user wants to avoid or minimize.
   - Preserve all entities, roles, objects, and contextual qualifiers from the original query.
   - Start from the original query's full meaning, then expand it; do not drop the main noun phrase that the question asks about.

2. **Identify the Exclusion Trap ($Q_{trap}$):**
   - Identify the specific condition, technical method, or contrastive state that the user explicitly wants to exclude, avoid, or minimize.
   - Expand this trap query into a comma-separated list of highly related negative keywords, sub-concepts, or opposing attributes.
   - This pool must capture the exact lexical and semantic landscape that a naive search engine might mistakenly retrieve as a false positive.

[CRITICAL RULE FOR NEGATIVE INTENT / CONTRASTIVE QUERIES]
- Carefully distinguish between "the target of the question" and "the constraint to exclude."
- If the user's core question inherently asks for a negative state, failure, or deficiency (e.g., "who failed to...", "not achieved", "without success"), that negative concept belongs entirely to $Q_{target}$, NOT $Q_{trap}$.
- In such inverse cases, $Q_{trap}$ must capture the opposite successful state, active accomplishments, or positive lexical magnets (e.g., "success", "achieved", "completed") that a naive search engine would incorrectly rank highly.
- If the query asks "what/who/which X had no/not/without Y", then "$X with no/not/without Y" is the target. Do NOT put Y alone into $Q_{trap}$.
- In inverse cases, $Q_{trap}$ should preserve the same entities and context anchors as $Q_{target}$ while flipping only the polarity or contrastive state.
- Avoid generic trap keywords such as "success" or "achievement" unless they are the concrete opposite of the original query. Prefer specific lexical opposites grounded in the original wording.

3. **Output Format Requirement:**
   - You must output the result strictly in the following JSON format. Do not include any conversational text or markdown explanation outside the JSON block.

{
  "Q_target": "Expanded target query string for the positive search stream",
  "Q_trap": "Comma-separated list of keywords representing the negative/excluded search space"
}

User Query ($Q_{user}$): "{user_query}"
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    rank: int


class CorpusPool:
    def __init__(self, sample: pd.DataFrame) -> None:
        self.documents: list[dict[str, str]] = []
        for row in sample.to_dict(orient="records"):
            row_id = str(row["id"])
            for doc_column in ("doc1", "doc2"):
                self.documents.append(
                    {
                        "doc_id": f"{row_id}::{doc_column}",
                        "sample_id": row_id,
                        "doc_column": doc_column,
                        "text": str(row[doc_column]),
                    }
                )
        self.doc_ids = [doc["doc_id"] for doc in self.documents]
        self.texts = [doc["text"] for doc in self.documents]


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: CorpusPool) -> None:
        self.corpus = corpus
        self.bm25 = BM25Okapi([tokenize(text) for text in corpus.texts])

    def search(self, query: str) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize(query))
        return ranked_results(self.corpus.doc_ids, scores)


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        corpus: CorpusPool,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            corpus.texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(self.doc_embeddings, query_embedding)
        return ranked_results(self.corpus.doc_ids, scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-FLIP Step 2-5 over a NevIR mini sample."
    )
    parser.add_argument(
        "--sample-csv",
        default="data/nevir_mini_train_q1_50_seed42.csv",
        help="Step 1 sample CSV path.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--retriever", choices=("bm25", "dense", "both"), default="both")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for dense retrieval.",
    )
    parser.add_argument(
        "--decomposition-mode",
        choices=("openai", "heuristic"),
        default="openai",
        help="Use OpenAI decomposition or a no-network NevIR q-pair fallback.",
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Fallback to the q-pair heuristic if OpenAI credentials/model fail.",
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=None,
        help="Restrict Anti-RRF reranking to the baseline top-N candidate pool.",
    )
    parser.add_argument(
        "--decompositions-jsonl",
        default=None,
        help="Reuse cached query decompositions instead of calling OpenAI.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ranked_results(doc_ids: list[str], scores: np.ndarray) -> list[SearchResult]:
    order = sorted(range(len(doc_ids)), key=lambda idx: (-float(scores[idx]), idx))
    return [
        SearchResult(doc_id=doc_ids[idx], score=float(scores[idx]), rank=rank)
        for rank, idx in enumerate(order, start=1)
    ]


def to_rank_dict(results: list[SearchResult]) -> dict[str, int]:
    return {result.doc_id: result.rank for result in results}


def load_sample(path: Path, max_samples: int | None) -> pd.DataFrame:
    sample = pd.read_csv(path)
    required = {"id", "query", "q1", "q2", "doc1", "doc2", "answer_doc", "trap_doc"}
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Sample CSV is missing required columns: {missing}")
    if max_samples is not None:
        sample = sample.head(max_samples).copy()
    return sample.reset_index(drop=True)


def make_openai_decomposition(query: str, model: str) -> dict[str, str]:
    client = OpenAI()
    prompt = DECOMPOSITION_PROMPT.replace("{user_query}", query)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty decomposition response")
    return parse_decomposition_json(content)


def make_heuristic_decomposition(row: pd.Series) -> dict[str, str]:
    target_query = str(row["query"])
    query_column = row.get("query_column", "q1")
    trap_query = str(row["q2"] if query_column == "q1" else row["q1"])
    return {"Q_target": target_query, "Q_trap": trap_query}


def parse_decomposition_json(content: str) -> dict[str, str]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Decomposition response must be a JSON object")
    target = parsed.get("Q_target")
    trap = parsed.get("Q_trap")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Decomposition response is missing Q_target")
    if not isinstance(trap, str):
        raise ValueError("Decomposition response is missing Q_trap")
    return {"Q_target": target.strip(), "Q_trap": trap.strip()}


def decompose_queries(
    sample: pd.DataFrame,
    *,
    mode: Literal["openai", "heuristic"],
    model: str,
    allow_heuristic_fallback: bool,
) -> list[dict[str, str]]:
    decompositions: list[dict[str, str]] = []
    has_openai_credentials = bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_ADMIN_KEY")
        or os.getenv("OPENAI_WORKLOAD_IDENTITY")
    )
    if mode == "openai" and not has_openai_credentials:
        if allow_heuristic_fallback:
            mode = "heuristic"
        else:
            raise RuntimeError(
                "OpenAI credentials are missing. Set OPENAI_API_KEY in .env, "
                "or rerun with --allow-heuristic-fallback for a structural dry run."
            )

    total = len(sample)
    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        print(f"Decomposing query {index}/{total}: {row['id']}", flush=True)
        if mode == "heuristic":
            decompositions.append(make_heuristic_decomposition(row))
            continue

        try:
            decompositions.append(make_openai_decomposition(str(row["query"]), model))
        except Exception:
            if not allow_heuristic_fallback:
                raise
            decompositions.append(make_heuristic_decomposition(row))
    return decompositions


def load_decompositions(path: Path, sample: pd.DataFrame) -> list[dict[str, str]]:
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_id[str(row["id"])] = parse_decomposition_json(
                json.dumps(
                    {"Q_target": row["Q_target"], "Q_trap": row["Q_trap"]},
                    ensure_ascii=False,
                )
            )

    missing = [str(row["id"]) for _, row in sample.iterrows() if str(row["id"]) not in by_id]
    if missing:
        raise ValueError(f"Cached decomposition file is missing ids: {missing[:5]}")
    return [by_id[str(row["id"])] for _, row in sample.iterrows()]


def anti_rrf_rank(
    corpus: CorpusPool,
    target_results: list[SearchResult],
    trap_results: list[SearchResult],
    *,
    k: int,
    beta: float,
    candidate_doc_ids: list[str] | None = None,
    baseline_results: list[SearchResult] | None = None,
) -> list[SearchResult]:
    target_ranks = to_rank_dict(target_results)
    trap_ranks = to_rank_dict(trap_results)
    worst_rank = len(corpus.doc_ids) + 1
    doc_ids = candidate_doc_ids if candidate_doc_ids is not None else corpus.doc_ids

    scores = {}
    for doc_id in doc_ids:
        r_target = target_ranks.get(doc_id, worst_rank)
        r_trap = trap_ranks.get(doc_id, worst_rank)
        scores[doc_id] = (1.0 / (k + r_target)) - beta * (1.0 / (k + r_trap))

    if candidate_doc_ids is None:
        return ranked_results(corpus.doc_ids, np.array([scores[doc_id] for doc_id in corpus.doc_ids]))

    if baseline_results is None:
        raise ValueError("baseline_results is required when candidate_doc_ids is provided")

    baseline_ranks = to_rank_dict(baseline_results)
    sorted_candidates = sorted(
        candidate_doc_ids,
        key=lambda doc_id: (-scores[doc_id], baseline_ranks[doc_id]),
    )
    candidate_set = set(candidate_doc_ids)
    remainder = [result.doc_id for result in baseline_results if result.doc_id not in candidate_set]
    final_doc_ids = sorted_candidates + remainder

    return [
        SearchResult(doc_id=doc_id, score=scores.get(doc_id, float("-inf")), rank=rank)
        for rank, doc_id in enumerate(final_doc_ids, start=1)
    ]



def evaluate_rankings(
    sample: pd.DataFrame,
    rankings: dict[str, list[SearchResult]],
    *,
    method: str,
    retriever: str,
    top_k: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rows = []
    for _, row in sample.iterrows():
        sample_id = str(row["id"])
        answer_doc_id = f"{sample_id}::{row['answer_doc']}"
        trap_doc_id = f"{sample_id}::{row['trap_doc']}"
        ranks = to_rank_dict(rankings[sample_id])
        answer_rank = ranks[answer_doc_id]
        trap_rank = ranks[trap_doc_id]
        rows.append(
            {
                "retriever": retriever,
                "method": method,
                "id": sample_id,
                "query": row["query"],
                "answer_doc_id": answer_doc_id,
                "trap_doc_id": trap_doc_id,
                "answer_rank": answer_rank,
                "trap_rank": trap_rank,
                "recall_at_k": int(answer_rank <= top_k),
                "violation_at_k": int(trap_rank <= top_k),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "retriever": retriever,
        "method": method,
        "samples": int(len(df)),
        f"recall@{top_k}": float(df["recall_at_k"].mean()),
        f"violation_rate@{top_k}": float(df["violation_at_k"].mean()),
        "mean_answer_rank": float(df["answer_rank"].mean()),
        "mean_trap_rank": float(df["trap_rank"].mean()),
    }
    return summary, rows


def run_for_retriever(
    sample: pd.DataFrame,
    corpus: CorpusPool,
    retriever: BM25Retriever | DenseRetriever,
    decompositions: list[dict[str, str]],
    *,
    k: int,
    beta: float,
    top_k: int,
    candidate_top_n: int | None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    baseline_rankings: dict[str, list[SearchResult]] = {}
    kflip_rankings: dict[str, list[SearchResult]] = {}

    for decomposition, (_, row) in zip(decompositions, sample.iterrows(), strict=True):
        sample_id = str(row["id"])
        baseline_rankings[sample_id] = retriever.search(str(row["query"]))
        candidate_doc_ids = None
        if candidate_top_n is not None:
            candidate_doc_ids = [
                result.doc_id for result in baseline_rankings[sample_id][:candidate_top_n]
            ]
        target_results = retriever.search(decomposition["Q_target"])
        trap_results = retriever.search(decomposition["Q_trap"])
        kflip_rankings[sample_id] = anti_rrf_rank(
            corpus,
            target_results,
            trap_results,
            k=k,
            beta=beta,
            candidate_doc_ids=candidate_doc_ids,
            baseline_results=baseline_rankings[sample_id],
        )

    baseline_summary, baseline_rows = evaluate_rankings(
        sample,
        baseline_rankings,
        method="baseline",
        retriever=retriever.name,
        top_k=top_k,
    )
    kflip_method = "kflip" if candidate_top_n is None else f"kflip_top{candidate_top_n}"
    kflip_summary, kflip_rows = evaluate_rankings(
        sample,
        kflip_rankings,
        method=kflip_method,
        retriever=retriever.name,
        top_k=top_k,
    )
    return [baseline_summary, kflip_summary], baseline_rows + kflip_rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    args = parse_args()
    sample_path = Path(args.sample_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(sample_path, args.max_samples)
    corpus = CorpusPool(sample)
    if args.decompositions_jsonl:
        decompositions = load_decompositions(Path(args.decompositions_jsonl), sample)
    else:
        decompositions = decompose_queries(
            sample,
            mode=args.decomposition_mode,
            model=args.openai_model,
            allow_heuristic_fallback=args.allow_heuristic_fallback,
        )

    decomposition_rows = []
    for decomposition, (_, row) in zip(decompositions, sample.iterrows(), strict=True):
        decomposition_rows.append(
            {
                "id": str(row["id"]),
                "query": str(row["query"]),
                "Q_target": decomposition["Q_target"],
                "Q_trap": decomposition["Q_trap"],
            }
        )
    write_jsonl(decomposition_rows, output_dir / "query_decompositions.jsonl")

    retrievers: list[BM25Retriever | DenseRetriever] = []
    if args.retriever in ("bm25", "both"):
        retrievers.append(BM25Retriever(corpus))
    if args.retriever in ("dense", "both"):
        retrievers.append(DenseRetriever(corpus, model_name=args.dense_model))

    summaries = []
    detail_rows = []
    for retriever in retrievers:
        retriever_summaries, retriever_details = run_for_retriever(
            sample,
            corpus,
            retriever,
            decompositions,
            k=args.k,
            beta=args.beta,
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
        )
        summaries.extend(retriever_summaries)
        detail_rows.extend(retriever_details)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(detail_rows)
    summary_df.to_csv(output_dir / "scoreboard.csv", index=False)
    detail_df.to_csv(output_dir / "ranking_details.csv", index=False)

    with (output_dir / "scoreboard.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    print(f"Sample rows: {len(sample)}")
    print(f"Corpus documents: {len(corpus.doc_ids)}")
    print(f"Decompositions: {output_dir / 'query_decompositions.jsonl'}")
    print(f"Ranking details: {output_dir / 'ranking_details.csv'}")
    print(f"Scoreboard: {output_dir / 'scoreboard.csv'}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
