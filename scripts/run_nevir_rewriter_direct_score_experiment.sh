#!/usr/bin/env bash
set -euo pipefail

SAMPLE_CSV="${SAMPLE_CSV:-data/nevir_mini_train_q2_1000_seed42.csv}"
DECOMPOSITIONS_JSONL="${DECOMPOSITIONS_JSONL:-outputs/nevir_rewriter_gpt4o_mini_v2_recall_preserving_q2/decompositions.jsonl}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-rewriter_gpt4o_mini_v2_recall_preserving_q2}"
TOP_KS="${TOP_KS:-1,3,5,10}"
CANDIDATE_TOP_NS="${CANDIDATE_TOP_NS:-5,10,20,all}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
GAMMAS="${GAMMAS:-0,0.25,0.5,0.75,1}"
BETAS="${BETAS:-0.3,0.5,0.75,1}"

run_one() {
  local label="$1"
  local model="$2"
  local backend="$3"
  local trust_remote_code="$4"
  local query_prompt_name="${5:-}"
  local model_dtype="${6:-auto}"

  local matrix_path="results/score_matrices/nevir_1000_${label}_${EXPERIMENT_TAG}.npz"
  local cache_dir="data/nevir_cache/nevir_1000_${label}_${EXPERIMENT_TAG}"

  local common_args=(
    --sample-csv "$SAMPLE_CSV"
    --decompositions-jsonl "$DECOMPOSITIONS_JSONL"
    --dense-model "$model"
    --dense-backend "$backend"
    --retriever-label "$label"
    --dense-doc-batch-size 64
    --dense-query-batch-size 64
    --cache-dir "$cache_dir"
    --top-ks "$TOP_KS"
    --candidate-top-ns "$CANDIDATE_TOP_NS"
    --alphas "$ALPHAS"
    --gammas "$GAMMAS"
    --betas "$BETAS"
    --model-dtype "$model_dtype"
  )

  if [[ "$trust_remote_code" == "true" ]]; then
    common_args+=(--trust-remote-code)
  fi
  if [[ -n "$query_prompt_name" ]]; then
    common_args+=(--query-prompt-name "$query_prompt_name")
  fi

  python scripts/fast_nevir_score_anti_rrf_sweep.py \
    "${common_args[@]}" \
    --score-mode no_target \
    --save-score-matrix-path "$matrix_path" \
    --output-dir "results/nevir_1000_${label}_${EXPERIMENT_TAG}_no_target"

  python scripts/fast_nevir_score_anti_rrf_sweep.py \
    "${common_args[@]}" \
    --score-mode target_minus_trap \
    --load-score-matrix-path "$matrix_path" \
    --output-dir "results/nevir_1000_${label}_${EXPERIMENT_TAG}_target_minus_trap"
}

run_one "text_embedding_3_small" "text-embedding-3-small" "openai" "false"
run_one "qwen3_embedding_0_6b" "Qwen/Qwen3-Embedding-0.6B" "sentence_transformers" "true" "query"
run_one "qwen3_embedding_4b" "Qwen/Qwen3-Embedding-4B" "sentence_transformers" "true" "query" "float16"
