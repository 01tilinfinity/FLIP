#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-all}"
SAMPLE_CSV="${SAMPLE_CSV:-data/excluir_manual_1000_seed42.csv}"
DECOMPOSITIONS_JSONL="${DECOMPOSITIONS_JSONL:-outputs/excluir_rewriter_gpt4o_mini/decompositions.jsonl}"
CORPUS_JSON="${CORPUS_JSON:-data/excluir_raw/corpus.json}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1,1.5,2}"
GAMMAS="${GAMMAS:-0.25,0.5,0.75,1,1.5,2}"
BETAS="${BETAS:-0.1,0.2,0.3,0.5,0.75,1}"
CANDIDATE_TOP_NS="${CANDIDATE_TOP_NS:-5,10,20,all}"
TOP_KS="${TOP_KS:-3,5,7,9}"

mkdir -p logs results/score_matrices

run_model() {
  local preset="$1"
  local model="$2"
  local backend="$3"
  local slug="$4"
  local label="$5"
  local doc_batch="$6"
  local query_batch="$7"
  shift 7
  local extra_args=("$@")

  local matrix_path="results/score_matrices/excluir_1000_${slug}_rewriter_gpt4o_mini.npz"
  local no_target_dir="results/excluir_1000_${slug}_rewriter_gpt4o_mini_no_target_top3579"
  local target_dir="results/excluir_1000_${slug}_rewriter_gpt4o_mini_target_minus_trap_top3579"
  local no_target_log="logs/${slug}_excluir_1000_rewriter_gpt4o_mini_no_target_top3579.log"
  local target_log="logs/${slug}_excluir_1000_rewriter_gpt4o_mini_target_minus_trap_top3579.log"

  echo "Running ${preset}: baseline-minus-trap and target-minus-trap"

  python scripts/fast_excluir_score_anti_rrf_sweep.py \
    --sample-csv "${SAMPLE_CSV}" \
    --decompositions-jsonl "${DECOMPOSITIONS_JSONL}" \
    --corpus-json "${CORPUS_JSON}" \
    --retriever dense \
    --dense-backend "${backend}" \
    --dense-model "${model}" \
    --retriever-label "${label}" \
    --dense-doc-batch-size "${doc_batch}" \
    --dense-query-batch-size "${query_batch}" \
    --score-mode no_target \
    --alphas "${ALPHAS}" \
    --betas "${BETAS}" \
    --candidate-top-ns "${CANDIDATE_TOP_NS}" \
    --top-ks "${TOP_KS}" \
    --save-score-matrix-path "${matrix_path}" \
    --output-dir "${no_target_dir}" \
    "${extra_args[@]}" \
    2>&1 | tee "${no_target_log}"

  python scripts/fast_excluir_score_anti_rrf_sweep.py \
    --sample-csv "${SAMPLE_CSV}" \
    --decompositions-jsonl "${DECOMPOSITIONS_JSONL}" \
    --corpus-json "${CORPUS_JSON}" \
    --retriever dense \
    --dense-backend "${backend}" \
    --dense-model "${model}" \
    --retriever-label "${label}-target-minus-trap" \
    --score-mode target_minus_trap \
    --gammas "${GAMMAS}" \
    --betas "${BETAS}" \
    --candidate-top-ns "${CANDIDATE_TOP_NS}" \
    --top-ks "${TOP_KS}" \
    --load-score-matrix-path "${matrix_path}" \
    --output-dir "${target_dir}" \
    2>&1 | tee "${target_log}"
}

run_preset() {
  case "$1" in
    bge-m3)
      run_model "$1" "BAAI/bge-m3" "sentence_transformers" "bge_m3" "bge-m3-rewriter-gpt4o-mini" 64 64 \
        --trust-remote-code --model-dtype bfloat16
      ;;
    qwen3-0.6b)
      run_model "$1" "Qwen/Qwen3-Embedding-0.6B" "sentence_transformers" "qwen3_embedding_0_6b" "qwen3-embedding-0.6b-rewriter-gpt4o-mini" 32 64 \
        --trust-remote-code --query-prompt-name query --model-dtype bfloat16
      ;;
    qwen3-4b)
      run_model "$1" "Qwen/Qwen3-Embedding-4B" "sentence_transformers" "qwen3_embedding_4b" "qwen3-embedding-4b-rewriter-gpt4o-mini" 8 16 \
        --trust-remote-code --query-prompt-name query --model-dtype bfloat16
      ;;
    openai-small)
      run_model "$1" "text-embedding-3-small" "openai" "text_embedding_3_small" "text-embedding-3-small-rewriter-gpt4o-mini" 256 256
      ;;
    all)
      run_preset bge-m3
      run_preset qwen3-0.6b
      run_preset qwen3-4b
      run_preset openai-small
      ;;
    *)
      echo "Unknown preset: $1" >&2
      echo "Usage: $0 [all|bge-m3|qwen3-0.6b|qwen3-4b|openai-small]" >&2
      exit 2
      ;;
  esac
}

run_preset "${PRESET}"
