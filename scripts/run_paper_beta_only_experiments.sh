#!/usr/bin/env bash
set -euo pipefail

# Reproducible beta-only experiments for ExcluIR and BoolQuestions.
#
# Usage:
#   bash scripts/run_paper_beta_only_experiments.sh
#
# Environment overrides:
#   PYTHON_BIN=.venv/bin/python
#   REWRITER_MODEL=gpt-4o-mini
#   BETA_VALUES=0,0.01,...,1.00
#   RUN_EXCLUIR_FULL=1
#   RUN_BOOLQUESTIONS=1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REWRITER_MODEL="${REWRITER_MODEL:-gpt-4o-mini}"
BETA_VALUES="${BETA_VALUES:-$(python - <<'PY'
print(",".join(f"{i/100:.2f}" for i in range(101)))
PY
)}"
RUN_EXCLUIR_FULL="${RUN_EXCLUIR_FULL:-1}"
RUN_BOOLQUESTIONS="${RUN_BOOLQUESTIONS:-1}"
PROMPT_PATH="${PROMPT_PATH:-prompts/excluir_rewriter_gpt4o_mini/v3_oracle_style_trap_system.txt}"

MODELS=(
  "text-embedding-3-small|text-embedding-3-small|text_embedding_3_small|openai|"
  "text-embedding-3-large|text-embedding-3-large|text_embedding_3_large|openai|"
  "Qwen/Qwen3-Embedding-0.6B|Qwen/Qwen3-Embedding-0.6B|qwen3_embedding_0_6b|sentence_transformers|query"
  "Qwen/Qwen3-Embedding-4B|Qwen/Qwen3-Embedding-4B|qwen3_embedding_4b|sentence_transformers|query"
)

run_model_matrix() {
  local dataset_key="$1"
  local sample_csv="$2"
  local decomp_jsonl="$3"
  local corpus_json="$4"
  local cache_dir="$5"
  local matrix_dir="$6"

  mkdir -p "${matrix_dir}"
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_label dense_model safe_label backend query_prompt <<<"${spec}"
    local matrix_path="${matrix_dir}/${dataset_key}_${safe_label}_v3_oracle_style_trap.npz"
    local output_dir="results/${dataset_key}_${safe_label}_v3_oracle_style_trap_matrix_build"
    local prompt_args=()
    if [[ -n "${query_prompt}" ]]; then
      prompt_args+=(--query-prompt-name "${query_prompt}")
    fi
    if [[ -f "${matrix_path}" ]]; then
      echo "[skip] existing matrix ${matrix_path}"
      continue
    fi
    echo "[matrix] ${dataset_key} ${model_label}"
    "${PYTHON_BIN}" scripts/fast_excluir_score_anti_rrf_sweep.py \
      --sample-csv "${sample_csv}" \
      --decompositions-jsonl "${decomp_jsonl}" \
      --corpus-json "${corpus_json}" \
      --output-dir "${output_dir}" \
      --retriever dense \
      --dense-model "${dense_model}" \
      --dense-backend "${backend}" \
      --retriever-label "${model_label}" \
      --cache-dir "${cache_dir}" \
      --save-score-matrix-path "${matrix_path}" \
      --candidate-top-ns all \
      --alphas 1 \
      --betas 0 \
      --top-ks 3,5,7,9 \
      --trust-remote-code \
      --model-dtype auto \
      --dense-doc-batch-size 64 \
      --dense-query-batch-size 64 \
      "${prompt_args[@]}"
  done
}

run_beta_sweeps() {
  local dataset_key="$1"
  local matrix_dir="$2"
  local output_root="$3"
  local baseline_dir="${output_root}/baseline_minus_q_trap"
  local target_dir="${output_root}/q_target_minus_q_trap"
  local matrix_args=()

  for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_label _dense_model safe_label _backend _query_prompt <<<"${spec}"
    matrix_args+=(--matrix "${model_label}" "${matrix_dir}/${dataset_key}_${safe_label}_v3_oracle_style_trap.npz")
  done

  "${PYTHON_BIN}" scripts/run_excluir_beta_only_sweep.py \
    "${matrix_args[@]}" \
    --score-source baseline \
    --betas "${BETA_VALUES}" \
    --output-dir "${baseline_dir}"

  "${PYTHON_BIN}" scripts/run_excluir_beta_only_sweep.py \
    "${matrix_args[@]}" \
    --score-source target \
    --betas "${BETA_VALUES}" \
    --output-dir "${target_dir}"
}

if [[ "${RUN_EXCLUIR_FULL}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/setup_excluir_sample.py \
    --sample-size 3452 \
    --preserve-order \
    --output-stem excluir_manual_full \
    --output-dir data

  "${PYTHON_BIN}" scripts/generate_excluir_rewrites.py \
    --sample-csv data/excluir_manual_full.csv \
    --output-jsonl outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap_full/decompositions.jsonl \
    --system-prompt-path "${PROMPT_PATH}" \
    --model "${REWRITER_MODEL}" \
    --workers 4 \
    --temperature 0

  run_model_matrix \
    "excluir_full" \
    "data/excluir_manual_full.csv" \
    "outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap_full/decompositions.jsonl" \
    "data/excluir_raw/corpus.json" \
    "data/excluir_cache" \
    "results/score_matrices"

  run_beta_sweeps \
    "excluir_full" \
    "results/score_matrices" \
    "results/paper_beta_only/excluir_full"
fi

if [[ "${RUN_BOOLQUESTIONS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/setup_boolquestions_not_sample.py \
    --output-dir data/boolquestions_not_323 \
    --output-stem boolquestions_not_323

  "${PYTHON_BIN}" scripts/generate_excluir_rewrites.py \
    --sample-csv data/boolquestions_not_323/boolquestions_not_323.csv \
    --output-jsonl outputs/boolquestions_not_323_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl \
    --system-prompt-path "${PROMPT_PATH}" \
    --model "${REWRITER_MODEL}" \
    --workers 4 \
    --temperature 0

  run_model_matrix \
    "boolquestions_not_323" \
    "data/boolquestions_not_323/boolquestions_not_323.csv" \
    "outputs/boolquestions_not_323_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl" \
    "data/boolquestions_not_323/corpus.json" \
    "data/boolquestions_not_323/cache" \
    "results/score_matrices"

  run_beta_sweeps \
    "boolquestions_not_323" \
    "results/score_matrices" \
    "results/paper_beta_only/boolquestions_not_323"
fi

COMBINE_INPUTS=()
if [[ -f results/paper_beta_only/excluir_full/baseline_minus_q_trap/beta_only_sweep_results.csv ]]; then
  COMBINE_INPUTS+=(--input ExcluIR full baseline-q_trap results/paper_beta_only/excluir_full/baseline_minus_q_trap/beta_only_sweep_results.csv)
  COMBINE_INPUTS+=(--input ExcluIR full q_target-q_trap results/paper_beta_only/excluir_full/q_target_minus_q_trap/beta_only_sweep_results.csv)
fi
if [[ -f results/paper_beta_only/boolquestions_not_323/baseline_minus_q_trap/beta_only_sweep_results.csv ]]; then
  COMBINE_INPUTS+=(--input BoolQuestions not-323 baseline-q_trap results/paper_beta_only/boolquestions_not_323/baseline_minus_q_trap/beta_only_sweep_results.csv)
  COMBINE_INPUTS+=(--input BoolQuestions not-323 q_target-q_trap results/paper_beta_only/boolquestions_not_323/q_target_minus_q_trap/beta_only_sweep_results.csv)
fi
if [[ "${#COMBINE_INPUTS[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" scripts/combine_beta_only_sweeps.py \
    "${COMBINE_INPUTS[@]}" \
    --output-csv results/paper_beta_only/combined_beta_only_sweep.csv \
    --output-tsv results/paper_beta_only/combined_beta_only_sweep.tsv
fi

echo "Done. Results are under results/paper_beta_only and matrices are under results/score_matrices."
