#!/usr/bin/env bash
set -euo pipefail

python scripts/fast_excluir_score_anti_rrf_sweep.py \
  --sample-csv data/excluir_manual_1000_seed42.csv \
  --decompositions-jsonl data/excluir_manual_1000_seed42_decompositions.jsonl \
  --corpus-json data/excluir_raw/corpus.json \
  --retriever dense \
  --dense-model BAAI/bge-m3 \
  --retriever-label bge-m3 \
  --trust-remote-code \
  --model-dtype bfloat16 \
  --dense-doc-batch-size 64 \
  --dense-query-batch-size 64 \
  --candidate-top-ns 5,10,20,all \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0.3,0.5,0.75,1 \
  --top-ks 3,5,7,9 \
  --score-mode no_target \
  --cache-dir data/excluir_cache/bge_m3 \
  --output-dir results/excluir_1000_bge_m3_score_anti_rrf_no_target_top3579 \
  2>&1 | tee logs/bge_m3_excluir_1000_no_target_top3579.log

python scripts/fast_excluir_score_anti_rrf_sweep.py \
  --sample-csv data/excluir_manual_1000_seed42.csv \
  --decompositions-jsonl data/excluir_manual_1000_seed42_decompositions.jsonl \
  --corpus-json data/excluir_raw/corpus.json \
  --retriever dense \
  --dense-model infly/inf-retriever-v1 \
  --retriever-label inf-retriever-v1 \
  --trust-remote-code \
  --query-prompt-name query \
  --model-dtype bfloat16 \
  --dense-doc-batch-size 64 \
  --dense-query-batch-size 64 \
  --candidate-top-ns 5,10,20,all \
  --alphas 0,0.25,0.5,0.75,1 \
  --betas 0.3,0.5,0.75,1 \
  --top-ks 3,5,7,9 \
  --score-mode no_target \
  --cache-dir data/excluir_cache/inf_retriever_v1 \
  --output-dir results/excluir_1000_inf_retriever_v1_score_anti_rrf_no_target_top3579 \
  2>&1 | tee logs/inf_retriever_excluir_1000_no_target_top3579.log
