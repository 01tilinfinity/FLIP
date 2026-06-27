# ExcluIR 1000 - rewriter_gpt4o_mini_v3_oracle_style_trap

Date: 2026-06-27

## Prompt

- `prompts/excluir_rewriter_gpt4o_mini/v3_oracle_style_trap_system.txt`
- Prompt policy: preserve recall while making `q_trap` closer to the earlier oracle-style trap anchor extraction.

## Sample

- `data/excluir_manual_1000_seed42.csv`
- Rows: 1000

## Decomposition

```bash
python scripts/generate_excluir_rewrites.py \
  --sample-csv data/excluir_manual_1000_seed42.csv \
  --output-jsonl outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl \
  --system-prompt-path prompts/excluir_rewriter_gpt4o_mini/v3_oracle_style_trap_system.txt \
  --model gpt-4o-mini \
  --query-column query \
  --temperature 0 \
  --workers 2 \
  --max-retries 10
```

Local outputs:

- `outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl`
- `logs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap_generate.log`

## Scoring

```bash
EXPERIMENT_TAG=rewriter_gpt4o_mini_v3_oracle_style_trap \
DECOMPOSITIONS_JSONL=outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl \
./scripts/run_excluir_rewriter_direct_score_experiment.sh qwen3-0.6b

EXPERIMENT_TAG=rewriter_gpt4o_mini_v3_oracle_style_trap \
DECOMPOSITIONS_JSONL=outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl \
./scripts/run_excluir_rewriter_direct_score_experiment.sh qwen3-4b

EXPERIMENT_TAG=rewriter_gpt4o_mini_v3_oracle_style_trap \
DECOMPOSITIONS_JSONL=outputs/excluir_rewriter_gpt4o_mini_v3_oracle_style_trap/decompositions.jsonl \
./scripts/run_excluir_rewriter_direct_score_experiment.sh openai-small
```

Local summaries:

- `results/summary_rewriter_gpt4o_mini_v3_oracle_style_trap_goal_check.csv`
- `results/summary_rewriter_gpt4o_mini_v3_oracle_style_trap_all_configs.csv`

Local score matrices:

- `results/score_matrices/excluir_1000_qwen3_embedding_0_6b_rewriter_gpt4o_mini_v3_oracle_style_trap.npz`
- `results/score_matrices/excluir_1000_qwen3_embedding_4b_rewriter_gpt4o_mini_v3_oracle_style_trap.npz`
- `results/score_matrices/excluir_1000_text_embedding_3_small_rewriter_gpt4o_mini_v3_oracle_style_trap.npz`

Local per-model result directories:

- `results/excluir_1000_qwen3_embedding_0_6b_rewriter_gpt4o_mini_v3_oracle_style_trap_no_target_top3579/`
- `results/excluir_1000_qwen3_embedding_0_6b_rewriter_gpt4o_mini_v3_oracle_style_trap_target_minus_trap_top3579/`
- `results/excluir_1000_qwen3_embedding_4b_rewriter_gpt4o_mini_v3_oracle_style_trap_no_target_top3579/`
- `results/excluir_1000_qwen3_embedding_4b_rewriter_gpt4o_mini_v3_oracle_style_trap_target_minus_trap_top3579/`
- `results/excluir_1000_text_embedding_3_small_rewriter_gpt4o_mini_v3_oracle_style_trap_no_target_top3579/`
- `results/excluir_1000_text_embedding_3_small_rewriter_gpt4o_mini_v3_oracle_style_trap_target_minus_trap_top3579/`

## Quick Result

- `text-embedding-3-small`, `baseline - q_trap`: strict goal passed. Best `candidate_top_n=all`, `alpha=0.75`, `beta=0.1`.
- `Qwen/Qwen3-Embedding-0.6B`, `baseline - q_trap`: strict goal passed. Best `candidate_top_n=all`, `alpha=1.5`, `beta=0.1`.
- `Qwen/Qwen3-Embedding-4B`: large violation drop, but strict recall preservation failed.
- `q_target - q_trap`: large violation drop for all models, but strict recall preservation failed for all models.
