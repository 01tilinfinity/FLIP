# K-FLIP Dual-Stream Experiment

## Environment

```bash
conda env create -f environment.yml
conda activate flip
cp .env.example .env
```

Fill `.env` with local credentials. `.env` and related secret files are ignored
by git.

## Step 1: NevIR mini sample

This repo starts with a small, deterministic sample from
`orionweller/NevIR` for cheap debugging.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_nevir_sample.py --split train --sample-size 50 --seed 42 --query-column q1
```

Outputs:

- `data/nevir_mini_train_q1_50_seed42.csv`
- `data/nevir_mini_train_q1_50_seed42.jsonl`

For the planned K-FLIP setup, use `q1` so `doc1` is the answer document and
`doc2` is the trap document. If `q2` is selected later, the script keeps the raw
`doc1` and `doc2` columns and flips the explicit `answer_doc` / `trap_doc`
labels.

## HotpotQA sample

HotpotQA does not ship as explicit `doc1` / `doc2` contrastive pairs. The loader
maps supporting-fact paragraphs to `doc1` and the highest-overlap non-supporting
paragraph from the same example to `doc2`.

```bash
python scripts/setup_hotpotqa_sample.py \
  --config distractor \
  --split train \
  --sample-size 1000 \
  --seed 42
```

## Steps 2-5: Local retrieval, decomposition, reranking, evaluation

```bash
conda activate flip
python scripts/run_kflip_experiment.py \
  --sample-csv data/nevir_mini_train_q1_50_seed42.csv \
  --retriever both \
  --decomposition-mode openai \
  --allow-heuristic-fallback \
  --openai-model gpt-5.4-mini
```

Candidate-guarded reranking can reuse an existing decomposition cache and avoid
another OpenAI call:

```bash
python scripts/run_kflip_experiment.py \
  --sample-csv data/nevir_mini_train_q1_50_seed42.csv \
  --retriever both \
  --decompositions-jsonl outputs/openai_full_v3/query_decompositions.jsonl \
  --candidate-top-n 10 \
  --output-dir outputs/openai_full_v3_top10
```

Score-level Anti-RRF sweeps normalize baseline/target/trap scores per query and
rerank with `alpha * baseline + target - beta * trap`:

```bash
python scripts/sweep_score_anti_rrf.py \
  --decompositions-jsonl outputs/openai_full_v3/query_decompositions.jsonl \
  --candidate-top-ns 5,10,20,all \
  --alphas 0,0.25,0.5,0.75,1,1.5,2 \
  --betas 0.1,0.2,0.3,0.5,0.75,1 \
  --output-dir results/score_anti_rrf
```

For larger samples, use the summary-only fast sweep:

```bash
python scripts/fast_score_anti_rrf_sweep.py \
  --sample-csv data/hotpotqa_distractor_train_1000_seed42.csv \
  --decompositions-jsonl outputs/hotpotqa_1000_bm25_heuristic/query_decompositions.jsonl \
  --retriever both \
  --output-dir results/hotpotqa_1000_score_anti_rrf
```

Softer variants add hinge penalties and confidence gates:

```bash
python scripts/sweep_improved_anti_rrf.py \
  --decompositions-jsonl outputs/openai_full_v3/query_decompositions.jsonl \
  --candidate-top-ns 5,10,20,all \
  --output-dir results/improved_anti_rrf
```

## ExcluIR direct rewrite-to-score experiments

The current ExcluIR experiment removes oracle target/trap queries from scoring.
It first rewrites each original query into inferred `q_target` and `q_trap`
with GPT-4o mini, then scores the full ExcluIR corpus directly with dense
embeddings.

Canonical prompt:

```text
prompts/excluir_rewriter_gpt4o_mini_system.txt
```

Generate or resume rewrites:

```bash
python scripts/generate_excluir_rewrites.py \
  --sample-csv data/excluir_manual_1000_seed42.csv \
  --output-jsonl outputs/excluir_rewriter_gpt4o_mini/decompositions.jsonl \
  --model gpt-4o-mini \
  --workers 8
```

Run the latest direct scoring sweep:

```bash
scripts/run_excluir_rewriter_direct_score_experiment.sh all
```

Available presets:

```text
all
bge-m3
qwen3-0.6b
qwen3-4b
openai-small
```

The runner evaluates:

```text
baseline:           score(d) = sim(RQ_rewrite, d)
baseline_minus_trap: final(d) = alpha * sim(RQ_rewrite, d) - beta * sim(q_trap, d)
target_minus_trap:   final(d) = gamma * sim(q_target, d) - beta * sim(q_trap, d)
```

Summarize embedding-model comparisons:

```bash
python scripts/summarize_excluir_embedding_model_comparison.py
```

Latest tracked result summaries:

- `results/excluir_embedding_model_comparison_rewriter_gpt4o_mini/embedding_model_comparison_summary.md`
- `results/excluir_embedding_model_comparison_rewriter_gpt4o_mini/openai_qwen3_4b_baseline_vs_antirrf_recall_violation.csv`
- `results/excluir_embedding_model_comparison_rewriter_gpt4o_mini/direct_rewrite_to_score_summary.md`

Large local artifacts are intentionally not tracked:

- `data/excluir_raw/`
- `data/excluir_cache/`
- `results/score_matrices/`
- `logs/`

Outputs:

- `outputs/query_decompositions.jsonl`
- `outputs/ranking_details.csv`
- `outputs/scoreboard.csv`
- `outputs/scoreboard.json`

If OpenAI credentials are not available yet, run a structural smoke test with
the NevIR paired-query fallback:

```bash
python scripts/run_kflip_experiment.py \
  --sample-csv data/nevir_mini_train_q1_50_seed42.csv \
  --retriever bm25 \
  --decomposition-mode heuristic \
  --max-samples 5 \
  --output-dir outputs/smoke
```
