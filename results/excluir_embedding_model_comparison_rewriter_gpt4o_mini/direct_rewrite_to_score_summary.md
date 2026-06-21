# ExcluIR Direct Rewrite-to-Score Experiment Summary

## Setup

- Dataset: ExcluIR 1,000-query sample
- Corpus: full ExcluIR corpus, 90,406 documents
- Rewriter output: `outputs/excluir_rewriter_gpt4o_mini/decompositions.jsonl`
- Rewriter prompt: `prompts/excluir_rewriter_gpt4o_mini_system.txt`
- Rewriter script: `scripts/generate_excluir_rewrites.py`
- Direct scoring runner: `scripts/run_excluir_rewriter_direct_score_experiment.sh`
- Scoring path: direct `rewrite -> dense score -> anti-RRF/reranking`
- G-Eval: not used for scoring in this run

## Methods

- Baseline: `score(d) = sim(RQ_rewrite, d)`
- Baseline-minus-trap: `final_score(d) = alpha * sim(RQ_rewrite, d) - beta * sim(q_trap_hat, d)`
- Target-minus-trap: `final_score(d) = gamma * sim(q_target_hat, d) - beta * sim(q_trap_hat, d)`

Gold positive/trap indices were used only for evaluation, not for scoring.

## Best Results

| embedding_model | method | gamma/alpha | beta | candidate_top_n | Recall@3 | Violation@3 | Gap@3 | Recall@5 | Violation@5 | Gap@5 | Avg Gap | Right Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-Embedding-0.6B | baseline |  |  |  | 0.931 | 0.666 | 0.265 | 0.942 | 0.740 | 0.202 | 0.205 | 0.728 |
| Qwen/Qwen3-Embedding-0.6B | baseline_minus_trap | 1.0 | 0.75 | all | 0.806 | 0.146 | 0.660 | 0.827 | 0.168 | 0.659 | 0.660 | 0.890 |
| Qwen/Qwen3-Embedding-0.6B | target_minus_trap | 0.25 | 0.1 | all | 0.898 | 0.084 | 0.814 | 0.914 | 0.097 | 0.817 | 0.813 | 0.946 |
| Qwen/Qwen3-Embedding-4B | baseline |  |  |  | 0.934 | 0.692 | 0.242 | 0.942 | 0.746 | 0.196 | 0.194 | 0.740 |
| Qwen/Qwen3-Embedding-4B | baseline_minus_trap | 1.0 | 0.75 | all | 0.839 | 0.174 | 0.665 | 0.858 | 0.202 | 0.656 | 0.661 | 0.888 |
| Qwen/Qwen3-Embedding-4B | target_minus_trap | 1.0 | 0.5 | all | 0.893 | 0.077 | 0.816 | 0.910 | 0.090 | 0.820 | 0.818 | 0.948 |

## Takeaway

Qwen3-Embedding-4B slightly improves the best target-minus-trap exclusion metrics over 0.6B, especially Violation@3 and Avg Gap, but it does not improve Recall@3. The 0.6B model keeps the best target-minus-trap Recall@3 among the two Qwen models.
