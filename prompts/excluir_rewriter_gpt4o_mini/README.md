# ExcluIR Rewriter Prompt Versions

Policy: keep every prompt version as an immutable experiment artifact.

- Do not overwrite an existing `v*_system.txt` file after it has been used.
- Create a new file for each prompt change, for example `v2_short_trap_system.txt`.
- Use a matching output directory, for example:
  `outputs/excluir_rewriter_gpt4o_mini_v2_short_trap/decompositions.jsonl`.
- Each generated decomposition row records `system_prompt_path` and
  `system_prompt_sha256`, so score results can be traced back to the exact
  prompt content.
- If an output JSONL already contains rows from another recorded prompt hash,
  `scripts/generate_excluir_rewrites.py` will refuse to append unless
  `--allow-mixed-prompt-output` is passed.

Current versions:

| Version | File | Notes |
| --- | --- | --- |
| v1 | `v1_base_system.txt` | Baseline target/trap decomposition prompt. |
| v2 | `v2_recall_preserving_system.txt` | Recall-preserving target/trap prompt with conservative trap precision. |
