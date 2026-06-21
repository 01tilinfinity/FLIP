| embedding_model | method | gamma/alpha | beta | candidate_top_n | Recall@3 | Violation@3 | Gap@3 | Recall@5 | Violation@5 | Gap@5 | Avg Gap | Right Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAAI/bge-m3 | baseline |  |  |  | 0.922 | 0.819 | 0.103 | 0.940 | 0.876 | 0.064 | 0.066 | 0.670 |
| BAAI/bge-m3 | baseline_minus_trap | 1.0 | 0.75 | all | 0.685 | 0.099 | 0.586 | 0.722 | 0.115 | 0.607 | 0.608 | 0.874 |
| BAAI/bge-m3 | target_minus_trap | 0.25 | 0.1 | all | 0.864 | 0.087 | 0.777 | 0.890 | 0.102 | 0.788 | 0.785 | 0.942 |
| Qwen/Qwen3-Embedding-0.6B | baseline |  |  |  | 0.931 | 0.666 | 0.265 | 0.942 | 0.740 | 0.202 | 0.205 | 0.728 |
| Qwen/Qwen3-Embedding-0.6B | baseline_minus_trap | 1.0 | 0.75 | all | 0.806 | 0.146 | 0.660 | 0.827 | 0.168 | 0.659 | 0.660 | 0.890 |
| Qwen/Qwen3-Embedding-0.6B | target_minus_trap | 0.25 | 0.1 | all | 0.898 | 0.084 | 0.814 | 0.914 | 0.097 | 0.817 | 0.813 | 0.946 |
| Qwen/Qwen3-Embedding-4B | baseline |  |  |  | 0.934 | 0.692 | 0.242 | 0.942 | 0.746 | 0.196 | 0.194 | 0.740 |
| Qwen/Qwen3-Embedding-4B | baseline_minus_trap | 1.0 | 0.75 | all | 0.839 | 0.174 | 0.665 | 0.858 | 0.202 | 0.656 | 0.661 | 0.888 |
| Qwen/Qwen3-Embedding-4B | target_minus_trap | 1.0 | 0.5 | all | 0.893 | 0.077 | 0.816 | 0.910 | 0.090 | 0.820 | 0.818 | 0.948 |
| text-embedding-3-small | baseline |  |  |  | 0.931 | 0.835 | 0.096 | 0.946 | 0.883 | 0.063 | 0.064 | 0.666 |
| text-embedding-3-small | baseline_minus_trap | 1.0 | 0.75 | all | 0.809 | 0.187 | 0.622 | 0.835 | 0.208 | 0.627 | 0.623 | 0.863 |
| text-embedding-3-small | target_minus_trap | 1.0 | 0.5 | all | 0.860 | 0.077 | 0.783 | 0.884 | 0.092 | 0.792 | 0.793 | 0.947 |
