# Teacher check: `pool` (policy v2)

Teacher `claude-sonnet-5-5` (effort `low`, structured outputs, Batch API; refusals retried live with server-side fallback) vs gold labels. Metrics cover rows the teacher labeled.

| Source | n | F1 | Precision | Recall | Accuracy | Cohen's κ | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| aegis2_train | 2560 | 0.690 | 0.877 | 0.569 | 0.747 | 0.493 | 0.078 |
| wildguardtrain | 2780 | 0.872 | 0.963 | 0.797 | 0.887 | 0.773 | 0.029 |
| ALL | 5340 | 0.788 | 0.926 | 0.686 | 0.820 | 0.638 | 0.052 |

**Coverage:** 5680/5964 labeled (95.2%) · served by `claude-sonnet-5-5` 5680
**Statuses:** ok 5680, refusal:cyber 242, refusal:bio 26, refusal:general_harms 11, refusal:reasoning_extraction 5
**Tokens per request:** 1,708 in / 99 out · **Actual batch cost:** $13.15 for 5964 requests
