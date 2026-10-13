# Teacher check: `pilot` (policy v2)

Teacher `claude-sonnet-5-5` (effort `low`, structured outputs, Batch API; refusals retried live with server-side fallback) vs gold labels. Metrics cover rows the teacher labeled.

| Source | n | F1 | Precision | Recall | Accuracy | Cohen's κ | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| harmbench_val | 127 | 0.913 | 0.921 | 0.906 | 0.913 | 0.827 | 0.079 |
| wildguardtest | 145 | 0.797 | 0.979 | 0.671 | 0.834 | 0.665 | 0.013 |
| ALL | 272 | 0.857 | 0.946 | 0.784 | 0.871 | 0.742 | 0.043 |

**Coverage:** 272/300 labeled (90.7%) · served by `claude-sonnet-5-5` 245, `claude-sonnet-5` 27
**Statuses:** ok 272, refusal:bio 16, refusal:cyber 11, refusal:general_harms 1
**Tokens per request:** 1,972 in / 96 out · **Actual batch cost:** $0.86 for 300 requests
