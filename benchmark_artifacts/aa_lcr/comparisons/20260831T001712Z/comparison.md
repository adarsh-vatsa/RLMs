# AA-LCR Four-Run Comparison

Questions: 100

| Experiment | Accuracy | Calls | Input tokens | Cache hits |
|---|---:|---:|---:|---:|
| direct_262k | 0.440 | 200 | 10704833 | 0 |
| hybrid_262k | 0.440 | 206 | 10706569 | 0 |
| direct_64k | 0.210 | 200 | 6049452 | 0 |
| hybrid_64k | 0.410 | 206 | 5666317 | 0 |

## Paired deltas

- hybrid_minus_direct_262k: 0.0
- hybrid_minus_direct_64k: 0.2
- direct_64k_minus_262k: -0.23
- hybrid_64k_minus_262k: -0.03
- direct_reasoning_loss_262k_minus_64k: 0.23
- hybrid_reasoning_loss_262k_minus_64k: 0.03
- hybrid_reasoning_retention: 0.931818

Semantic hits requiring audit: 0
