# MRCR v2 suite 20261128_think_pilot

| condition | policy | window | questions | prompt kept | input tokens | exact match | MRCR score | correct | gained vs middle | lost vs middle | p vs middle | exact match at 133K | exact match at 267K | exact match at 533K | exact match at 1.06M | prefix compliance | output tokens | reasoning tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| middle_128k | truncation: middle | 131,072 | 12/12 | 45% | 126,976 | 25.0% | 0.346 | 3/12 |  |  |  | 33% | 33% | 0% | 33% | 100% | 4,118 | 3,762 | 0 | 0 | 38.2 |  |  | 8 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | 12/12 | 45% | 126,952 | 66.7% | 0.676 | 8/12 | 5 | 0 | 0.062 | 100% | 100% | 0% | 67% | 100% | 3,653 | 3,262 | 0 | 0 | 34.7 | 3.2 | 1 | 9 |
