# LongBench-v2 suite 20261128_think_pilot

| condition | policy | window | questions | prompt kept | input tokens | accuracy | correct | gained vs middle | lost vs middle | p vs middle | invalid answers | reasoning tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| middle | truncation: middle | 65,536 | 20 | 55% | 65,528 | 55.0% | 11/20 |  |  |  | 0 | 2,447 | 0 | 0 | 19.6 |  |  | 11 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 20 | 55% | 65,510 | 50.0% | 10/20 | 1 | 2 | 1.000 | 0 | 5,499 | 0 | 0 | 40.2 | 1.5 | 1 | 15 |
