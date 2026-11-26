# AA-LCR suite 20261128_think_pilot, graded by gpt-5.6-luna

| condition | policy | window | prompt kept | input tokens | documents covered | questions missing a document | accuracy | correct | gained vs middle | lost vs middle | p vs middle | agreement with Qwen3.6-35B-A3B | output tokens | reasoning tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full prompt | 262,144 | 100% | 100,788 |  |  | 70.0% | 14/20 |  |  |  | 100% | 2,440 | 2,337 | 0 | 0 | 23.0 |  |  | 8 |
| middle | truncation: middle | 65,536 | 62% | 61,439 |  |  | 40.0% | 8/20 |  |  |  | 95% | 2,862 | 2,746 | 0 | 0 | 21.8 |  |  | 7 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 62% | 61,410 | 100% | 0 | 55.0% | 11/20 | 5 | 2 | 0.453 | 90% | 2,815 | 2,692 | 0 | 0 | 21.5 | 0.6 | 0 | 8 |
