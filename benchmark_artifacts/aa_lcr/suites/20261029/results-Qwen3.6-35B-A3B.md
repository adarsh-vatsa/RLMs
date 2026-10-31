# AA-LCR suite 20261029, graded by Qwen3.6-35B-A3B

| condition | policy | window | prompt kept | input tokens | documents covered | questions missing a document | accuracy | correct | gained vs middle | lost vs middle | p vs middle | output tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full prompt | 262,144 | 100% | 106,578 |  |  | 65.0% | 65/100 |  |  |  | 522 | 0 | 0 | 9.7 |  |  | 17 |
| middle | truncation: middle | 65,536 | 58% | 61,439 |  |  | 41.0% | 41/100 |  |  |  | 807 | 5 | 0 | 8.2 |  |  | 15 |
| middle_marked | truncation: middle-marked | 65,536 | 58% | 61,440 |  |  | 41.0% | 41/100 | 4 | 4 | 1.000 | 789 | 6 | 0 | 8.1 |  |  | 15 |
| head | truncation: head | 65,536 | 58% | 61,440 |  |  | 41.0% | 41/100 | 15 | 15 | 1.000 | 799 | 6 | 0 | 8.1 |  |  | 15 |
| dense | dense | 65,536 | 55% | 58,174 | 79% | 63 | 55.0% | 55/100 | 21 | 7 | 0.013 | 651 | 2 | 0 | 7.0 | 0.2 | 3 | 15 |
| position | position | 65,536 | 56% | 58,412 | 82% | 62 | 46.0% | 46/100 | 19 | 14 | 0.487 | 743 | 3 | 0 | 7.6 | 0.2 |  | 14 |
| random0 | random seed 0 | 65,536 | 55% | 57,932 | 81% | 66 | 35.0% | 35/100 | 11 | 17 | 0.345 | 1,053 | 10 | 0 | 9.5 | 0.2 |  | 17 |
| random1 | random seed 1 | 65,536 | 55% | 58,070 | 81% | 67 | 30.0% | 30/100 | 10 | 21 | 0.071 | 1,188 | 9 | 0 | 10.3 | 0.2 |  | 18 |
| random2 | random seed 2 | 65,536 | 56% | 58,324 | 84% | 67 | 33.0% | 33/100 | 10 | 18 | 0.185 | 924 | 8 | 0 | 8.7 | 0.2 |  | 15 |
| bm25 | bm25 | 65,536 | 55% | 58,271 | 76% | 64 | 54.0% | 54/100 | 24 | 11 | 0.041 | 855 | 6 | 0 | 8.3 | 0.2 |  | 15 |
| fusion | fusion | 65,536 | 55% | 58,126 | 78% | 69 | 53.0% | 53/100 | 21 | 9 | 0.043 | 548 | 1 | 0 | 6.4 | 0.2 | 3 | 14 |
| middle_32k | truncation: middle | 32,768 | 27% | 28,671 |  |  | 20.0% | 20/100 |  |  |  | 1,003 | 8 | 0 | 7.1 |  |  | 13 |
| position_32k | position | 32,768 | 25% | 25,794 | 48% | 86 | 26.0% | 26/100 | 12 | 6 | 0.238 | 747 | 5 | 0 | 5.5 | 0.1 |  | 10 |
| middle_80k | truncation: middle | 81,920 | 74% | 77,823 |  |  | 46.0% | 46/100 |  |  |  | 921 | 7 | 0 | 10.1 |  |  | 18 |
| position_80k | position | 81,920 | 71% | 74,504 | 93% | 34 | 49.0% | 49/100 | 13 | 10 | 0.678 | 699 | 1 | 0 | 8.5 | 0.3 |  | 15 |
