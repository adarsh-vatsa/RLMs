# AA-LCR suite 20261029, graded by gpt-5.6-luna

| condition | policy | window | prompt kept | input tokens | documents covered | questions missing a document | accuracy | correct | gained vs middle | lost vs middle | p vs middle | agreement with Qwen3.6-35B-A3B | output tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full prompt | 262,144 | 100% | 106,578 |  |  | 65.0% | 65/100 |  |  |  | 100% | 522 | 0 | 0 | 9.7 |  |  | 17 |
| middle | truncation: middle | 65,536 | 58% | 61,439 |  |  | 40.0% | 40/100 |  |  |  | 97% | 807 | 5 | 0 | 8.2 |  |  | 15 |
| middle_marked | truncation: middle-marked | 65,536 | 58% | 61,440 |  |  | 40.0% | 40/100 | 3 | 3 | 1.000 | 99% | 789 | 6 | 0 | 8.1 |  |  | 15 |
| head | truncation: head | 65,536 | 58% | 61,440 |  |  | 42.0% | 42/100 | 16 | 14 | 0.856 | 99% | 799 | 6 | 0 | 8.1 |  |  | 15 |
| dense | dense | 65,536 | 55% | 58,174 | 79% | 63 | 56.0% | 56/100 | 22 | 6 | 0.004 | 99% | 651 | 2 | 0 | 7.0 | 0.2 | 3 | 15 |
| position | position | 65,536 | 56% | 58,412 | 82% | 62 | 46.0% | 46/100 | 18 | 12 | 0.362 | 100% | 743 | 3 | 0 | 7.6 | 0.2 |  | 14 |
| random0 | random seed 0 | 65,536 | 55% | 57,932 | 81% | 66 | 35.0% | 35/100 | 12 | 17 | 0.458 | 100% | 1,053 | 10 | 0 | 9.5 | 0.2 |  | 17 |
| random1 | random seed 1 | 65,536 | 55% | 58,070 | 81% | 67 | 33.0% | 33/100 | 12 | 19 | 0.281 | 97% | 1,188 | 9 | 0 | 10.3 | 0.2 |  | 18 |
| random2 | random seed 2 | 65,536 | 56% | 58,324 | 84% | 67 | 32.0% | 32/100 | 8 | 16 | 0.152 | 99% | 924 | 8 | 0 | 8.7 | 0.2 |  | 15 |
| bm25 | bm25 | 65,536 | 55% | 58,271 | 76% | 64 | 54.0% | 54/100 | 24 | 10 | 0.024 | 100% | 855 | 6 | 0 | 8.3 | 0.2 |  | 15 |
| fusion | fusion | 65,536 | 55% | 58,126 | 78% | 69 | 53.0% | 53/100 | 21 | 8 | 0.024 | 100% | 548 | 1 | 0 | 6.4 | 0.2 | 3 | 14 |
| middle_32k | truncation: middle | 32,768 | 27% | 28,671 |  |  | 21.0% | 21/100 |  |  |  | 99% | 1,003 | 8 | 0 | 7.1 |  |  | 13 |
| position_32k | position | 32,768 | 25% | 25,794 | 48% | 86 | 27.0% | 27/100 | 12 | 6 | 0.238 | 99% | 747 | 5 | 0 | 5.5 | 0.1 |  | 10 |
| middle_80k | truncation: middle | 81,920 | 74% | 77,823 |  |  | 45.0% | 45/100 |  |  |  | 99% | 921 | 7 | 0 | 10.1 |  |  | 18 |
| position_80k | position | 81,920 | 71% | 74,504 | 93% | 34 | 51.0% | 51/100 | 16 | 10 | 0.327 | 98% | 699 | 1 | 0 | 8.5 | 0.3 |  | 15 |
| bm25_32k | bm25 | 32,768 | 23% | 24,246 | 47% | 85 | 36.0% | 36/100 | 25 | 10 | 0.017 | 100% | 903 | 10 | 0 | 6.3 | 0.1 |  | 11 |
| bm25_80k | bm25 | 81,920 | 71% | 74,774 | 84% | 50 | 52.0% | 52/100 | 18 | 11 | 0.265 | 99% | 702 | 3 | 0 | 8.5 | 0.3 |  | 15 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 58% | 61,406 | 100% | 0 | 58.0% | 58/100 | 23 | 5 | 0.001 | 99% | 584 | 3 | 0 | 6.8 | 0.7 | 1 | 14 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | 27% | 28,628 | 98% | 7 | 56.0% | 56/100 | 40 | 5 | 0.000 | 98% | 810 | 7 | 0 | 6.0 | 0.8 | 1 | 13 |
| upgraded_80k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 81,920 | 74% | 77,779 | 100% | 0 | 64.0% | 64/100 | 23 | 4 | 0.000 | 98% | 525 | 3 | 0 | 7.6 | 0.8 | 1 | 16 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | 58% | 61,404 | 95% | 22 | 63.0% | 63/100 | 27 | 4 | 0.000 | 97% | 526 | 2 | 0 | 6.5 | 0.8 | 1 | 14 |
| ladder3_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 32,768 | 27% | 28,635 | 78% | 66 | 56.0% | 56/100 | 38 | 3 | 0.000 | 97% | 778 | 7 | 0 | 5.8 | 0.8 | 1 | 13 |
| ladder3_80k | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 81,920 | 74% | 77,779 | 97% | 18 | 62.0% | 62/100 | 22 | 5 | 0.002 | 100% | 525 | 1 | 0 | 7.6 | 0.9 | 1 | 16 |
