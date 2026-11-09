# AA-LCR suite 20261029, graded by Qwen3.6-35B-A3B

| condition | policy | window | prompt kept | input tokens | documents covered | questions missing a document | accuracy | correct | gained vs middle | lost vs middle | p vs middle | agreement with gpt-5.6-luna | output tokens | output limit hit | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | full prompt | 262,144 | 100% | 106,578 |  |  | 65.0% | 65/100 |  |  |  | 100% | 522 | 0 | 0 | 9.7 |  |  | 17 |
| middle | truncation: middle | 65,536 | 58% | 61,439 |  |  | 41.0% | 41/100 |  |  |  | 97% | 807 | 5 | 0 | 8.2 |  |  | 15 |
| middle_marked | truncation: middle-marked | 65,536 | 58% | 61,440 |  |  | 41.0% | 41/100 | 4 | 4 | 1.000 | 99% | 789 | 6 | 0 | 8.1 |  |  | 15 |
| head | truncation: head | 65,536 | 58% | 61,440 |  |  | 41.0% | 41/100 | 15 | 15 | 1.000 | 99% | 799 | 6 | 0 | 8.1 |  |  | 15 |
| dense | dense | 65,536 | 55% | 58,174 | 79% | 63 | 55.0% | 55/100 | 21 | 7 | 0.013 | 99% | 651 | 2 | 0 | 7.0 | 0.2 | 3 | 15 |
| position | position | 65,536 | 56% | 58,412 | 82% | 62 | 46.0% | 46/100 | 19 | 14 | 0.487 | 100% | 743 | 3 | 0 | 7.6 | 0.2 |  | 14 |
| random0 | random seed 0 | 65,536 | 55% | 57,932 | 81% | 66 | 35.0% | 35/100 | 11 | 17 | 0.345 | 100% | 1,053 | 10 | 0 | 9.5 | 0.2 |  | 17 |
| random1 | random seed 1 | 65,536 | 55% | 58,070 | 81% | 67 | 30.0% | 30/100 | 10 | 21 | 0.071 | 97% | 1,188 | 9 | 0 | 10.3 | 0.2 |  | 18 |
| random2 | random seed 2 | 65,536 | 56% | 58,324 | 84% | 67 | 33.0% | 33/100 | 10 | 18 | 0.185 | 99% | 924 | 8 | 0 | 8.7 | 0.2 |  | 15 |
| bm25 | bm25 | 65,536 | 55% | 58,271 | 76% | 64 | 54.0% | 54/100 | 24 | 11 | 0.041 | 100% | 855 | 6 | 0 | 8.3 | 0.2 |  | 15 |
| fusion | fusion | 65,536 | 55% | 58,126 | 78% | 69 | 53.0% | 53/100 | 21 | 9 | 0.043 | 100% | 548 | 1 | 0 | 6.4 | 0.2 | 3 | 14 |
| middle_32k | truncation: middle | 32,768 | 27% | 28,671 |  |  | 20.0% | 20/100 |  |  |  | 99% | 1,003 | 8 | 0 | 7.1 |  |  | 13 |
| position_32k | position | 32,768 | 25% | 25,794 | 48% | 86 | 26.0% | 26/100 | 12 | 6 | 0.238 | 99% | 747 | 5 | 0 | 5.5 | 0.1 |  | 10 |
| middle_80k | truncation: middle | 81,920 | 74% | 77,823 |  |  | 46.0% | 46/100 |  |  |  | 99% | 921 | 7 | 0 | 10.1 |  |  | 18 |
| position_80k | position | 81,920 | 71% | 74,504 | 93% | 34 | 49.0% | 49/100 | 13 | 10 | 0.678 | 98% | 699 | 1 | 0 | 8.5 | 0.3 |  | 15 |
| bm25_32k | bm25 | 32,768 | 23% | 24,246 | 47% | 85 | 36.0% | 36/100 | 26 | 10 | 0.011 | 100% | 903 | 10 | 0 | 6.3 | 0.1 |  | 11 |
| bm25_80k | bm25 | 81,920 | 71% | 74,774 | 84% | 50 | 51.0% | 51/100 | 16 | 11 | 0.442 | 99% | 702 | 3 | 0 | 8.5 | 0.3 |  | 15 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 58% | 61,406 | 100% | 0 | 59.0% | 59/100 | 25 | 7 | 0.002 | 99% | 584 | 3 | 0 | 6.8 | 0.7 | 1 | 14 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | 27% | 28,628 | 98% | 7 | 54.0% | 54/100 | 40 | 6 | 0.000 | 98% | 810 | 7 | 0 | 6.0 | 0.8 | 1 | 13 |
| upgraded_80k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 81,920 | 74% | 77,779 | 100% | 0 | 62.0% | 62/100 | 21 | 5 | 0.002 | 98% | 525 | 3 | 0 | 7.6 | 0.8 | 1 | 16 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | 58% | 61,404 | 95% | 22 | 60.0% | 60/100 | 25 | 6 | 0.001 | 97% | 526 | 2 | 0 | 6.5 | 0.8 | 1 | 14 |
| ladder3_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 32,768 | 27% | 28,635 | 78% | 66 | 53.0% | 53/100 | 37 | 4 | 0.000 | 97% | 778 | 7 | 0 | 5.8 | 0.8 | 1 | 13 |
| ladder3_80k | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 81,920 | 74% | 77,779 | 97% | 18 | 62.0% | 62/100 | 21 | 5 | 0.002 | 100% | 525 | 1 | 0 | 7.6 | 0.9 | 1 | 16 |
