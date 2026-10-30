# LongBench-v2 suite 20261029

| condition | policy | window | questions | prompt kept | input tokens | accuracy | correct | gained vs middle | lost vs middle | p vs middle | invalid answers | retried requests | generation s/question | retrieval + packing s/question | embedding minutes | run minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| middle | truncation: middle | 65,536 | 221 | 54% | 65,528 | 40.7% | 90/221 |  |  |  | 0 | 0 | 3.5 |  |  | 20 |
| dense | dense | 65,536 | 221 | 51% | 62,213 | 45.7% | 101/221 | 30 | 19 | 0.152 | 0 | 0 | 3.3 | 0.2 | 17 | 32 |
| full | full prompt | 262,144 | 221 | 100% | 137,165 | 46.6% | 103/221 |  |  |  | 0 | 0 | 8.6 |  |  | 37 |
| position | position | 65,536 | 221 | 51% | 62,085 | 42.5% | 94/221 | 11 | 7 | 0.481 | 0 | 0 | 3.3 | 0.2 |  | 15 |
| random0 | random seed 0 | 65,536 | 221 | 51% | 61,775 | 41.2% | 91/221 | 24 | 23 | 1.000 | 0 | 0 | 3.3 | 0.2 |  | 15 |
| middle_marked | truncation: middle-marked | 65,536 | 221 | 54% | 65,528 | 40.3% | 89/221 | 4 | 5 | 1.000 | 0 | 0 | 3.5 |  |  | 20 |
| head | truncation: head | 65,536 | 221 | 54% | 65,528 | 40.3% | 89/221 | 19 | 20 | 1.000 | 0 | 0 | 3.5 |  |  | 20 |
| bm25 | bm25 | 65,536 | 221 | 51% | 62,245 | 48.9% | 108/221 | 30 | 12 | 0.008 | 0 | 0 | 3.3 | 0.2 |  | 15 |
| fusion | fusion | 65,536 | 221 | 51% | 62,357 | 48.4% | 107/221 | 27 | 10 | 0.008 | 0 | 0 | 3.3 | 0.2 | 16 | 32 |
| ladder1 | fusion, 1024-token chunks | 65,536 | 221 | 54% | 65,062 | 49.8% | 110/221 | 27 | 7 | 0.001 | 0 | 0 | 3.5 | 0.4 | 9 | 26 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | 221 | 53% | 64,385 | 46.2% | 102/221 | 25 | 13 | 0.073 | 0 | 0 | 3.4 | 0.3 | 9 | 25 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | 221 | 54% | 65,500 | 47.1% | 104/221 | 26 | 12 | 0.034 | 0 | 0 | 3.5 | 1.3 | 9 | 29 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 221 | 54% | 65,500 | 47.1% | 104/221 | 26 | 12 | 0.034 | 0 | 0 | 3.5 | 1.3 | 9 | 29 |
| middle_32k | truncation: middle | 32,768 | 221 | 27% | 32,760 | 38.9% | 86/221 |  |  |  | 0 | 0 | 1.6 |  |  | 12 |
| position_32k | position | 32,768 | 221 | 24% | 29,367 | 41.2% | 91/221 | 13 | 8 | 0.383 | 0 | 0 | 1.5 | 0.1 |  | 8 |
| middle_128k | truncation: middle | 131,072 | 221 | 89% | 114,262 | 44.8% | 99/221 |  |  |  | 0 | 0 | 6.7 |  |  | 31 |
| position_128k | position | 131,072 | 221 | 87% | 112,427 | 45.2% | 100/221 | 2 | 1 | 1.000 | 0 | 0 | 6.6 | 0.2 |  | 27 |
