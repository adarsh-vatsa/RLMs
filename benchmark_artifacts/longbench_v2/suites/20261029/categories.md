# LongBench-v2 suite 20261029: accuracy by category

| condition | policy | window | category | questions | accuracy | correct |
|---|---|---|---|---|---|---|
| middle | truncation: middle | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| middle | truncation: middle | 65,536 | Long In-context Learning | 44 | 50.0% | 22 |
| middle | truncation: middle | 65,536 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| middle | truncation: middle | 65,536 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| middle | truncation: middle | 65,536 | Multi-Document QA | 46 | 52.2% | 24 |
| middle | truncation: middle | 65,536 | Single-Document QA | 82 | 30.5% | 25 |
| dense | dense | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| dense | dense | 65,536 | Long In-context Learning | 44 | 56.8% | 25 |
| dense | dense | 65,536 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| dense | dense | 65,536 | Long-dialogue History Understanding | 20 | 35.0% | 7 |
| dense | dense | 65,536 | Multi-Document QA | 46 | 56.5% | 26 |
| dense | dense | 65,536 | Single-Document QA | 82 | 39.0% | 32 |
| full | full prompt | 262,144 | Code Repository Understanding | 12 | 41.7% | 5 |
| full | full prompt | 262,144 | Long In-context Learning | 44 | 61.4% | 27 |
| full | full prompt | 262,144 | Long Structured Data Understanding | 17 | 52.9% | 9 |
| full | full prompt | 262,144 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| full | full prompt | 262,144 | Multi-Document QA | 46 | 50.0% | 23 |
| full | full prompt | 262,144 | Single-Document QA | 82 | 37.8% | 31 |
| position | position | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| position | position | 65,536 | Long In-context Learning | 44 | 50.0% | 22 |
| position | position | 65,536 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| position | position | 65,536 | Long-dialogue History Understanding | 20 | 50.0% | 10 |
| position | position | 65,536 | Multi-Document QA | 46 | 45.7% | 21 |
| position | position | 65,536 | Single-Document QA | 82 | 36.6% | 30 |
| random0 | random seed 0 | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| random0 | random seed 0 | 65,536 | Long In-context Learning | 44 | 45.5% | 20 |
| random0 | random seed 0 | 65,536 | Long Structured Data Understanding | 17 | 29.4% | 5 |
| random0 | random seed 0 | 65,536 | Long-dialogue History Understanding | 20 | 50.0% | 10 |
| random0 | random seed 0 | 65,536 | Multi-Document QA | 46 | 52.2% | 24 |
| random0 | random seed 0 | 65,536 | Single-Document QA | 82 | 32.9% | 27 |
| middle_marked | truncation: middle-marked | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| middle_marked | truncation: middle-marked | 65,536 | Long In-context Learning | 44 | 45.5% | 20 |
| middle_marked | truncation: middle-marked | 65,536 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| middle_marked | truncation: middle-marked | 65,536 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| middle_marked | truncation: middle-marked | 65,536 | Multi-Document QA | 46 | 50.0% | 23 |
| middle_marked | truncation: middle-marked | 65,536 | Single-Document QA | 82 | 32.9% | 27 |
| head | truncation: head | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| head | truncation: head | 65,536 | Long In-context Learning | 44 | 54.5% | 24 |
| head | truncation: head | 65,536 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| head | truncation: head | 65,536 | Long-dialogue History Understanding | 20 | 20.0% | 4 |
| head | truncation: head | 65,536 | Multi-Document QA | 46 | 52.2% | 24 |
| head | truncation: head | 65,536 | Single-Document QA | 82 | 31.7% | 26 |
| bm25 | bm25 | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25 | bm25 | 65,536 | Long In-context Learning | 44 | 59.1% | 26 |
| bm25 | bm25 | 65,536 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| bm25 | bm25 | 65,536 | Long-dialogue History Understanding | 20 | 55.0% | 11 |
| bm25 | bm25 | 65,536 | Multi-Document QA | 46 | 54.3% | 25 |
| bm25 | bm25 | 65,536 | Single-Document QA | 82 | 41.5% | 34 |
| fusion | fusion | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| fusion | fusion | 65,536 | Long In-context Learning | 44 | 63.6% | 28 |
| fusion | fusion | 65,536 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| fusion | fusion | 65,536 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| fusion | fusion | 65,536 | Multi-Document QA | 46 | 56.5% | 26 |
| fusion | fusion | 65,536 | Single-Document QA | 82 | 40.2% | 33 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Long In-context Learning | 44 | 61.4% | 27 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Long Structured Data Understanding | 17 | 52.9% | 9 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Multi-Document QA | 46 | 56.5% | 26 |
| ladder1 | fusion, 1024-token chunks | 65,536 | Single-Document QA | 82 | 41.5% | 34 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Long In-context Learning | 44 | 63.6% | 28 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Multi-Document QA | 46 | 47.8% | 22 |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 65,536 | Single-Document QA | 82 | 36.6% | 30 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Long In-context Learning | 44 | 65.9% | 29 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Multi-Document QA | 46 | 45.7% | 21 |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 65,536 | Single-Document QA | 82 | 40.2% | 33 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long In-context Learning | 44 | 65.9% | 29 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Multi-Document QA | 46 | 45.7% | 21 |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Single-Document QA | 82 | 40.2% | 33 |
| middle_32k | truncation: middle | 32,768 | Code Repository Understanding | 12 | 50.0% | 6 |
| middle_32k | truncation: middle | 32,768 | Long In-context Learning | 44 | 45.5% | 20 |
| middle_32k | truncation: middle | 32,768 | Long Structured Data Understanding | 17 | 17.6% | 3 |
| middle_32k | truncation: middle | 32,768 | Long-dialogue History Understanding | 20 | 30.0% | 6 |
| middle_32k | truncation: middle | 32,768 | Multi-Document QA | 46 | 52.2% | 24 |
| middle_32k | truncation: middle | 32,768 | Single-Document QA | 82 | 32.9% | 27 |
| position_32k | position | 32,768 | Code Repository Understanding | 12 | 50.0% | 6 |
| position_32k | position | 32,768 | Long In-context Learning | 44 | 47.7% | 21 |
| position_32k | position | 32,768 | Long Structured Data Understanding | 17 | 17.6% | 3 |
| position_32k | position | 32,768 | Long-dialogue History Understanding | 20 | 30.0% | 6 |
| position_32k | position | 32,768 | Multi-Document QA | 46 | 54.3% | 25 |
| position_32k | position | 32,768 | Single-Document QA | 82 | 36.6% | 30 |
| middle_128k | truncation: middle | 131,072 | Code Repository Understanding | 12 | 41.7% | 5 |
| middle_128k | truncation: middle | 131,072 | Long In-context Learning | 44 | 59.1% | 26 |
| middle_128k | truncation: middle | 131,072 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| middle_128k | truncation: middle | 131,072 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| middle_128k | truncation: middle | 131,072 | Multi-Document QA | 46 | 45.7% | 21 |
| middle_128k | truncation: middle | 131,072 | Single-Document QA | 82 | 37.8% | 31 |
| position_128k | position | 131,072 | Code Repository Understanding | 12 | 41.7% | 5 |
| position_128k | position | 131,072 | Long In-context Learning | 44 | 59.1% | 26 |
| position_128k | position | 131,072 | Long Structured Data Understanding | 17 | 52.9% | 9 |
| position_128k | position | 131,072 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| position_128k | position | 131,072 | Multi-Document QA | 46 | 47.8% | 22 |
| position_128k | position | 131,072 | Single-Document QA | 82 | 36.6% | 30 |
| bm25_32k | bm25 | 32,768 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25_32k | bm25 | 32,768 | Long In-context Learning | 44 | 54.5% | 24 |
| bm25_32k | bm25 | 32,768 | Long Structured Data Understanding | 17 | 35.3% | 6 |
| bm25_32k | bm25 | 32,768 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| bm25_32k | bm25 | 32,768 | Multi-Document QA | 46 | 56.5% | 26 |
| bm25_32k | bm25 | 32,768 | Single-Document QA | 82 | 39.0% | 32 |
| bm25_128k | bm25 | 131,072 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25_128k | bm25 | 131,072 | Long In-context Learning | 44 | 63.6% | 28 |
| bm25_128k | bm25 | 131,072 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| bm25_128k | bm25 | 131,072 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| bm25_128k | bm25 | 131,072 | Multi-Document QA | 46 | 45.7% | 21 |
| bm25_128k | bm25 | 131,072 | Single-Document QA | 82 | 39.0% | 32 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long In-context Learning | 44 | 65.9% | 29 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Multi-Document QA | 46 | 45.7% | 21 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Single-Document QA | 82 | 40.2% | 33 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Code Repository Understanding | 12 | 41.7% | 5 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long In-context Learning | 44 | 54.5% | 24 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Multi-Document QA | 46 | 56.5% | 26 |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Single-Document QA | 82 | 37.8% | 31 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Code Repository Understanding | 12 | 41.7% | 5 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long In-context Learning | 44 | 59.1% | 26 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Multi-Document QA | 46 | 50.0% | 23 |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Single-Document QA | 82 | 41.5% | 34 |
| random1 | random seed 1 | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| random1 | random seed 1 | 65,536 | Long In-context Learning | 44 | 52.3% | 23 |
| random1 | random seed 1 | 65,536 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| random1 | random seed 1 | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| random1 | random seed 1 | 65,536 | Multi-Document QA | 46 | 41.3% | 19 |
| random1 | random seed 1 | 65,536 | Single-Document QA | 82 | 29.3% | 24 |
| random2 | random seed 2 | 65,536 | Code Repository Understanding | 12 | 33.3% | 4 |
| random2 | random seed 2 | 65,536 | Long In-context Learning | 44 | 59.1% | 26 |
| random2 | random seed 2 | 65,536 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| random2 | random seed 2 | 65,536 | Long-dialogue History Understanding | 20 | 50.0% | 10 |
| random2 | random seed 2 | 65,536 | Multi-Document QA | 46 | 50.0% | 23 |
| random2 | random seed 2 | 65,536 | Single-Document QA | 82 | 39.0% | 32 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long In-context Learning | 44 | 63.6% | 28 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long Structured Data Understanding | 17 | 58.8% | 10 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Multi-Document QA | 46 | 50.0% | 23 |
| bm25_recipe | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Single-Document QA | 82 | 42.7% | 35 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long In-context Learning | 44 | 61.4% | 27 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long Structured Data Understanding | 17 | 41.2% | 7 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Long-dialogue History Understanding | 20 | 45.0% | 9 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Multi-Document QA | 46 | 58.7% | 27 |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | Single-Document QA | 82 | 37.8% | 31 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Code Repository Understanding | 12 | 41.7% | 5 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long In-context Learning | 44 | 63.6% | 28 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long Structured Data Understanding | 17 | 47.1% | 8 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Long-dialogue History Understanding | 20 | 40.0% | 8 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Multi-Document QA | 46 | 47.8% | 22 |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | Single-Document QA | 82 | 41.5% | 34 |
