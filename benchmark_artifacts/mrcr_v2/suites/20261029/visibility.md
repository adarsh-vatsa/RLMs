# MRCR v2 suite 20261029: visibility

| condition | policy | window | questions | exact match | target visible | all needles visible | exact match given target visible | exact match given target hidden | exact match given all needles visible | target visible at 133K | target visible at 267K | target visible at 533K | target visible at 1.06M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| middle | truncation: middle | 262,144 | 120 | 38.3% | 62% | 50% | 61.3% | 0.0% | 58.3% | 100% | 100% | 33% | 17% |
| dense | dense | 262,144 | 120 | 55.0% | 81% | 58% | 68.0% | 0.0% | 62.9% | 100% | 100% | 80% | 43% |
| position | position | 262,144 | 120 | 38.3% | 62% | 49% | 62.2% | 0.0% | 59.3% | 100% | 97% | 33% | 17% |
| random0 | random seed 0 | 262,144 | 120 | 41.7% | 63% | 48% | 65.8% | 0.0% | 59.6% | 100% | 93% | 50% | 10% |
| middle_marked | truncation: middle-marked | 262,144 | 120 | 38.3% | 62% | 50% | 61.3% | 0.0% | 58.3% | 100% | 100% | 33% | 17% |
| head | truncation: head | 262,144 | 120 | 38.3% | 68% | 52% | 56.1% | 0.0% | 53.2% | 100% | 100% | 53% | 20% |
| bm25 | bm25 | 262,144 | 120 | 60.0% | 93% | 84% | 64.3% | 0.0% | 62.4% | 100% | 100% | 93% | 80% |
| fusion | fusion | 262,144 | 120 | 55.8% | 89% | 72% | 62.6% | 0.0% | 59.3% | 100% | 100% | 93% | 63% |
| dense_7500 | dense, 7500-token chunks | 262,144 | 120 | 45.0% | 74% | 51% | 59.6% | 3.2% | 55.7% | 100% | 100% | 60% | 37% |
| dense_question | dense, full-question query | 262,144 | 120 | 45.8% | 78% | 58% | 58.5% | 0.0% | 57.1% | 100% | 100% | 70% | 43% |
| ladder1 | fusion, 1024-token chunks | 262,144 | 120 | 56.7% | 98% | 88% | 58.1% | 0.0% | 57.1% | 100% | 100% | 93% | 97% |
| ladder2 | fusion, 1024-token chunks, 1 neighbour each side | 262,144 | 120 | 61.7% | 100% | 99% | 61.7% |  | 61.3% | 100% | 100% | 100% | 100% |
| ladder3 | fusion, 1024-token chunks, 1 neighbour each side, fill packing | 262,144 | 120 | 61.7% | 100% | 99% | 61.7% |  | 61.3% | 100% | 100% | 100% | 100% |
| ladder4 | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 262,144 | 120 | 61.7% | 100% | 99% | 61.7% |  | 61.3% | 100% | 100% | 100% | 100% |
| middle_32k | truncation: middle | 32,768 | 120 | 7.5% | 9% | 0% | 81.8% | 0.0% |  | 17% | 17% | 3% | 0% |
| position_32k | position | 32,768 | 120 | 7.5% | 9% | 0% | 81.8% | 0.0% |  | 17% | 17% | 3% | 0% |
| middle_64k | truncation: middle | 65,536 | 120 | 15.0% | 18% | 1% | 85.7% | 0.0% | 100.0% | 37% | 27% | 7% | 0% |
| position_64k | position | 65,536 | 120 | 13.3% | 16% | 1% | 84.2% | 0.0% | 100.0% | 33% | 23% | 7% | 0% |
| middle_128k | truncation: middle | 131,072 | 120 | 26.7% | 42% | 22% | 64.0% | 0.0% | 59.3% | 90% | 53% | 13% | 10% |
| position_128k | position | 131,072 | 120 | 26.7% | 42% | 22% | 62.7% | 0.0% | 65.4% | 97% | 50% | 13% | 10% |
| oracle | oracle: fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 262,144 | 120 | 61.7% | 100% | 100% | 61.7% |  | 61.7% | 100% | 100% | 100% | 100% |
| oracle_32k | oracle: fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | 120 | 75.0% | 100% | 100% | 75.0% |  | 75.0% | 100% | 100% | 100% | 100% |
| oracle_64k | oracle: fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 120 | 63.3% | 100% | 100% | 63.3% |  | 63.3% | 100% | 100% | 100% | 100% |
| oracle_128k | oracle: fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | 120 | 59.2% | 100% | 100% | 59.2% |  | 59.2% | 100% | 100% | 100% | 100% |
| bm25_32k | bm25 | 32,768 | 120 | 25.0% | 33% | 5% | 75.0% | 0.0% | 83.3% | 57% | 43% | 23% | 10% |
| bm25_64k | bm25 | 65,536 | 120 | 36.7% | 52% | 29% | 71.0% | 0.0% | 74.3% | 90% | 63% | 33% | 20% |
| bm25_128k | bm25 | 131,072 | 120 | 47.5% | 76% | 54% | 62.6% | 0.0% | 56.9% | 100% | 90% | 70% | 43% |
| bm25_recipe_32k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | 120 | 45.8% | 65% | 29% | 70.5% | 0.0% | 77.1% | 90% | 67% | 60% | 43% |
| bm25_recipe_64k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 120 | 55.0% | 86% | 65% | 64.1% | 0.0% | 61.5% | 100% | 90% | 90% | 63% |
| bm25_recipe_128k | bm25, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | 120 | 57.5% | 98% | 93% | 58.5% | 0.0% | 57.1% | 100% | 100% | 97% | 97% |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 262,144 | 120 | 61.7% | 100% | 99% | 61.7% |  | 61.3% | 100% | 100% | 100% | 100% |
| upgraded_32k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 32,768 | 120 | 36.7% | 48% | 9% | 77.2% | 0.0% | 72.7% | 53% | 57% | 60% | 20% |
| upgraded_64k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | 120 | 43.3% | 71% | 41% | 61.2% | 0.0% | 57.1% | 97% | 70% | 73% | 43% |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | 120 | 55.8% | 89% | 77% | 62.6% | 0.0% | 58.7% | 100% | 97% | 93% | 67% |
