# MRCR v2 suite 20261128_think_pilot: visibility

| condition | policy | window | questions | exact match | target visible | all needles visible | exact match given target visible | oracle exact match where target visible | exact match given target hidden | exact match given all needles visible | target visible at 133K | target visible at 267K | target visible at 533K | target visible at 1.06M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| middle_128k | truncation: middle | 131,072 | 12 | 25.0% | 42% | 25% | 60.0% |  | 0.0% | 66.7% | 100% | 33% | 0% | 33% |
| upgraded_128k | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 131,072 | 12 | 66.7% | 92% | 75% | 72.7% |  | 0.0% | 77.8% | 100% | 100% | 100% | 67% |
