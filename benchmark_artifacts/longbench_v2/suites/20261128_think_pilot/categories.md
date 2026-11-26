# LongBench-v2 suite 20261128_think_pilot: accuracy by category

| condition | policy | window | category | questions | accuracy | correct |
|---|---|---|---|---|---|---|
| middle | truncation: middle | 65,536 | Long In-context Learning | 5 | 100.0% | 5 |
| middle | truncation: middle | 65,536 | Long Structured Data Understanding | 3 | 66.7% | 2 |
| middle | truncation: middle | 65,536 | Multi-Document QA | 4 | 25.0% | 1 |
| middle | truncation: middle | 65,536 | Single-Document QA | 8 | 37.5% | 3 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long In-context Learning | 5 | 60.0% | 3 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Long Structured Data Understanding | 3 | 66.7% | 2 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Multi-Document QA | 4 | 50.0% | 2 |
| upgraded | fusion, 1024-token chunks, 1 neighbour each side, fill packing, document floor | 65,536 | Single-Document QA | 8 | 37.5% | 3 |
