# AA-LCR Four-Run Results

## Technical summary

The results are positive for **retrieval under context pressure**. At 64K, the
hybrid system reached **41% accuracy versus 21% for direct middle truncation**, a
paired improvement of **20 percentage points** (document-set-clustered 95% CI:
**+9.3 to +32.2 points**). It recovered 28 questions that direct truncation
missed while regressing on 8.

At 262K, both systems scored **44%** and produced identical answers on all 100
questions. This was expected: every prompt fit within the 240K input budget, so
the hybrid runner used the same full-context `direct_fit` path as the direct
runner. The 64K hybrid retained **93.2% of full-context aggregate accuracy**
(41% versus 44%); its -3 point difference from the 262K result was within the
clustered interval (-14.8 to +7.7 points).

I agree with the proposed interpretation, with one qualification: longer inputs
could create the same retrieval opportunity at 262K, but this run does not prove
that outcome. Current full prompts were only 87.8K-122.1K tokens, well below the
240K budget, so the 262K retrieval route was never exercised.

## What AA-LCR measures

Artificial Analysis Long Context Reasoning (AA-LCR) tests free-form reasoning
over ordered sets of long documents rather than simple needle retrieval. The
reasoning track used here contains **100 questions across 30 document sets and
229 referenced documents**, covering academic work, company filings, government
consultations, industry reports, legal material, marketing, and surveys. Tasks
require cross-document identification, comparison, calculation, ranking, and
following exclusions or output constraints.

Representative examples include:

- Identify the company and quarter associated with a 13.5% operating-income
  decline, then report adjusted EBITDA: `Equinix, $901 million`.
- Rank industries by ACCC consumer-related infringement counts while excluding
  broadcasting: `Airline Industry (12), Accommodation Industry (4)`.
- Compare two listed companies' stockholders' equity at a specified date and
  report the difference in thousands: `5,597,456`.

Answers were judged for semantic equivalence by Qwen3.5-35B-A3B at temperature
zero, so formatting variants such as `$901 M` and `$901 million` can receive the
same grade.

## Results

| Experiment | Route for all 100 questions | Accuracy (95% CI) | Total input tokens | Wall time |
|---|---|---:|---:|---:|
| `direct_262k` | Full-context `direct_fit` | 44% (32.7%-56.2%) | 10.70M | 14m 47s |
| `hybrid_262k` | Full-context `direct_fit` | 44% (32.7%-56.2%) | 10.71M | 14m 32s |
| `direct_64k` | `middle_truncated` | 21% (13.8%-29.0%) | 6.05M | 10m 16s |
| `hybrid_64k` | `dense_child_packed` | 41% (30.2%-53.2%) | 5.67M | 12m 36s |

The 64K improvement appeared across every document category at the aggregate
level, although category samples are small and company documents account for 63
of the 100 questions. Company documents contributed 11 of the net 20 additional
correct answers. The hybrid run also used 6.3% fewer input tokens than direct
64K, but took 22.7% longer in this single run because retrieval and ingestion add
work; latency should not be treated as stable without repeated trials.

## Experimental validity and limitations

All four accepted runs used the same question order, dataset revision, prompts,
Qwen3.6-35B-A3B executor, Qwen3.5-35B-A3B grader, 512-token output allowance,
and temperature-zero non-thinking configuration. Every run completed all 100
questions with no API errors or invalid grades. Confidence intervals use 20,000
bootstrap samples clustered by the 30 document sets.

The key limitation is that **no exact or semantic cache hits occurred**. The
hybrid advantage therefore measures evidence retrieval and packing, not answer
reuse or cache savings. Six candidate semantic matches in each hybrid run were
checked and rejected. In addition, this is one run per cell with an LLM equality
grader; the 36 questions on which the two 64K methods disagreed merit manual
grading audit.

## Recommended next step

Create a controlled over-240K extension of the same document sets and rerun the
262K pair. The direct cell should apply its defined truncation policy while the
hybrid cell retrieves and packs evidence. This would directly test whether the
64K benefit carries into genuinely over-budget 262K inputs. A forced-retrieval
262K ablation on the current dataset would separately show whether selective
evidence packing helps even when the complete documents fit.

Evidence: [four-run comparison](../../benchmark_artifacts/aa_lcr/comparisons/20260831T001712Z/comparison.json),
[dataset overview](../aa_lcr.md), and [Jarvis runbook](../jarvis/AA_LCR_RUNBOOK.md).
