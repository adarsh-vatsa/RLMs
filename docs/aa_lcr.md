# AA-LCR Benchmark

AA-LCR stands for **Artificial Analysis Long Context Reasoning**. It evaluates
whether a model can reason over a large, ordered collection of documents and
produce a free-form answer. The reasoning track used in this repository contains
100 questions grouped into 30 document sets spanning academic papers, company
documents, legal material, and marketing sources.

AA-LCR is more demanding than a needle-in-a-haystack retrieval test. A question
may require the model to identify relevant documents, connect facts across them,
perform a calculation or comparison, apply exclusions, and format the result as
requested. Because answers are free-form, semantically equivalent wording should
be accepted rather than requiring exact string equality.

## Example tasks

The following examples are abbreviated from the reasoning track:

- **Cross-document identification:** Find the company and quarter associated
  with a 13.5% decline in operating income, then report adjusted EBITDA.
  Reference answer: `Equinix, $901 million`.
- **Comparison and ranking:** Rank the explicitly mentioned industries by ACCC
  consumer-related infringement counts while excluding broadcasting.
  Reference answer: `Airline Industry (12), Accommodation Industry (4)`.
- **Multi-step calculation:** Compare the total stockholders' equity of the
  Nasdaq- and NYSE-listed companies at a specified date and report the
  difference in thousands of dollars. Reference answer: `5,597,456`.

For example, a prediction such as `Equinix, $901 M` should be considered
equivalent to `Equinix, $901 million` when the company and amount are correct.

## Use in this repository

This project compares direct and semantic-cache-enabled execution at 262K and
64K context windows. It uses the official reasoning questions and reference
answers, but its model choices and equality grader make the resulting scores
internal comparison results rather than official AA-LCR leaderboard scores.

See the [AA-LCR Jarvis runbook](jarvis/AA_LCR_RUNBOOK.md) for dataset preparation,
service startup, experiment commands, and result comparison.

Official resources: [Artificial Analysis evaluation](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)
and [AA-LCR dataset](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR).
