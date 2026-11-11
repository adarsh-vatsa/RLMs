# Control runs

These runs separate the recipe's ranking from the rest of the recipe. They add
nine runs to the existing suites, `20261029`. [RERUN.md](RERUN.md) runs them on
g101, together with the rerun of the g202 runs; this page explains what they
are and how to read them.

| | |
| --- | --- |
| AA-LCR | 6 runs: `position_recipe` and `bm25_recipe` at 64K, and each with `_32k` and `_80k`. Each is graded by Qwen3.6, then by Luna. |
| MRCR v2 | 3 runs: `bm25_recipe_32k`, `bm25_recipe_64k` and `bm25_recipe_128k`. |

## What the runs are

The recipe is fusion ranking plus four steps: 1,024-token chunks with 128 of
overlap, one neighbouring chunk on each side, `fill` packing and the document
floor. `position` and `bm25` ran without those steps, with 7,500-token chunks
(3,500 on MRCR) and `stop` packing. So the recipe's margin over them is partly
a difference in chunking, packing and, on AA-LCR, how many documents each
prompt covers.

Each control keeps the recipe's four steps and changes only the ranking:

- **`position_recipe`** ranks chunks by their distance from the nearer end of
  their document, ignoring the question. Compared with `upgraded`, it shows what
  ranking by the question adds to the same prompt structure. It runs on AA-LCR
  only. On one document, position keeps the head and tail, which is middle
  truncation with an omission marker. On MRCR, `position` is already within 2
  points of `middle` at every window, and `middle_marked` matches it exactly.
- **`bm25_recipe`** ranks chunks by BM25 alone. Compared with `upgraded`, it
  shows what fusion with dense ranking adds. Neither control embeds anything.

The interval pairs for both are in `execution/intervals.py`: each control
against its unmatched version, `bm25_recipe` and `upgraded` against
`position_recipe`, and `upgraded` against `bm25_recipe`, at each window.

## Why one node

Greedy decoding gives the same answer to the same prompt on g101 and g102, but
not on g202, which served the recipe runs and the BM25 sweep on 3 November. The
[experiment list](../workshop_paper_todo.md#experiment-list) has the details.
The controls are compared with the recipe runs, so they must share a node with
them. [RERUN.md](RERUN.md) reruns the recipe runs on g101 and runs the controls
there too, so every comparison in the paper uses g101 or g102.

## Results

`results-gpt-5.6-luna.md` (AA-LCR) and `results.md` (MRCR) gain one row per
control. Their policy column reads, for example, `position, 1024-token chunks,
1 neighbour each side, fill packing, document floor`. On AA-LCR, compare the
`documents covered` column with `upgraded`: with the floor, both should cover
nearly every document.

The comparisons that decide the paper's lead and headline are in
`intervals-gpt-5.6-luna.md` and `intervals.md`:

| Row | Question it answers |
| --- | --- |
| `upgraded` against `position_recipe` | Does ranking by the question help, with structure, chunking and coverage held equal? |
| `bm25_recipe` against `position_recipe` | The same for BM25 alone |
| `upgraded` against `bm25_recipe` | Does fusion with dense ranking add anything to BM25? |
| `position_recipe` against `position`, `bm25_recipe` against `bm25` | What the four steps add to each ranking |

Item 4 of the [workshop paper to-do](../workshop_paper_todo.md#remaining-work)
says how each outcome changes the paper.
