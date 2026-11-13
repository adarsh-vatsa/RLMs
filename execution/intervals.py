"""Paired bootstrap intervals and randomization tests for differences between the conditions of one suite.

Questions that share a document are not independent, so each replicate resamples whole clusters
(document sets or documents) and, where strata are given (MRCR conversations), resamples within each
stratum. The randomization test swaps the two conditions' outcomes cluster by cluster. Every comparison
in a suite uses the same replicates and the same swaps.
"""

from __future__ import annotations

import numpy as np


BUDGETS = ("", "_32k", "_64k", "_80k", "_128k")
# Compared besides each condition against middle truncation at its window; a pair is skipped when
# the suite lacks either condition.
PAIRS = [
    ("full", "middle"), ("dense", "position"),
    *[(f"bm25{budget}", f"position{budget}") for budget in BUDGETS],
    *[(f"upgraded{budget}", f"position{budget}") for budget in BUDGETS],
    ("bm25", "dense"), ("fusion", "dense"),
    ("dense_7500", "dense"), ("dense_question", "dense"),
    ("ladder1", "fusion"), ("ladder2", "ladder1"), ("ladder3", "ladder2"), ("ladder4", "ladder3"),
    ("upgraded", "fusion"), ("upgraded", "ladder4"), ("upgraded", "full"), ("bm25", "full"),
    *[(f"upgraded{budget}", f"bm25{budget}") for budget in BUDGETS],
    *[(f"upgraded{budget}", f"ladder3{budget}") for budget in BUDGETS],
    *[(f"oracle{budget}", f"upgraded{budget}") for budget in BUDGETS],
    ("oracle_32k", "oracle"),
    # Controls that take the recipe's chunking, neighbours, packing and floor, so only the ranking differs.
    *[(f"position_recipe{budget}", f"position{budget}") for budget in BUDGETS],
    *[(f"bm25_recipe{budget}", f"bm25{budget}") for budget in BUDGETS],
    *[(f"bm25_recipe{budget}", f"position_recipe{budget}") for budget in BUDGETS],
    *[(f"upgraded{budget}", f"position_recipe{budget}") for budget in BUDGETS],
    *[(f"upgraded{budget}", f"bm25_recipe{budget}") for budget in BUDGETS],
]
REPLICATES = 10000
SEED = 20261109
PERMUTATIONS = 100000
PERMUTATION_SEED = 20261113


def resample_weights(clusters, strata, replicates=REPLICATES, seed=SEED):
    """How often each question is drawn in each replicate: whole clusters, resampled within each stratum."""
    questions = sorted(clusters)
    weights = np.zeros((replicates, len(questions)))
    rng = np.random.default_rng(seed)
    for stratum in sorted(set(strata.values())):
        names = sorted({clusters[question] for question in questions if strata[question] == stratum})
        counts = np.zeros((replicates, len(names)))
        draws = rng.integers(len(names), size=(replicates, len(names)))
        np.add.at(counts, (np.arange(replicates)[:, None], draws), 1)
        column = {name: index for index, name in enumerate(names)}
        for index, question in enumerate(questions):
            if strata[question] == stratum:
                weights[:, index] = counts[:, column[clusters[question]]]
    return questions, weights


def swaps(clusters, questions, permutations=PERMUTATIONS, seed=PERMUTATION_SEED):
    """Each question's cluster index, and random signs that swap the two conditions of a pair per cluster."""
    names = sorted(set(clusters.values()))
    column = {name: index for index, name in enumerate(names)}
    membership = np.array([column[clusters[question]] for question in questions])
    signs = np.random.default_rng(seed).integers(0, 2, size=(permutations, len(names)), dtype=np.int8)
    return membership, (2 * signs - 1).astype(np.int32)


def randomization_p(a, b, membership, signs):
    """Two-sided p-value of the mean paired difference under random swaps of the two conditions' outcomes."""
    # Integer counts keep the comparison with the observed gap exact.
    per_cluster = np.rint(np.bincount(membership, weights=a - b, minlength=signs.shape[1])).astype(np.int32)
    swapped = np.abs(signs @ per_cluster)
    return (1 + int((swapped >= abs(int(per_cluster.sum()))).sum())) / (len(signs) + 1)


def compare(conditions, clusters, strata=None, *, metric, references=None, replicates=REPLICATES, seed=SEED,
            permutations=PERMUTATIONS):
    """Table rows for each comparison, with a 95% percentile interval for the difference and the
    uncorrected p-value of a paired randomization test.

    `conditions` maps each condition name, in suite order, to its `window`, whether it is the `middle`
    truncation reference, and its `outcomes`, question -> correct, covering the questions in `clusters`.
    `references` maps a window to the condition, such as full context or an oracle, that bounds what a
    run at that window can recover from middle truncation; each run against middle then also gets the
    share of that gap it recovers.
    """
    for name, condition in conditions.items():
        if not clusters.keys() <= condition["outcomes"].keys():
            raise ValueError(f"{name} does not have a valid outcome for every question of the suite")
    questions, weights = resample_weights(clusters, strata or dict.fromkeys(clusters, ""), replicates, seed)
    membership, signs = swaps(clusters, questions, permutations)
    total = weights.sum(axis=1)
    outcomes = {name: np.array([condition["outcomes"][question] for question in questions], dtype=float)
                for name, condition in conditions.items()}
    middle = {condition["window"]: name for name, condition in conditions.items() if condition["middle"]}
    pairs = [(name, middle[condition["window"]]) for name, condition in conditions.items()
             if not condition["middle"] and condition["window"] in middle]
    pairs += [pair for pair in PAIRS if set(pair) <= conditions.keys() and pair not in pairs]
    rows = []
    for candidate, baseline in pairs:
        a, b = outcomes[candidate], outcomes[baseline]
        low, high = np.percentile((weights * (a - b)).sum(axis=1) / total, [2.5, 97.5])
        row = {
            "condition": candidate, "window": conditions[candidate]["window"],
            "baseline": baseline, "baseline window": conditions[baseline]["window"],
            metric: a.mean(), f"baseline {metric}": b.mean(), "difference": a.mean() - b.mean(),
            "95% low": low, "95% high": high,
            "gained": int(((a == 1) & (b == 0)).sum()), "lost": int(((a == 0) & (b == 1)).sum()),
            "p": randomization_p(a, b, membership, signs),
        }
        reference = (references or {}).get(conditions[candidate]["window"])
        if baseline == middle.get(conditions[candidate]["window"]) and reference in conditions.keys() - {candidate}:
            gap = outcomes[reference] - b
            row.update({"reference": reference, "gap recovered": (a - b).mean() / gap.mean()})
            resampled_gap = (weights * gap).sum(axis=1)
            # A resample in which the reference does no better than middle leaves the ratio undefined;
            # the difference and its interval then carry the result.
            if (resampled_gap > 0).all():
                row["gap recovered 95% low"], row["gap recovered 95% high"] = np.percentile(
                    (weights * (a - b)).sum(axis=1) / resampled_gap, [2.5, 97.5])
        rows.append(row)
    columns = ["condition", "window", "baseline", "baseline window", metric, f"baseline {metric}", "difference",
               "95% low", "95% high", "gained", "lost", "p"]
    if references:
        columns += ["reference", "gap recovered", "gap recovered 95% low", "gap recovered 95% high"]
    return columns, rows
