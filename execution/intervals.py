"""Paired bootstrap intervals for differences between the conditions of one suite.

Questions that share a document are not independent, so each replicate resamples whole clusters
(document sets or documents) and, where strata are given (MRCR conversations), resamples within each
stratum. Every comparison in a suite uses the same replicates.
"""

from __future__ import annotations

import numpy as np


BUDGETS = ("", "_32k", "_64k", "_80k", "_128k")
# Compared besides each condition against middle truncation at its window; a pair is skipped when
# the suite lacks either condition.
PAIRS = [
    ("full", "middle"), ("dense", "position"), ("bm25", "dense"), ("fusion", "dense"),
    ("dense_7500", "dense"), ("dense_question", "dense"),
    ("ladder1", "fusion"), ("ladder2", "ladder1"), ("ladder3", "ladder2"), ("ladder4", "ladder3"),
    ("upgraded", "fusion"), ("upgraded", "ladder4"), ("upgraded", "full"), ("bm25", "full"),
    *[(f"upgraded{budget}", f"bm25{budget}") for budget in BUDGETS],
    *[(f"upgraded{budget}", f"ladder3{budget}") for budget in BUDGETS],
    *[(f"oracle{budget}", f"upgraded{budget}") for budget in BUDGETS],
    ("oracle_32k", "oracle"),
]
REPLICATES = 10000
SEED = 20261109


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


def compare(conditions, clusters, strata=None, *, metric, replicates=REPLICATES, seed=SEED):
    """Table rows for each comparison, with a 95% percentile interval for the difference.

    `conditions` maps each condition name, in suite order, to its `window`, whether it is the `middle`
    truncation reference, and its `outcomes`, question -> correct, for every question in `clusters`.
    """
    for name, condition in conditions.items():
        if condition["outcomes"].keys() != clusters.keys():
            raise ValueError(f"{name} does not have a valid outcome for every question of the suite")
    questions, weights = resample_weights(clusters, strata or dict.fromkeys(clusters, ""), replicates, seed)
    middle = {condition["window"]: name for name, condition in conditions.items() if condition["middle"]}
    pairs = [(name, middle[condition["window"]]) for name, condition in conditions.items()
             if not condition["middle"] and condition["window"] in middle]
    pairs += [pair for pair in PAIRS if set(pair) <= conditions.keys() and pair not in pairs]
    rows = []
    for candidate, baseline in pairs:
        a, b = (np.array([conditions[name]["outcomes"][question] for question in questions], dtype=float)
                for name in (candidate, baseline))
        low, high = np.percentile((weights * (a - b)).sum(axis=1) / weights.sum(axis=1), [2.5, 97.5])
        rows.append({
            "condition": candidate, "window": conditions[candidate]["window"],
            "baseline": baseline, "baseline window": conditions[baseline]["window"],
            metric: a.mean(), f"baseline {metric}": b.mean(), "difference": a.mean() - b.mean(),
            "95% low": low, "95% high": high,
            "gained": int(((a == 1) & (b == 0)).sum()), "lost": int(((a == 0) & (b == 1)).sum()),
        })
    columns = ["condition", "window", "baseline", "baseline window", metric, f"baseline {metric}", "difference",
               "95% low", "95% high", "gained", "lost"]
    return columns, rows
