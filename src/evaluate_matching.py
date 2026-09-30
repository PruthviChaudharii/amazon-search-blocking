# src/evaluate_matching.py
"""
Evaluation utilities for the supervised matching pipeline.
All functions operate on the candidate pairs produced by ``blocking_v2`` and the
predicted match probabilities from a trained ``MatchingModel``.
The implementation is lightweight – it does **not** load the full 1.7M test set;
instead it expects the caller to pass in a list of dictionaries, each representing
one (source1, candidate) pair with the necessary fields.

Key concepts
------------
* **Entity‑level macro F0.5** – For each Source‑1 entity we compute precision and
  recall of the predicted match set versus the ground‑truth set, then average
  the F0.5 across entities. This emphasises precision (β=0.5).
* **Threshold sweep** – Evaluate the macro F0.5 (and precision/recall) for a
  series of probability thresholds to help pick the operating point.
* **Singleton handling** – If an entity has no ground‑truth matches we treat any
  predicted match as a false positive and a missing prediction as true negative.
* **Match‑set evaluation** – For a given threshold we produce a mapping
  ``entity_id -> set(candidate_ids)`` and compare it to the gold mapping.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _group_by_entity(pairs: Iterable[Dict]) -> Dict[str, set]:
    """Group candidate IDs by Source‑1 entity ID.

    ``pairs`` must contain the keys ``s1_id`` and ``cand_id``.
    Returns a dict ``entity_id -> set(candidate_ids)``.
    """
    groups: Dict[str, set] = defaultdict(set)
    for p in pairs:
        groups[p["s1_id"]].add(p["cand_id"])
    return groups


def _precision_recall(
    pred: set, truth: set
) -> Tuple[float, float]:
    """Compute precision and recall for a single entity.

    Handles edge cases where ``truth`` or ``pred`` may be empty.
    """
    if not pred:
        if not truth:
            return 1.0, 1.0  # both empty → perfect
        return 0.0, 0.0
    if not truth:
        return 0.0, 1.0  # no true matches, any prediction is a FP
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return precision, recall


def entity_level_macro_f05(
    predicted_pairs: Iterable[Dict],
    gold_pairs: Iterable[Dict],
) -> float:
    """Calculate macro‑averaged F0.5 across Source‑1 entities.

    Parameters
    ----------
    predicted_pairs: iterable of dicts with ``s1_id`` and ``cand_id`` (predicted).
    gold_pairs: iterable of dicts with ``s1_id`` and ``cand_id`` (ground truth).

    Returns
    -------
    macro_f05: float – average of per‑entity F0.5 scores.
    """
    pred_map = _group_by_entity(predicted_pairs)
    gold_map = _group_by_entity(gold_pairs)
    all_entities = set(pred_map) | set(gold_map)
    f_scores = []
    beta_sq = 0.5 ** 2
    for eid in all_entities:
        p, r = _precision_recall(pred_map.get(eid, set()), gold_map.get(eid, set()))
        if p == 0.0 and r == 0.0:
            f = 0.0
        else:
            f = (1 + beta_sq) * p * r / (beta_sq * p + r)
        f_scores.append(f)
    return float(np.mean(f_scores)) if f_scores else 0.0


def threshold_sweep(
    candidate_pairs: List[Dict],
    gold_pairs: List[Dict],
    prob_key: str = "prob",
    thresholds: List[float] | None = None,
) -> List[Dict]:
    """Evaluate macro F0.5, precision and recall for a list of thresholds.

    ``candidate_pairs`` must contain ``s1_id``, ``cand_id`` and a probability
    column (default name ``prob``). ``gold_pairs`` is the ground‑truth mapping.
    Returns a list of dictionaries, one per threshold, with keys:
        ``threshold``, ``macro_f05``, ``macro_precision``, ``macro_recall``.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05).tolist()
    results = []
    for th in thresholds:
        filtered = [p for p in candidate_pairs if p.get(prob_key, 0.0) >= th]
        # compute per‑entity precision/recall then macro‑average
        pred_map = _group_by_entity(filtered)
        gold_map = _group_by_entity(gold_pairs)
        all_entities = set(pred_map) | set(gold_map)
        precisions, recalls = [], []
        beta_sq = 0.5 ** 2
        f05s = []
        for eid in all_entities:
            p, r = _precision_recall(pred_map.get(eid, set()), gold_map.get(eid, set()))
            precisions.append(p)
            recalls.append(r)
            if p == 0.0 and r == 0.0:
                f = 0.0
            else:
                f = (1 + beta_sq) * p * r / (beta_sq * p + r)
            f05s.append(f)
        macro_f05 = float(np.mean(f05s)) if f05s else 0.0
        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = float(np.mean(recalls)) if recalls else 0.0
        results.append({
            "threshold": th,
            "macro_f05": macro_f05,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
        })
    return results


def predict_match_set(
    candidate_pairs: List[Dict], threshold: float = 0.5, prob_key: str = "prob"
) -> List[Dict]:
    """Return the predicted match set for a given threshold.

    The returned list contains dictionaries with ``s1_id`` and ``cand_id`` for
    each pair whose probability exceeds ``threshold``.
    """
    return [
        {"s1_id": p["s1_id"], "cand_id": p["cand_id"]}
        for p in candidate_pairs
        if p.get(prob_key, 0.0) >= threshold
    ]


# Helper for quick inspection (not part of the public API)
def _debug_print_metrics(results: List[Dict]):
    for r in results:
        print(
            f"Thresh={r['threshold']:.2f} | F0.5={r['macro_f05']:.4f} "
            f"Prec={r['macro_precision']:.4f} Rec={r['macro_recall']:.4f}"
        )

