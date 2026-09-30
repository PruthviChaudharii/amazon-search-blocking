# src/prepare_training_data.py
"""
Prepare leakage‑safe training and validation sets for the supervised matching model.
Key changes from the original baseline:
* Uses ``HistGradientBoostingClassifier`` (memory‑efficient, works on millions of rows).
* Limits hard‑negative sampling to ``MAX_NEG_PER_POS = 2``.
* Generates features directly into a ``numpy`` array (float32) to avoid large Python object overhead.
* Estimates the number of negatives before full feature generation.
* Records runtime for training and inference, then writes a concise markdown report.
"""

import csv
import os
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# Local imports (project modules)
from matching_features import build_feature_dict
from matching_model import MatchingModel
from evaluate_matching import entity_level_macro_f05, threshold_sweep, predict_match_set

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO = 0.80  # proportion of S1 entities for training
MAX_NEG_PER_POS = 2  # hard negatives per positive (reduced from 3)
NEG_SAMPLE_ATTEMPTS = 50  # attempts to find a hard negative per slot

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_source(file_path: str, id_col: int = 0, name_col: int = 1, addr_col: int = 2, country_col: int = 3):
    """Load a source TSV into a dict keyed by entity ID.
    Returns a dict ``entity_id -> {"name":..., "addr":..., "country":...}``.
    """
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) <= max(id_col, name_col, addr_col, country_col):
                continue
            eid = row[id_col].strip()
            data[eid] = {
                "name": row[name_col].strip(),
                "addr": row[addr_col].strip(),
                "country": row[country_col].strip() if len(row) > country_col else "",
            }
    return data

def load_ground_truth(gt_path: str):
    """Load ground‑truth pairs (source1_id, target_id)."""
    pairs = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            src = row[0].strip()
            # a row may contain a comma‑separated list of targets
            for tgt in row[1].split(','):
                tgt = tgt.strip()
                if tgt:
                    pairs.append((src, tgt))
    return pairs

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train_dir = Path("dataset/train")
source1 = load_source(train_dir / "train_source1.tsv")
source2 = load_source(train_dir / "train_source2.tsv")
source3 = load_source(train_dir / "train_source3.tsv")
ground_truth = load_ground_truth(train_dir / "train_ground_truth.tsv")

print(f"Loaded {len(source1):,} S1, {len(source2):,} S2, {len(source3):,} S3 records.")
print(f"Ground-truth pairs (expanded): {len(ground_truth):,}")

# Build positive lookup per S1
pos_by_s1 = defaultdict(set)
for s1_id, tgt_id in ground_truth:
    pos_by_s1[s1_id].add(tgt_id)

positive_pairs = []
for s1_id, tgt_ids in pos_by_s1.items():
    s1_rec = source1.get(s1_id)
    if not s1_rec:
        continue
    for tgt_id in tgt_ids:
        tgt_rec = source2.get(tgt_id) or source3.get(tgt_id)
        if not tgt_rec:
            continue
        positive_pairs.append({
            "s1_id": s1_id,
            "s1_name": s1_rec["name"],
            "s1_address": s1_rec["addr"],
            "s1_country": s1_rec["country"],
            "cand_id": tgt_id,
            "cand_name": tgt_rec["name"],
            "cand_address": tgt_rec["addr"],
            "cand_country": tgt_rec["country"],
            "label": 1,
        })

print(f"Constructed {len(positive_pairs):,} positive pairs.")

# ---------------------------------------------------------------------------
# Build country index for hard‑negative sampling (S2+S3 combined)
# ---------------------------------------------------------------------------
country_to_targets = defaultdict(list)
for eid, rec in {**source2, **source3}.items():
    country_to_targets[rec["country"]].append((eid, rec))

# ---------------------------------------------------------------------------
# Sample hard negatives
# ---------------------------------------------------------------------------
negative_pairs = []
for pos in positive_pairs:
    s1_country = pos["s1_country"]
    candidates = country_to_targets.get(s1_country, [])
    true_ids = pos_by_s1.get(pos["s1_id"], set())
    random.shuffle(candidates)
    neg_added = 0
    attempts = 0
    while neg_added < MAX_NEG_PER_POS and attempts < NEG_SAMPLE_ATTEMPTS * MAX_NEG_PER_POS:
        attempts += 1
        if not candidates:
            break
        cand_id, cand_rec = candidates[attempts % len(candidates)]
        if cand_id in true_ids:
            continue
        # Simple hard‑negative heuristic: at least one token overlap in names
        name_overlap = len(set(pos["s1_name"].lower().split()) & set(cand_rec["name"].lower().split()))
        if name_overlap < 1:
            continue
        negative_pairs.append({
            "s1_id": pos["s1_id"],
            "s1_name": pos["s1_name"],
            "s1_address": pos["s1_address"],
            "s1_country": s1_country,
            "cand_id": cand_id,
            "cand_name": cand_rec["name"],
            "cand_address": cand_rec["addr"],
            "cand_country": cand_rec["country"],
            "label": 0,
        })
        neg_added += 1

print(f"Sampled {len(negative_pairs):,} hard negatives (target ≤ {MAX_NEG_PER_POS} per positive).")

# ---------------------------------------------------------------------------
# Assemble full dataset and split by S1 entity (leakage‑safe)
# ---------------------------------------------------------------------------
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

unique_s1 = list({p["s1_id"] for p in all_pairs})
random.shuffle(unique_s1)
train_cnt = int(len(unique_s1) * TRAIN_RATIO)
train_s1_set = set(unique_s1[:train_cnt])
val_s1_set = set(unique_s1[train_cnt:])

train_data = [p for p in all_pairs if p["s1_id"] in train_s1_set]
val_data = [p for p in all_pairs if p["s1_id"] in val_s1_set]

print(f"Train S1 entities: {len(train_s1_set):,}, validation S1 entities: {len(val_s1_set):,}")
print(f"Train pairs: {len(train_data):,}, validation pairs: {len(val_data):,}")

# ---------------------------------------------------------------------------
# Feature generation – build a NumPy matrix directly (float32)
# ---------------------------------------------------------------------------
feature_keys = None

def build_matrix(pairs):
    global feature_keys
    n = len(pairs)
    # First pair defines the key order
    if feature_keys is None:
        sample_feat = build_feature_dict({
            "s1_name": pairs[0]["s1_name"],
            "s1_address": pairs[0]["s1_address"],
            "s1_country": pairs[0]["s1_country"],
            "cand_name": pairs[0]["cand_name"],
            "cand_address": pairs[0]["cand_address"],
            "cand_country": pairs[0]["cand_country"],
            "cand_id": pairs[0]["cand_id"],
        })
        feature_keys = sorted(sample_feat.keys())
    X = np.empty((n, len(feature_keys)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    for i, p in enumerate(pairs):
        fdict = build_feature_dict({
            "s1_name": p["s1_name"],
            "s1_address": p["s1_address"],
            "s1_country": p["s1_country"],
            "cand_name": p["cand_name"],
            "cand_address": p["cand_address"],
            "cand_country": p["cand_country"],
            "cand_id": p["cand_id"],
        })
        X[i, :] = [float(fdict[k]) for k in feature_keys]
        y[i] = p["label"]
    return X, y

print("Generating training feature matrix …")
X_train, y_train = build_matrix(train_data)
print("Generating validation feature matrix …")
X_val, y_val = build_matrix(val_data)

# ---------------------------------------------------------------------------
# Model training – HistGradientBoostingClassifier
# ---------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier

model = MatchingModel(classifier=HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42,
))

start_train = time.time()
model.fit([dict(zip(feature_keys, row)) for row in X_train], y_train.tolist())
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f}s")

model_path = Path("models/matching_model.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(str(model_path))
print(f"Model saved to {model_path}")

# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------
start_inf = time.time()
val_probs = model.predict_proba([dict(zip(feature_keys, row)) for row in X_val])
inf_time = time.time() - start_inf
print(f"Inference on validation set: {inf_time:.2f}s")

# Attach probabilities for metric utilities
for prob, pair in zip(val_probs, val_data):
    pair["prob"] = prob

# Pairwise metrics at 0.5 threshold (simple baseline)
threshold = 0.5
pairwise_tp = sum(1 for p in val_data if p["label"] == 1 and p["prob"] >= threshold)
pairwise_fp = sum(1 for p in val_data if p["label"] == 0 and p["prob"] >= threshold)
pairwise_fn = sum(1 for p in val_data if p["label"] == 1 and p["prob"] < threshold)
pairwise_precision = pairwise_tp / (pairwise_tp + pairwise_fp) if (pairwise_tp + pairwise_fp) > 0 else 0.0
pairwise_recall = pairwise_tp / (pairwise_tp + pairwise_fn) if (pairwise_tp + pairwise_fn) > 0 else 0.0
beta_sq = 0.5 ** 2
pairwise_f05 = (1 + beta_sq) * pairwise_precision * pairwise_recall / (beta_sq * pairwise_precision + pairwise_recall) if (pairwise_precision + pairwise_recall) > 0 else 0.0

# Entity‑level macro F0.5 sweep
sweep = threshold_sweep(val_data, [], prob_key="prob", thresholds=[i/100 for i in range(0, 101, 5)])
best = max(sweep, key=lambda x: x["macro_f05"])
best_thresh = best["threshold"]
entity_macro_f05 = best["macro_f05"]

# Singleton analysis (entities with no true matches in validation)
true_counts = Counter()
for p in val_data:
    if p["label"] == 1:
        true_counts[p["s1_id"]] += 1
singleton_ids = [eid for eid in val_s1_set if true_counts.get(eid, 0) == 0]
singleton_fp = sum(1 for p in val_data if p["s1_id"] in singleton_ids and p["prob"] >= best_thresh)
singleton_fp_rate = singleton_fp / len(singleton_ids) if singleton_ids else 0.0

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
report_path = Path("reports/matching_baseline_results.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Baseline Matching Model Results (HistGradientBoosting)\n\n")
    f.write(f"**Positive pairs:** {len(positive_pairs):,}\n")
    f.write(f"**Negative pairs:** {len(negative_pairs):,}\n")
    f.write(f"**Total training rows:** {len(X_train):,}\n")
    f.write(f"**Training S1 entities:** {len(train_s1_set):,}\n")
    f.write(f"**Validation S1 entities:** {len(val_s1_set):,}\n\n")
    f.write(f"**Feature count:** {len(feature_keys)}\n\n")
    f.write(f"**Training runtime:** {train_time:.2f}s\n")
    f.write(f"**Inference runtime (validation):** {inf_time:.2f}s\n\n")
    f.write("## Pairwise metrics @ 0.5 threshold\n")
    f.write(f"- Precision: {pairwise_precision:.4f}\n")
    f.write(f"- Recall: {pairwise_recall:.4f}\n")
    f.write(f"- F0.5: {pairwise_f05:.4f}\n\n")
    f.write("## Entity‑level macro F0.5 (best threshold)\n")
    f.write(f"- Best threshold: {best_thresh:.2f}\n")
    f.write(f"- Macro F0.5: {entity_macro_f05:.4f}\n\n")
    f.write("## Singleton performance\n")
    f.write(f"- Validation singletons: {len(singleton_ids):,}\n")
    f.write(f"- False‑positive predictions on singletons (best thresh): {singleton_fp} ({singleton_fp_rate:.2%})\n\n")
    f.write(f"**Model checkpoint:** `{model_path}`\n")
    f.write("\nGenerated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

print(f"Report written to {report_path}")
