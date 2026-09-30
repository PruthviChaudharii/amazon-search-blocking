# reports/matching_model_design.md

## Matching Model Design Document

### 1. Feature Groups
The feature engineering module **src/matching_features.py** provides a deterministic, lightweight set of features grouped as follows:

| Group | Features (example) |
|-------|--------------------|
| **Name‑based** | `name_exact_eq`, `core_name_eq`, `token_overlap`, `token_overlap_idf`, `char_sim`, `trigram_sim`, `wordorder_jaccard`, `legal_suffix_agreement` |
| **Address‑based** | `addr_exact_eq`, `addr_token_overlap`, `postal_agreement`, `house_num_agreement`, `missing_addr` |
| **Contact / Identifier** | `phone_agreement`, `domain_eq` |
| **Miscellaneous** | `country_agreement`, `source_indicator`, `missing_name`, `missing_address` |

All features are simple numeric scalars (0/1, float in \[0,1\]) that can be concatenated into a flat vector.

### 2. Positive‑Pair Construction
A **positive pair** is a `(Source‑1 entity, candidate)` where the candidate originates from **Source‑2** or **Source‑3** and is listed in the ground‑truth file `train_ground_truth.tsv`.

* Load `train_ground_truth.tsv` – each line contains `source1_id`, `source2_or_3_id`.
* Join with the original source TSVs (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`) to retrieve the raw attribute fields.
* For every ground‑truth match, compute the feature dictionary using `build_feature_dict`.

### 3. Hard‑Negative Sampling
To teach the classifier what *not* to match we sample **hard negatives**:

1. **Same‑Country, Different Entity** – candidates that share the same country as the source entity but are not listed as a match.
2. **High‑Scoring Blocking Candidates** – after running the V2 blocking pipeline (with the optimized Pass‑C settings), take the top‑N candidates per source that are **not** in the gold set. These are the most confusing non‑matches.
3. **Random Negatives** – a small proportion from other countries to keep the model robust.

Each negative is labeled `0`.

### 4. Source‑1 Entity‑Level Train/Validation Split (Leakage Prevention)
To avoid any leakage between training and evaluation we split **by Source‑1 entity**:

* **Training set** – 80 % of distinct `source1_id`s.
* **Validation set** – the remaining 20 %.

All candidates (positives and negatives) associated with a given `source1_id` stay together in the same split. This guarantees that the model never sees any candidate for a test entity during training.

### 5. Model Interface (`src/matching_model.py`)
A thin wrapper around a scikit‑learn classifier (default: `GradientBoostingClassifier`).  It provides:

* `fit(X, y)` – accepts a list of feature dictionaries and binary labels.
* `predict_proba(X)` – returns the probability of the positive class.
* `save(path)` / `load(path)` – persistence via `joblib` while also storing the feature order.

The wrapper is deliberately lightweight; the user can replace the underlying classifier (e.g., LightGBM, CatBoost) without changing the surrounding pipeline.

### 6. Evaluation (`src/evaluate_matching.py`)
The evaluation utilities compute **entity‑level macro F0.5**, which emphasizes precision (β = 0.5).  The workflow:

1. **Threshold sweep** – iterate over probability thresholds (default 0.0 → 1.0 step 0.05).
2. For each threshold, construct the predicted match‑set (`s1_id → {cand_id}`).
3. Compute per‑entity precision, recall and F0.5, then macro‑average across all entities.
4. Return a table of metrics for helping the user pick the optimal threshold.

Singleton handling is built‑in: if an entity has no ground‑truth matches, any predicted match counts as a false positive, and a missing prediction counts as true negative.

### 7. Threshold Optimization for Macro F0.5
Because the business objective favours precision (reducing false‑positives), we recommend choosing the smallest threshold that yields the **maximum macro F0.5** on the validation split.  The `threshold_sweep` function returns the full curve so the user can inspect the trade‑off.

---

**Next Steps (once `task‑178` finishes):**
1. Load the generated candidate file (`output/candidate_pairs_v2.tsv`).
2. Generate training data using the positive‑pair construction and hard‑negative sampling described above.
3. Fit a `MatchingModel`, persist it with `model.save('models/matching_model.joblib')`.
4. Run a threshold sweep on the validation set and pick the optimal threshold.
5. Apply the model and chosen threshold to the full test candidate set and evaluate with the macro F0.5 metric.

The files created are ready for these downstream steps.
