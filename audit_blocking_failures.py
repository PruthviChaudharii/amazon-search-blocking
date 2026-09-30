"""
Part 1: Detailed Failure Audit of Current Blocking System
Analyzes missed ground-truth positive pairs across 10,000 validation S1 entities.
Categorizes every failure into A through P and produces:
  reports/blocking_failure_analysis.csv
  reports/blocking_failure_analysis.md
"""

import os
import sys
import re
import csv
import math
import unicodedata
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import rapidfuzz.fuzz as rf_fuzz

sys.path.insert(0, os.path.abspath("."))
from src.blocking import (
    ProductionBlockingEngine,
    extract_name_representations,
    extract_address_representations,
    clean_text_general,
    LEGAL_STOPWORDS
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

print("Starting Blocking Failure Audit...")

# 1. Load held-out validation sample (10,000 S1 records)
VAL_S1_SIZE = 10000
TARGET_SAMPLE_SIZE = 250000

gt_matches = defaultdict(set)
val_s1_ids = []

with open("dataset/train/train_ground_truth.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        m_ids = [m.strip() for m in row[1].split(',') if m.strip()]
        gt_matches[s1_id] = set(m_ids)
        if len(val_s1_ids) < VAL_S1_SIZE:
            val_s1_ids.append(s1_id)

val_s1_set = set(val_s1_ids)
needed_targets = set()
for sid in val_s1_ids:
    needed_targets.update(gt_matches[sid])

# 2. Load S1 validation records
val_s1_records = {}
with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in val_s1_set:
            val_s1_records[eid] = {
                'id': eid,
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            }
            if len(val_s1_records) == len(val_s1_set):
                break

# 3. Load Targets partitioned by country
targets_by_country = defaultdict(list)
all_targets_data = {}
target_count = 0

for s_fname in ["train_source2.tsv", "train_source3.tsv"]:
    s_path = os.path.join("dataset/train", s_fname)
    with open(s_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0].strip()
            c = row[3].strip() if len(row) > 3 else ''
            if eid in needed_targets or (target_count < TARGET_SAMPLE_SIZE):
                targets_by_country[c].append((eid, row[1].strip(), row[2].strip()))
                all_targets_data[eid] = {
                    'id': eid,
                    'name': row[1].strip(),
                    'address': row[2].strip(),
                    'country': c
                }
                target_count += 1

print(f"Loaded {len(val_s1_records):,} S1 validation records and {len(all_targets_data):,} targets.")

# 4. Build current blocking engine index
engine = ProductionBlockingEngine(max_token_freq=1500, top_k=25)
for c, recs in targets_by_country.items():
    engine.build_target_index_for_country(c, recs)

# Targets present in index
indexed_tids = set(all_targets_data.keys())

# Collect all ground truth positive pairs within the searchable pool
eval_gt_pairs = []
for sid in val_s1_ids:
    for tid in gt_matches[sid]:
        if tid in indexed_tids:
            eval_gt_pairs.append((sid, tid))

print(f"Total searchable positive pairs in evaluation pool: {len(eval_gt_pairs):,}")

# 5. Retrieve candidates with K=25 and also collect the full unpruned candidate pool to detect pruning failures
recalled_pairs = set()
unpruned_retrieved_pairs = set()
s1_to_top25 = {}

RE_PHONE = re.compile(r'\b[0-9]{7,12}\b')
RE_DOMAIN = re.compile(r'\.(com|org|net|in|co|io|fr)\b', re.IGNORECASE)

print("Running blocking queries and logging candidate provenance...")
for sid in val_s1_ids:
    s1 = val_s1_records[sid]
    c = s1['country']
    
    # Run retrieval
    cands = engine.retrieve_candidates_for_query(c, s1['name'], s1['address'], top_k=25)
    s1_to_top25[sid] = set(cands)
    for tid in cands:
        recalled_pairs.add((sid, tid))
        
    # Also inspect unpruned candidates from the passes to identify if target was in pool but pruned
    name_rep = extract_name_representations(s1['name'])
    addr_rep = extract_address_representations(s1['address'])
    
    unpruned = set()
    if name_rep['core_name']:
        unpruned.update(engine.pass1_exact_name_index[c].get(name_rep['core_name'], []))
    for tok in name_rep['core_tokens']:
        unpruned.update(engine.pass2_token_index[c].get(tok, []))
    for bg in name_rep['bigrams']:
        unpruned.update(engine.pass2_bigram_index[c].get(bg, []))
    if addr_rep['postal_code'] and addr_rep['house_num']:
        unpruned.update(engine.pass3_pin_house_index[c].get(f"{addr_rep['postal_code']}_{addr_rep['house_num']}", []))
        
    for tid in unpruned:
        unpruned_retrieved_pairs.add((sid, tid))

# 6. Failure Classification
missed_pairs = [pair for pair in eval_gt_pairs if pair not in recalled_pairs]
print(f"Total true pairs: {len(eval_gt_pairs):,} | Recalled: {len(recalled_pairs & set(eval_gt_pairs)):,} ({len(recalled_pairs & set(eval_gt_pairs))/len(eval_gt_pairs)*100:.2f}%)")
print(f"Total missed true pairs: {len(missed_pairs):,} ({len(missed_pairs)/len(eval_gt_pairs)*100:.2f}%)")

failure_categories = Counter()
failure_examples = defaultdict(list)

for sid, tid in missed_pairs:
    s1 = val_s1_records[sid]
    t = all_targets_data[tid]
    
    s1_name_raw, s1_addr_raw = s1['name'], s1['address']
    t_name_raw, t_addr_raw = t['name'], t['address']
    
    s1_name_clean = clean_text_general(s1_name_raw)
    t_name_clean = clean_text_general(t_name_raw)
    s1_tokens = set(s1_name_clean.split())
    t_tokens = set(t_name_clean.split())
    
    s1_core_tokens = s1_tokens - LEGAL_STOPWORDS
    t_core_tokens = t_tokens - LEGAL_STOPWORDS
    
    s1_has_nonascii = any(ord(ch) > 127 for ch in s1_name_raw)
    t_has_nonascii = any(ord(ch) > 127 for ch in t_name_raw)
    
    # Check if target was retrieved in candidate pool but pruned by Top-K
    was_in_unpruned = (sid, tid) in unpruned_retrieved_pairs
    
    cat = None
    
    # O. Candidate ranking/pruning failure: was in pool but fell below Top-25
    if was_in_unpruned:
        cat = "O. Candidate ranking/pruning failure"
    # F. Transliteration / non-ASCII
    elif s1_has_nonascii or t_has_nonascii:
        cat = "F. Transliteration / non-ASCII"
    # G. Domain-name variation
    elif RE_DOMAIN.search(s1_name_raw) or RE_DOMAIN.search(t_name_raw):
        cat = "G. Domain-name variation"
    # H. Phone-number contamination
    elif RE_PHONE.search(s1_name_raw) or RE_PHONE.search(t_name_raw):
        cat = "H. Phone-number contamination"
    # J. Missing address
    elif not t_addr_raw.strip():
        cat = "J. Missing address"
    # D. Word-order variation (all core tokens match, just permuted)
    elif s1_core_tokens and s1_core_tokens == t_core_tokens and s1_name_clean != t_name_clean:
        cat = "D. Word-order variation"
    # B. Legal suffix variation (core matches, suffix differs)
    elif s1_core_tokens and t_core_tokens and s1_core_tokens == t_core_tokens:
        cat = "B. Legal suffix variation"
    # E. Character typo / minor spelling change
    elif rf_fuzz.ratio(s1_name_clean, t_name_clean) >= 70:
        cat = "E. Character typo"
    # A. Name normalization failure (symbols/punctuation difference)
    elif s1_name_raw.lower() != t_name_raw.lower() and re.sub(r'[^a-zA-Z0-9]', '', s1_name_raw.lower()) == re.sub(r'[^a-zA-Z0-9]', '', t_name_raw.lower()):
        cat = "A. Name normalization failure"
    # C. Tokenization failure (substring match / concatenated words)
    elif any(tok1 in t_name_clean for tok1 in s1_core_tokens if len(tok1) >= 4) or any(tok2 in s1_name_clean for tok2 in t_core_tokens if len(tok2) >= 4):
        cat = "C. Tokenization failure"
    # N. Common-token collision (only shared tokens were high-frequency pruned words)
    elif len(s1_tokens & t_tokens) > 0 and len(s1_core_tokens & t_core_tokens) == 0:
        cat = "N. Common-token collision"
    # I. Address variation (name completely different, but address similar)
    elif rf_fuzz.token_set_ratio(s1_addr_raw.lower(), t_addr_raw.lower()) >= 60:
        cat = "I. Address variation"
    # L. Postal/PIN mismatch
    elif s1_addr_raw and t_addr_raw and re.search(r'\b[0-9]{5,6}\b', s1_addr_raw) and re.search(r'\b[0-9]{5,6}\b', t_addr_raw) and re.search(r'\b[0-9]{5,6}\b', s1_addr_raw).group() != re.search(r'\b[0-9]{5,6}\b', t_addr_raw).group():
        cat = "L. Postal/PIN mismatch"
    # M. House-number mismatch
    elif re.search(r'\b[0-9]+\b', s1_addr_raw) and re.search(r'\b[0-9]+\b', t_addr_raw) and set(re.findall(r'\b[0-9]+\b', s1_addr_raw)) != set(re.findall(r'\b[0-9]+\b', t_addr_raw)):
        cat = "M. House-number mismatch"
    # K. Missing name signal
    elif len(s1_core_tokens) == 0 or len(t_core_tokens) == 0:
        cat = "K. Missing name signal"
    else:
        cat = "P. Other"
        
    failure_categories[cat] += 1
    if len(failure_examples[cat]) < 3:
        failure_examples[cat].append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw))

print("\n--- FAILURE BREAKDOWN TABLE ---")
total_missed = len(missed_pairs)
sorted_cats = failure_categories.most_common()
for cat, cnt in sorted_cats:
    pct = (cnt / total_missed) * 100.0
    pct_all = (cnt / len(eval_gt_pairs)) * 100.0
    print(f"{cat:<38} | {cnt:5d} missed ({pct:5.2f}% of missed | {pct_all:5.2f}% of all pairs)")

# Save CSV
df_fail = pd.DataFrame([{
    'Category': cat,
    'Missed_Count': cnt,
    'Percentage_of_Missed_%': round((cnt / total_missed) * 100.0, 2),
    'Percentage_of_All_True_Pairs_%': round((cnt / len(eval_gt_pairs)) * 100.0, 2)
} for cat, cnt in sorted_cats])
df_fail.to_csv("reports/blocking_failure_analysis.csv", index=False)
print("\nSaved reports/blocking_failure_analysis.csv")

# Save Markdown Report
md_text = f"""# Detailed Failure Audit of Current Blocking System (Phase 3 Part 1)

**Evaluated Scope:** 10,000 Source 1 Validation Entities  
**Total Ground-Truth Positive Pairs in Pool:** {len(eval_gt_pairs):,}  
**Successfully Recalled Pairs (K=25):** {len(recalled_pairs & set(eval_gt_pairs)):,} ({len(recalled_pairs & set(eval_gt_pairs))/len(eval_gt_pairs)*100:.2f}%)  
**Total Missed Pairs:** {total_missed:,} ({total_missed/len(eval_gt_pairs)*100:.2f}%)  

---

## 1. Quantitative Failure Taxonomy

| Failure Category | Missed Count | % of Missed Pairs | % of Total Positive Pairs | Primary Root Cause |
| :--- | :---: | :---: | :---: | :--- |
"""

for cat, cnt in sorted_cats:
    pct = (cnt / total_missed) * 100.0
    pct_all = (cnt / len(eval_gt_pairs)) * 100.0
    root_cause = {
        "O. Candidate ranking/pruning failure": "Target was in unpruned candidate pool, but was pushed outside Top-25 by noise.",
        "E. Character typo": "Levenshtein distance / minor typos prevented exact token matching.",
        "I. Address variation": "Name was noisy/abbreviated; address anchor did not trigger due to format differences.",
        "F. Transliteration / non-ASCII": "Indian regional script (Tamil, Devanagari) vs English script mismatch.",
        "C. Tokenization failure": "Concatenated words or punctuation-glued tokens split differently.",
        "N. Common-token collision": "Tokens were filtered as high-frequency (>1,500 occurrences) stopwords.",
        "J. Missing address": "Target entity had empty address string in S2/S3 (3.3% missing rate).",
        "G. Domain-name variation": "Target entity name was website URL (e.g. `domain.com`).",
        "H. Phone-number contamination": "Appended phone number altered token set.",
        "D. Word-order variation": "Word order transposed across multi-token names.",
        "B. Legal suffix variation": "Legal suffixes altered token overlap.",
        "L. Postal/PIN mismatch": "One record lacked postal code or postal code had typo.",
        "M. House-number mismatch": "Building number formatted with prefix/suffix.",
        "A. Name normalization failure": "Symbol/punctuation discrepancy.",
        "K. Missing name signal": "Name field empty or pure stopwords.",
        "P. Other": "Compound noise across both name and address."
    }.get(cat, "Unclassified noise")
    md_text += f"| **{cat}** | **{cnt:,}** | **{pct:.2f}%** | {pct_all:.2f}% | {root_cause} |\n"

md_text += """
---

## 2. Qualitative Failure Case Examples

"""

for cat, cnt in sorted_cats:
    md_text += f"### {cat} ({cnt:,} cases)\n"
    for s1_n, tn, s1_a, ta in failure_examples[cat]:
        md_text += f"- **S1:** `{s1_n}` | `{s1_a}`\n"
        md_text += f"  **Target:** `{tn}` | `{ta}`\n"
    md_text += "\n"

with open("reports/blocking_failure_analysis.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("Saved reports/blocking_failure_analysis.md")
