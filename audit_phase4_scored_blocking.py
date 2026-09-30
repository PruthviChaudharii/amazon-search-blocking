import os
import sys
import csv
import re
import time
from collections import defaultdict, Counter
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

print("Testing Ranked Candidate Selection...")
# We use the same 5000 S1 records and test ranking candidates by simple token overlap
BENCHMARK_S1_SIZE = 2000

gt_matches = defaultdict(set)
sampled_s1_ids = []
with open("dataset/train/train_ground_truth.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        m_ids = [m.strip() for m in row[1].split(',') if m.strip()]
        gt_matches[s1_id] = set(m_ids)
        if len(sampled_s1_ids) < BENCHMARK_S1_SIZE:
            sampled_s1_ids.append(s1_id)

sampled_s1_set = set(sampled_s1_ids)
all_needed_target_ids = set()
for s1 in sampled_s1_ids:
    all_needed_target_ids.update(gt_matches[s1])

s1_records = {}
with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in sampled_s1_set:
            s1_records[eid] = {
                'id': eid,
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            }
            if len(s1_records) == len(sampled_s1_set):
                break

target_records = {}
for s_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
    with open(s_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid in all_needed_target_ids or (len(target_records) < 50000):
                target_records[eid] = {
                    'id': eid,
                    'name': row[1].strip(),
                    'address': row[2].strip(),
                    'country': row[3].strip() if len(row) > 3 else ''
                }

LEGAL_TERMS = {'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 
               'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprises', 'associates', 'the', 'and', 'of'}

def clean_tokens(s):
    s = s.lower()
    s = re.sub(r'\.(com|org|net|in|co|io|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return set(w for w in s.split() if w not in LEGAL_TERMS and len(w) > 1)

# Precompute tokens
s1_tokens_map = {sid: clean_tokens(rec['name']) | clean_tokens(rec['address']) for sid, rec in s1_records.items()}
target_tokens_map = {tid: clean_tokens(rec['name']) | clean_tokens(rec['address']) for tid, rec in target_records.items()}

# Inverted index on Target by Country and (Name tokens + Address 5/6 digit PIN / numbers)
country_token_index = defaultdict(list)
for tid, rec in target_records.items():
    c = rec['country']
    for tok in clean_tokens(rec['name']):
        if len(tok) >= 3:
            country_token_index[(c, 'name', tok)].append(tid)
    # house num / pin
    pins = re.findall(r'\b[0-9]{4,6}\b', rec['address'])
    for p in pins:
        country_token_index[(c, 'pin', p)].append(tid)

# Test candidate recall when we score and take Top-15, Top-25, Top-40 candidates
for top_k in [10, 20, 30, 50]:
    t0 = time.time()
    total_true = 0
    recalled = 0
    cand_counts = []
    
    for s1_id in sampled_s1_ids:
        s1_rec = s1_records[s1_id]
        c = s1_rec['country']
        true_m = gt_matches[s1_id] & set(target_records.keys())
        total_true += len(true_m)
        
        s1_toks = s1_tokens_map[s1_id]
        
        # Accumulate candidate frequencies (scores)
        candidate_scores = Counter()
        for tok in clean_tokens(s1_rec['name']):
            if len(tok) >= 3:
                key = (c, 'name', tok)
                if key in country_token_index:
                    postings = country_token_index[key]
                    if len(postings) <= 1000: # filter ultra-generic tokens
                        for tid in postings:
                            candidate_scores[tid] += 2.0 # name match gets higher weight
                            
        pins = re.findall(r'\b[0-9]{4,6}\b', s1_rec['address'])
        for p in pins:
            key = (c, 'pin', p)
            if key in country_token_index:
                postings = country_token_index[key]
                if len(postings) <= 500:
                    for tid in postings:
                        candidate_scores[tid] += 1.5
                        
        # Take Top K
        top_cands = [tid for tid, _ in candidate_scores.most_common(top_k)]
        cand_counts.append(len(top_cands))
        recalled += len(set(top_cands) & true_m)
        
    rec = recalled / total_true if total_true > 0 else 0
    print(f"Top-{top_k} Scoring Candidates -> Recall: {rec*100:.2f}% ({recalled}/{total_true}), Mean Cands: {np.mean(cand_counts):.1f}, Median: {np.median(cand_counts):.0f}, P95: {np.percentile(cand_counts, 95):.0f}, Time: {time.time()-t0:.2f}s")
