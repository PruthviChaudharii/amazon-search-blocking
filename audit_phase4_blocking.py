import os
import sys
import csv
import re
import time
from collections import defaultdict, Counter
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

print("Starting Blocking Strategies Benchmark...")

# Load a benchmark sample: 5,000 S1 entities (with their ground truth matches)
# And evaluate against a search space of all their matches PLUS a large random pool of targets (e.g., 100,000 targets)
BENCHMARK_S1_SIZE = 5000
BENCHMARK_TARGET_SIZE = 100000

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

print(f"Benchmark subset: {len(sampled_s1_ids)} S1 records.")
print(f"Ground truth target matches in subset: {len(all_needed_target_ids)} targets.")

# Load S1 records
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

# Load Target records (all needed true matches + background noise records up to BENCHMARK_TARGET_SIZE)
target_records = {}
# Read S2
with open("dataset/train/train_source2.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in all_needed_target_ids or (len(target_records) < BENCHMARK_TARGET_SIZE // 2):
            target_records[eid] = {
                'id': eid,
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            }

# Read S3
with open("dataset/train/train_source3.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in all_needed_target_ids or (len(target_records) < BENCHMARK_TARGET_SIZE):
            target_records[eid] = {
                'id': eid,
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            }

print(f"Target pool loaded: {len(target_records):,} records (contains {sum(1 for tid in all_needed_target_ids if tid in target_records)} true matches).")

# Normalization utilities
LEGAL_TERMS = {'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 
               'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprises', 'associates', 'the', 'and', 'of'}

def clean_name(s):
    # remove brackets, punctuation, domain suffixes
    s = s.lower()
    s = re.sub(r'\.(com|org|net|in|co|io|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    words = [w for w in s.split() if w not in LEGAL_TERMS]
    return words

def clean_addr(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) > 1]

def extract_pin(s):
    m = re.findall(r'\b[0-9]{5,6}\b', s)
    return m[0] if m else ''

def extract_house_num(s):
    m = re.findall(r'\b[0-9]+[a-z]?\b', s.lower())
    return m[0] if m else ''

# ----------------- BENCHMARK FUNCTION -----------------
def evaluate_blocking(name, block_func_s1, block_func_target, max_candidates_per_s1=100):
    t0 = time.time()
    
    # 1. Build inverted index from Target records
    index = defaultdict(list)
    for tid, rec in target_records.items():
        keys = block_func_target(rec)
        for k in keys:
            index[k].append(tid)
            
    # Cap overly generic blocks (blocks with > 2000 targets) to avoid exploding complexity
    valid_index = {k: v for k, v in index.items() if len(v) <= 2000}
    
    # 2. Retrieve candidates for S1
    candidate_counts = []
    zero_cand = 0
    total_true_matches = 0
    recalled_matches = 0
    
    total_possible_pairs = len(sampled_s1_ids) * len(target_records)
    total_generated_pairs = 0
    
    for s1_id in sampled_s1_ids:
        s1_rec = s1_records[s1_id]
        true_m = gt_matches[s1_id] & set(target_records.keys())
        total_true_matches += len(true_m)
        
        keys = block_func_s1(s1_rec)
        cands = set()
        for k in keys:
            if k in valid_index:
                cands.update(valid_index[k])
                
        # Limit candidates if requested
        if len(cands) > max_candidates_per_s1:
            cands = set(list(cands)[:max_candidates_per_s1])
            
        c_len = len(cands)
        candidate_counts.append(c_len)
        total_generated_pairs += c_len
        if c_len == 0:
            zero_cand += 1
            
        recalled = len(cands & true_m)
        recalled_matches += recalled
        
    elapsed = time.time() - t0
    recall = recalled_matches / total_true_matches if total_true_matches > 0 else 0.0
    rr = 1.0 - (total_generated_pairs / total_possible_pairs)
    
    print(f"\n--- Strategy: {name} ---")
    print(f"  Candidate Recall:         {recall*100:.2f}% ({recalled_matches:,}/{total_true_matches:,})")
    print(f"  Avg candidates per S1:    {np.mean(candidate_counts):.2f}")
    print(f"  Median candidates:        {np.median(candidate_counts):.1f}")
    print(f"  P95 candidates:           {np.percentile(candidate_counts, 95):.1f}")
    print(f"  P99 candidates:           {np.percentile(candidate_counts, 99):.1f}")
    print(f"  Reduction Ratio:          {rr*100:.6f}%")
    print(f"  S1 with 0 candidates:     {zero_cand:,} ({zero_cand/len(sampled_s1_ids)*100:.2f}%)")
    print(f"  Elapsed time:             {elapsed:.2f} s")
    return {
        'name': name,
        'recall': recall,
        'mean_cands': np.mean(candidate_counts),
        'median_cands': np.median(candidate_counts),
        'p95_cands': np.percentile(candidate_counts, 95),
        'p99_cands': np.percentile(candidate_counts, 99),
        'rr': rr,
        'zero_cand': zero_cand,
        'elapsed': elapsed
    }

# STRATEGY 1: Exact Normalized Name + Country
def strat1_keys(rec):
    c = rec['country']
    words = clean_name(rec['name'])
    if not words: return []
    return [f"{c}_name_{'_'.join(words)}"]

# STRATEGY 2: First 2 Significant Name Words + Country
def strat2_keys(rec):
    c = rec['country']
    words = clean_name(rec['name'])
    if not words: return []
    keys = []
    if len(words) >= 2:
        keys.append(f"{c}_prefix2_{words[0]}_{words[1]}")
    keys.append(f"{c}_word1_{words[0]}")
    return keys

# STRATEGY 3: Significant Name Tokens (Inverted Index) + Country
def strat3_keys(rec):
    c = rec['country']
    words = clean_name(rec['name'])
    # Emit token blocks for words of length >= 4
    keys = []
    for w in words:
        if len(w) >= 4:
            keys.append(f"{c}_tok_{w}")
    return keys

# STRATEGY 4: Address PIN / House Number + City Token
def strat4_keys(rec):
    c = rec['country']
    pin = extract_pin(rec['address'])
    h_num = extract_house_num(rec['address'])
    addr_words = clean_addr(rec['address'])
    keys = []
    if pin and h_num:
        keys.append(f"{c}_pin_{pin}_h_{h_num}")
    elif pin:
        keys.append(f"{c}_pin_{pin}")
    if h_num and len(addr_words) >= 2:
        keys.append(f"{c}_h_{h_num}_w_{addr_words[-1]}")
    return keys

# STRATEGY 5: Multi-Pass Union (Country-Aware Multi-Pass Blocking)
# Pass 1: Exact Core Name
# Pass 2: First 2 Name Tokens
# Pass 3: Significant Name Token (length >= 4)
# Pass 4: Address PIN + House Num
# Pass 5: Character 3-grams of core name (for typos / transliterations)
def strat5_keys(rec):
    c = rec['country']
    name_words = clean_name(rec['name'])
    keys = []
    
    # 1. Full cleaned name
    if name_words:
        keys.append(f"{c}_fullname_{'_'.join(name_words)}")
        
        # 2. First 2 tokens if available
        if len(name_words) >= 2:
            keys.append(f"{c}_w2_{name_words[0]}_{name_words[1]}")
            
        # 3. Individual significant tokens (length >= 5 to prevent high frequency explosion)
        for w in name_words:
            if len(w) >= 5:
                keys.append(f"{c}_sigw_{w}")
                
    # 4. Address anchor (PIN + House number or Locality)
    pin = extract_pin(rec['address'])
    h_num = extract_house_num(rec['address'])
    if pin and h_num:
        keys.append(f"{c}_pinh_{pin}_{h_num}")
        
    return keys

# Run benchmarks
res1 = evaluate_blocking("1. Exact Normalized Core Name", strat1_keys, strat1_keys)
res2 = evaluate_blocking("2. First 2 Name Words Prefix", strat2_keys, strat2_keys)
res3 = evaluate_blocking("3. Significant Name Tokens (len>=4)", strat3_keys, strat3_keys)
res4 = evaluate_blocking("4. Address Anchor (PIN/House/Locality)", strat4_keys, strat4_keys)
res5 = evaluate_blocking("5. Multi-Pass Union (Name + Address Anchor)", strat5_keys, strat5_keys)
res5_top50 = evaluate_blocking("6. Multi-Pass Union (Capped at 50 cands)", strat5_keys, strat5_keys, max_candidates_per_s1=50)
res5_top30 = evaluate_blocking("7. Multi-Pass Union (Capped at 30 cands)", strat5_keys, strat5_keys, max_candidates_per_s1=30)
