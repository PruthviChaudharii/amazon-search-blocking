import os
import sys
import csv
import re
from collections import Counter, defaultdict
import numpy as np

# Set UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

print("Loading ground truth sample...")
s1_to_matches = {}
with open("dataset/train/train_ground_truth.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        if len(row) < 2: continue
        s1_id = row[0].strip()
        m_ids = [m.strip() for m in row[1].split(',') if m.strip()]
        if m_ids:
            s1_to_matches[s1_id] = m_ids
            if len(s1_to_matches) >= 30000:
                break

needed_s1 = set(s1_to_matches.keys())
needed_matches = set()
for mids in s1_to_matches.values():
    needed_matches.update(mids)

s1_data = {}
with open("dataset/train/train_source1.tsv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0].strip()
        if eid in needed_s1:
            s1_data[eid] = {
                'name': row[1].strip(),
                'address': row[2].strip(),
                'country': row[3].strip() if len(row) > 3 else ''
            }
            if len(s1_data) == len(needed_s1):
                break

target_data = {}
for s_file in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
    with open(s_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0].strip()
            if eid in needed_matches:
                target_data[eid] = {
                    'name': row[1].strip(),
                    'address': row[2].strip(),
                    'country': row[3].strip() if len(row) > 3 else ''
                }

def clean_tokens(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return set(s.split())

LEGAL_SUFFIXES = {'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation', 
                  'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'enterprises', 'associates'}

# Detailed noise and transformation detectors
legal_suffix_diffs = 0
word_order_swaps = 0
and_amp_diffs = 0
pure_typo_edits = 0
non_ascii_count = 0
examples_legal = []
examples_word_order = []
examples_typos = []
examples_weak_name = []
examples_weak_addr = []
examples_both_noisy = []

for s1_id, m_ids in s1_to_matches.items():
    if s1_id not in s1_data: continue
    s1 = s1_data[s1_id]
    s1_name_raw = s1['name']
    s1_addr_raw = s1['address']
    s1_tokens = clean_tokens(s1_name_raw)
    s1_addr_tokens = clean_tokens(s1_addr_raw)
    
    if any(ord(c) > 127 for c in s1_name_raw + s1_addr_raw):
        non_ascii_count += 1
        
    for mid in m_ids:
        if mid not in target_data: continue
        t = target_data[mid]
        t_name_raw = t['name']
        t_addr_raw = t['address']
        t_tokens = clean_tokens(t_name_raw)
        t_addr_tokens = clean_tokens(t_addr_raw)
        
        # Legal suffix differences
        s1_leg = s1_tokens & LEGAL_SUFFIXES
        t_leg = t_tokens & LEGAL_SUFFIXES
        if s1_leg != t_leg:
            legal_suffix_diffs += 1
            if len(examples_legal) < 5:
                examples_legal.append((s1_name_raw, t_name_raw))
                
        # & vs and
        if ('&' in s1_name_raw and 'and' in t_name_raw.lower()) or ('&' in t_name_raw and 'and' in s1_name_raw.lower()):
            and_amp_diffs += 1
            
        # Word order swaps (same tokens without order)
        s1_clean_list = [w for w in re.sub(r'[^a-z0-9\s]', ' ', s1_name_raw.lower()).split() if w not in LEGAL_SUFFIXES]
        t_clean_list = [w for w in re.sub(r'[^a-z0-9\s]', ' ', t_name_raw.lower()).split() if w not in LEGAL_SUFFIXES]
        if set(s1_clean_list) == set(t_clean_list) and s1_clean_list != t_clean_list and len(s1_clean_list) > 1:
            word_order_swaps += 1
            if len(examples_word_order) < 5:
                examples_word_order.append((s1_name_raw, t_name_raw))
                
        # Jaccards
        nj = len(s1_tokens & t_tokens) / max(1, len(s1_tokens | t_tokens))
        aj = len(s1_addr_tokens & t_addr_tokens) / max(1, len(s1_addr_tokens | t_addr_tokens))
        
        if nj < 0.3 and aj > 0.4 and len(examples_weak_name) < 5:
            examples_weak_name.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))
        elif nj > 0.8 and aj < 0.2 and len(examples_weak_addr) < 5:
            examples_weak_addr.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))
        elif nj < 0.5 and aj < 0.3 and len(examples_both_noisy) < 5:
            examples_both_noisy.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))

total_pairs = len(target_data)
print(f"Total matching pairs analyzed: {total_pairs:,}")
print(f"Pairs with Legal Suffix Mismatch: {legal_suffix_diffs:,} ({legal_suffix_diffs/total_pairs*100:.2f}%)")
print(f"Pairs with Word-Order Swaps: {word_order_swaps:,} ({word_order_swaps/total_pairs*100:.2f}%)")
print(f"Pairs with '&' vs 'and': {and_amp_diffs:,} ({and_amp_diffs/total_pairs*100:.2f}%)")
print(f"Records containing non-ASCII / Unicode characters: {non_ascii_count:,}")

print("\n--- SAMPLE LEGAL SUFFIX TRANSFORMATIONS ---")
for s1_n, tn in examples_legal:
    print(f"  S1: '{s1_n}'  <==>  Target: '{tn}'")

print("\n--- SAMPLE WORD ORDER SWAPS ---")
for s1_n, tn in examples_word_order:
    print(f"  S1: '{s1_n}'  <==>  Target: '{tn}'")

print("\n--- SAMPLE WEAK NAME, STRONG ADDRESS ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_weak_name:
    print(f"  Name: '{s1_n}' vs '{tn}' (Name Jaccard: {nj:.2f})")
    print(f"  Addr: '{s1_a}' vs '{ta}' (Addr Jaccard: {aj:.2f})")
    print("  " + "-"*35)

print("\n--- SAMPLE STRONG NAME, WEAK ADDRESS ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_weak_addr:
    print(f"  Name: '{s1_n}' vs '{tn}' (Name Jaccard: {nj:.2f})")
    print(f"  Addr: '{s1_a}' vs '{ta}' (Addr Jaccard: {aj:.2f})")
    print("  " + "-"*35)

print("\n--- SAMPLE BOTH NOISY ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_both_noisy:
    print(f"  Name: '{s1_n}' vs '{tn}' (Name Jaccard: {nj:.2f})")
    print(f"  Addr: '{s1_a}' vs '{ta}' (Addr Jaccard: {aj:.2f})")
    print("  " + "-"*35)
