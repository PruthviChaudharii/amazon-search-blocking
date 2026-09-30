import os
import sys
import csv
import re
from collections import Counter, defaultdict
import numpy as np

# Load ground truth sample of S1 and their matches
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
            if len(s1_to_matches) >= 50000:  # 50,000 ground truth positive pairs for deep pattern analysis
                break

needed_s1 = set(s1_to_matches.keys())
needed_matches = set()
for mids in s1_to_matches.values():
    needed_matches.update(mids)

print(f"Targeting {len(needed_s1)} S1 entities and {len(needed_matches)} matching S2/S3 entities...")

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
# Read S2
with open("dataset/train/train_source2.tsv", 'r', encoding='utf-8') as f:
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

# Read S3
with open("dataset/train/train_source3.tsv", 'r', encoding='utf-8') as f:
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

print(f"Loaded {len(s1_data)} S1 records and {len(target_data)} matched records. Now analyzing matching patterns...")

def normalize_text(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return ' '.join(s.split())

def extract_pin(s, country):
    if country == 'India':
        m = re.findall(r'\b[1-9][0-9]{5}\b', s)
        return m[0] if m else None
    elif country == 'US':
        m = re.findall(r'\b[0-9]{5}(?:-[0-9]{4})?\b', s)
        return m[0][:5] if m else None
    else:
        # Generic postal code
        m = re.findall(r'\b[0-9]{4,6}\b', s)
        return m[0] if m else None

def extract_numbers(s):
    return set(re.findall(r'\b[0-9]+\b', s))

def jaccard_sim(tokens1, tokens2):
    if not tokens1 or not tokens2: return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    return len(s1 & s2) / len(s1 | s2)

LEGAL_SUFFIXES = {'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation', 
                  'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc'}

# Pattern tracking
exact_name_match = 0
exact_norm_name_match = 0
name_jaccards = []
addr_jaccards = []
exact_pin_match = 0
pin_present_both = 0
pin_conflict = 0
number_overlap = []
country_match = 0
country_mismatch = 0
total_pairs = 0

weak_name_strong_addr = 0 # name jaccard < 0.3, addr jaccard > 0.5
strong_name_weak_addr = 0 # name jaccard > 0.8, addr jaccard < 0.2
both_noisy = 0            # name jaccard < 0.5, addr jaccard < 0.3
both_strong = 0           # name jaccard > 0.8, addr jaccard > 0.5

examples_weak_name_strong_addr = []
examples_strong_name_weak_addr = []
examples_both_noisy = []

for s1_id, m_ids in s1_to_matches.items():
    if s1_id not in s1_data: continue
    s1_rec = s1_data[s1_id]
    s1_name_raw = s1_rec['name']
    s1_addr_raw = s1_rec['address']
    s1_ctry = s1_rec['country']
    
    s1_name_norm = normalize_text(s1_name_raw)
    s1_addr_norm = normalize_text(s1_addr_raw)
    s1_name_tokens = s1_name_norm.split()
    s1_addr_tokens = s1_addr_norm.split()
    s1_pin = extract_pin(s1_addr_raw, s1_ctry)
    s1_nums = extract_numbers(s1_addr_raw)
    
    for mid in m_ids:
        if mid not in target_data: continue
        total_pairs += 1
        t_rec = target_data[mid]
        t_name_raw = t_rec['name']
        t_addr_raw = t_rec['address']
        t_ctry = t_rec['country']
        
        # Country
        if s1_ctry == t_ctry:
            country_match += 1
        else:
            country_mismatch += 1
            
        # Name matching
        if s1_name_raw == t_name_raw:
            exact_name_match += 1
        t_name_norm = normalize_text(t_name_raw)
        if s1_name_norm == t_name_norm:
            exact_norm_name_match += 1
            
        t_name_tokens = t_name_norm.split()
        nj = jaccard_sim(s1_name_tokens, t_name_tokens)
        name_jaccards.append(nj)
        
        # Address matching
        t_addr_norm = normalize_text(t_addr_raw)
        t_addr_tokens = t_addr_norm.split()
        aj = jaccard_sim(s1_addr_tokens, t_addr_tokens)
        addr_jaccards.append(aj)
        
        # PIN code
        t_pin = extract_pin(t_addr_raw, t_ctry)
        if s1_pin and t_pin:
            pin_present_both += 1
            if s1_pin == t_pin:
                exact_pin_match += 1
            else:
                pin_conflict += 1
                
        # House/Building numbers in address
        t_nums = extract_numbers(t_addr_raw)
        if s1_nums and t_nums:
            num_j = len(s1_nums & t_nums) / len(s1_nums | t_nums)
            number_overlap.append(num_j)
            
        # Cross category behavior
        if nj < 0.3 and aj > 0.4:
            weak_name_strong_addr += 1
            if len(examples_weak_name_strong_addr) < 5:
                examples_weak_name_strong_addr.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))
        elif nj > 0.8 and aj < 0.2:
            strong_name_weak_addr += 1
            if len(examples_strong_name_weak_addr) < 5:
                examples_strong_name_weak_addr.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))
        elif nj < 0.5 and aj < 0.3:
            both_noisy += 1
            if len(examples_both_noisy) < 5:
                examples_both_noisy.append((s1_name_raw, t_name_raw, s1_addr_raw, t_addr_raw, nj, aj))
        elif nj > 0.8 and aj > 0.5:
            both_strong += 1

print(f"\nAnalyzed {total_pairs:,} positive matching pairs.")
print(f"Country Consistency: {country_match:,} match ({country_match/total_pairs*100:.2f}%), {country_mismatch:,} mismatch ({country_mismatch/total_pairs*100:.2f}%)")
print(f"Exact Name Match (raw): {exact_name_match:,} ({exact_name_match/total_pairs*100:.2f}%)")
print(f"Exact Name Match (normalized): {exact_norm_name_match:,} ({exact_norm_name_match/total_pairs*100:.2f}%)")
print(f"Name Token Jaccard: mean={np.mean(name_jaccards):.4f}, median={np.median(name_jaccards):.4f}, >0.5: {sum(1 for x in name_jaccards if x>0.5)/total_pairs*100:.2f}%, >0.8: {sum(1 for x in name_jaccards if x>0.8)/total_pairs*100:.2f}%")
print(f"Address Token Jaccard: mean={np.mean(addr_jaccards):.4f}, median={np.median(addr_jaccards):.4f}, >0.3: {sum(1 for x in addr_jaccards if x>0.3)/total_pairs*100:.2f}%, >0.5: {sum(1 for x in addr_jaccards if x>0.5)/total_pairs*100:.2f}%")
if pin_present_both:
    print(f"PIN/Postal present in both: {pin_present_both:,} ({pin_present_both/total_pairs*100:.2f}%)")
    print(f"  - PIN exactly matches: {exact_pin_match:,} ({exact_pin_match/pin_present_both*100:.2f}%)")
    print(f"  - PIN conflicts: {pin_conflict:,} ({pin_conflict/pin_present_both*100:.2f}%)")
if number_overlap:
    print(f"House/Street Number Overlap (when both have numbers): mean Jaccard = {np.mean(number_overlap):.4f}")

print(f"\nRelationship Categories:")
print(f"  Both strong (Name Jaccard > 0.8 & Addr Jaccard > 0.5): {both_strong:,} ({both_strong/total_pairs*100:.2f}%)")
print(f"  Strong name, weak address (Name > 0.8 & Addr < 0.2): {strong_name_weak_addr:,} ({strong_name_weak_addr/total_pairs*100:.2f}%)")
print(f"  Weak name, strong address (Name < 0.3 & Addr > 0.4): {weak_name_strong_addr:,} ({weak_name_strong_addr/total_pairs*100:.2f}%)")
print(f"  Both noisy (Name < 0.5 & Addr < 0.3): {both_noisy:,} ({both_noisy/total_pairs*100:.2f}%)")

print("\n--- EXAMPLES OF WEAK NAME, STRONG ADDRESS ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_weak_name_strong_addr:
    print(f"S1 Name: '{s1_n}' | T Name: '{tn}' (NJ: {nj:.2f})")
    print(f"S1 Addr: '{s1_a}' | T Addr: '{ta}' (AJ: {aj:.2f})")
    print("-" * 40)

print("\n--- EXAMPLES OF STRONG NAME, WEAK ADDRESS ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_strong_name_weak_addr:
    print(f"S1 Name: '{s1_n}' | T Name: '{tn}' (NJ: {nj:.2f})")
    print(f"S1 Addr: '{s1_a}' | T Addr: '{ta}' (AJ: {aj:.2f})")
    print("-" * 40)

print("\n--- EXAMPLES OF BOTH NOISY ---")
for s1_n, tn, s1_a, ta, nj, aj in examples_both_noisy:
    print(f"S1 Name: '{s1_n}' | T Name: '{tn}' (NJ: {nj:.2f})")
    print(f"S1 Addr: '{s1_a}' | T Addr: '{ta}' (AJ: {aj:.2f})")
    print("-" * 40)
