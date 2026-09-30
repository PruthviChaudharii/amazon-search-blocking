import os
import sys
import csv
from collections import Counter, defaultdict

train_gt_path = "dataset/train/train_ground_truth.tsv"

print("Analyzing Ground Truth...")
total_s1 = 0
zero_matches = 0
one_match = 0
multi_matches = 0
s2_matches = 0
s3_matches = 0
match_count_dist = Counter()
same_source_multiple = Counter() # S2 multi, S3 multi, both

# Sample ground truth records for inspection
sample_gt = []

with open(train_gt_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for i, row in enumerate(f):
        total_s1 += 1
        s1_id, tab, rest = row.partition('\t')
        rest = rest.strip()
        if not rest:
            zero_matches += 1
            match_count_dist[0] += 1
            if len(sample_gt) < 5:
                sample_gt.append((s1_id, []))
            continue
        
        m_ids = rest.split(',')
        k = len(m_ids)
        match_count_dist[k] += 1
        if k == 1:
            one_match += 1
        else:
            multi_matches += 1
            
        c_s2 = sum(1 for mid in m_ids if mid.startswith('S2-'))
        c_s3 = sum(1 for mid in m_ids if mid.startswith('S3-'))
        s2_matches += c_s2
        s3_matches += c_s3
        
        if c_s2 > 1 or c_s3 > 1:
            same_source_multiple[(c_s2 > 1, c_s3 > 1)] += 1
            
        if len(sample_gt) < 15:
            sample_gt.append((s1_id, m_ids))

total_matches = s2_matches + s3_matches

print(f"Total S1 entities: {total_s1:,}")
print(f"Zero matches (singletons): {zero_matches:,} ({zero_matches/total_s1*100:.2f}%)")
print(f"Exactly 1 match: {one_match:,} ({one_match/total_s1*100:.2f}%)")
print(f"Multiple matches (>1): {multi_matches:,} ({multi_matches/total_s1*100:.2f}%)")
print(f"Total matches found: {total_matches:,}")
print(f"  - S2 matches: {s2_matches:,} ({s2_matches/total_matches*100:.2f}%)")
print(f"  - S3 matches: {s3_matches:,} ({s3_matches/total_matches*100:.2f}%)")
print(f"Average matches per S1: {total_matches/total_s1:.4f}")
print(f"Average matches per non-singleton S1: {total_matches/(total_s1 - zero_matches):.4f}")

print("\nMatch count distribution (matches per S1):")
for k in sorted(match_count_dist.keys())[:15]:
    print(f"  {k} matches: {match_count_dist[k]:,} ({match_count_dist[k]/total_s1*100:.3f}%)")

print("\nMultiple matches from the SAME source:")
for (s2_multi, s3_multi), count in same_source_multiple.items():
    print(f"  S2>1: {s2_multi}, S3>1: {s3_multi} -> {count:,} ({count/total_s1*100:.2f}%)")
