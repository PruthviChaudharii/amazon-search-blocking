import os
import sys
import csv
from collections import Counter
import re

def analyze_tsv(path, name, max_sample=500000):
    print(f"\n=================== Analyzing {name} ({path}) ===================")
    total_rows = 0
    missing_counts = Counter()
    countries = Counter()
    id_prefixes = Counter()
    id_patterns = Counter()
    
    name_lengths = []
    addr_lengths = []
    name_word_counts = []
    addr_word_counts = []
    
    names_sample = set()
    addr_sample = set()
    dup_names_in_sample = 0
    dup_addrs_in_sample = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        print(f"Header: {header}")
        
        for row in reader:
            total_rows += 1
            if len(row) < 4:
                # pad if missing
                row = row + [''] * (4 - len(row))
            
            eid, bname, baddr, ctry = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            
            # Missing checks
            if not eid: missing_counts['entity_id'] += 1
            if not bname: missing_counts['business_name'] += 1
            if not baddr: missing_counts['business_address'] += 1
            if not ctry: missing_counts['country'] += 1
            
            # Country
            countries[ctry] += 1
            
            # ID pattern
            if '-' in eid:
                prefix, _, suffix = eid.partition('-')
                id_prefixes[prefix] += 1
                if total_rows <= 5:
                    id_patterns[f"{prefix}-len{len(suffix)}"] += 1
            else:
                id_prefixes['NO_HYPHEN'] += 1
                
            # Sample stats for names and addresses
            if total_rows <= max_sample:
                name_lengths.append(len(bname))
                addr_lengths.append(len(baddr))
                n_words = len(bname.split())
                a_words = len(baddr.split())
                name_word_counts.append(n_words)
                addr_word_counts.append(a_words)
                
                if bname in names_sample:
                    dup_names_in_sample += 1
                else:
                    names_sample.add(bname)
                    
                if baddr in addr_sample:
                    dup_addrs_in_sample += 1
                else:
                    addr_sample.add(baddr)

    print(f"Total Rows: {total_rows:,}")
    print("Missing Values:")
    for col in header:
        cnt = missing_counts[col]
        print(f"  {col}: {cnt:,} ({cnt/total_rows*100:.4f}%)")
        
    print("Country Distribution:")
    for ctry, cnt in countries.most_common():
        print(f"  '{ctry}': {cnt:,} ({cnt/total_rows*100:.2f}%)")
        
    print("Entity ID Prefixes:")
    for pref, cnt in id_prefixes.most_common():
        print(f"  '{pref}': {cnt:,}")
        
    import numpy as np
    print(f"String Statistics (sampled up to {min(total_rows, max_sample):,} rows):")
    print(f"  Business Name Length (chars): mean={np.mean(name_lengths):.1f}, median={np.median(name_lengths):.0f}, p95={np.percentile(name_lengths, 95):.0f}, max={max(name_lengths)}")
    print(f"  Business Name Word Count:     mean={np.mean(name_word_counts):.1f}, median={np.median(name_word_counts):.0f}, p95={np.percentile(name_word_counts, 95):.0f}, max={max(name_word_counts)}")
    print(f"  Address Length (chars):       mean={np.mean(addr_lengths):.1f}, median={np.median(addr_lengths):.0f}, p95={np.percentile(addr_lengths, 95):.0f}, max={max(addr_lengths)}")
    print(f"  Address Word Count:           mean={np.mean(addr_word_counts):.1f}, median={np.median(addr_word_counts):.0f}, p95={np.percentile(addr_word_counts, 95):.0f}, max={max(addr_word_counts)}")
    n_sample = len(name_lengths)
    print(f"  Name Uniqueness in sample:    {len(names_sample):,}/{n_sample:,} ({len(names_sample)/n_sample*100:.2f}% unique)")
    print(f"  Address Uniqueness in sample: {len(addr_sample):,}/{n_sample:,} ({len(addr_sample)/n_sample*100:.2f}% unique)")

if __name__ == "__main__":
    analyze_tsv("dataset/train/train_source1.tsv", "Train Source 1")
    analyze_tsv("dataset/train/train_source2.tsv", "Train Source 2")
    analyze_tsv("dataset/train/train_source3.tsv", "Train Source 3")
    analyze_tsv("dataset/test/test_source1.tsv", "Test Source 1")
    analyze_tsv("dataset/test/test_source2.tsv", "Test Source 2")
    analyze_tsv("dataset/test/test_source3.tsv", "Test Source 3")
