"""
Validation script for V2 candidate pairs
"""
import csv
import sys
import os
import numpy as np
from collections import Counter

csv.field_size_limit(sys.maxsize)

def validate_and_report():
    input_file = "output/candidate_pairs_v2.tsv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    # Load expected S1 IDs and targets
    s1_ids = set()
    s1_countries = {}
    with open("dataset/test/test_source1.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1_ids.add(row[0].strip())
            s1_countries[row[0].strip()] = row[3].strip() if len(row) > 3 else ''
            
    valid_targets = set()
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        with open(f"dataset/test/{fname}", 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                valid_targets.add(row[0].strip())
                
    # Validation flags
    is_valid = True
    errors = []
    
    # Stats
    seen_s1 = set()
    cand_lengths = []
    zero_cand_count = 0
    s2_count = 0
    s3_count = 0
    country_counts = Counter()
    total_pairs = 0
    
    # Read candidates
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append("Invalid header.")
            is_valid = False
            
        for i, row in enumerate(reader, start=2):
            if len(row) < 1 or len(row) > 2:
                errors.append(f"Row {i}: Malformed row.")
                is_valid = False
                continue
                
            sid = row[0].strip()
            if sid in seen_s1:
                errors.append(f"Row {i}: Duplicate S1 ID {sid}.")
                is_valid = False
            seen_s1.add(sid)
            
            if sid not in s1_ids:
                errors.append(f"Row {i}: Unknown S1 ID {sid}.")
                is_valid = False
                
            cands_str = row[1].strip() if len(row) > 1 else ""
            cands = [c.strip() for c in cands_str.split(',')] if cands_str else []
            
            c_len = len(cands)
            cand_lengths.append(c_len)
            total_pairs += c_len
            
            country = s1_countries.get(sid, 'Unknown')
            country_counts[country] += c_len
            
            if c_len == 0:
                zero_cand_count += 1
            else:
                if len(set(cands)) != c_len:
                    errors.append(f"Row {i}: Duplicate candidate IDs within row.")
                    is_valid = False
                    
                for cid in cands:
                    if cid not in valid_targets:
                        errors.append(f"Row {i}: Invalid target ID {cid}.")
                        is_valid = False
                    if cid.startswith('S2-'):
                        s2_count += 1
                    elif cid.startswith('S3-'):
                        s3_count += 1
                    else:
                        errors.append(f"Row {i}: Target ID {cid} has unknown prefix.")
                        is_valid = False
                        
    missing_s1 = s1_ids - seen_s1
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} Source 1 records.")
        is_valid = False

    print("=== VALIDATION ===")
    print(f"Status: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print("Errors (up to 10):")
        for e in errors[:10]:
            print(f" - {e}")
            
    print("\n=== STATISTICS ===")
    print(f"Total S1 Entities: {len(seen_s1):,}")
    print(f"Total Candidate Pairs: {total_pairs:,}")
    print(f"Avg Candidates/S1: {np.mean(cand_lengths):.2f}")
    print(f"Median: {int(np.median(cand_lengths))}")
    print(f"P90: {int(np.percentile(cand_lengths, 90))}")
    print(f"P95: {int(np.percentile(cand_lengths, 95))}")
    print(f"P99: {int(np.percentile(cand_lengths, 99))}")
    print(f"Max: {max(cand_lengths)}")
    print(f"Zero-Candidate Entities: {zero_cand_count:,} ({(zero_cand_count/len(seen_s1))*100:.2f}%)")
    print(f"S2 Candidates: {s2_count:,}")
    print(f"S3 Candidates: {s3_count:,}")
    print("\nCountry Breakdown:")
    for c, cnt in country_counts.most_common():
        print(f" - {c}: {cnt:,}")

    # Generate Report
    md = f"""# V2 Full Test Validation Report

## 1. Validation Status
**Result: {'PASS' if is_valid else 'FAIL'}**

**Validation Checks:**
- Exactly one row per test S1 entity: {'Pass' if not missing_s1 and len(seen_s1) == len(s1_ids) else 'Fail'}
- No duplicate S1 rows: {'Pass' if len(seen_s1) == len(cand_lengths) else 'Fail'}
- All candidate IDs belong to test S2/S3: {'Pass' if is_valid else 'Check logs'}
- No malformed rows: {'Pass' if is_valid else 'Check logs'}

## 2. Full Dataset Statistics
- **Total Candidate Pairs:** {total_pairs:,}
- **Average Candidates / S1:** {np.mean(cand_lengths):.2f}
- **Median Candidates / S1:** {int(np.median(cand_lengths))}
- **P90 Candidates / S1:** {int(np.percentile(cand_lengths, 90))}
- **P95 Candidates / S1:** {int(np.percentile(cand_lengths, 95))}
- **P99 Candidates / S1:** {int(np.percentile(cand_lengths, 99))}
- **Max Candidates / S1:** {max(cand_lengths)}
- **Zero-Candidate Entities:** {zero_cand_count:,} ({(zero_cand_count/len(seen_s1))*100:.2f}%)

## 3. Source Breakdown
- **Source 2 Targets:** {s2_count:,}
- **Source 3 Targets:** {s3_count:,}

## 4. Country Breakdown
"""
    for c, cnt in country_counts.most_common():
        md += f"- **{c}**: {cnt:,}\n"
        
    with open("reports/v2_full_test_validation.md", "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    validate_and_report()
