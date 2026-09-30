# Final Candidate Profile Report (`output/candidate_pairs.tsv`)

**File Path:** `output/candidate_pairs.tsv`  
**File Size:** 534.37 MB  
**Total Source 1 Rows:** 1,732,544  
**Total Generated Candidate Pairs:** 41,738,875  
**Candidate Generation Strategy:** Multi-Pass Country-Partitioned Inverted Index with Evidence-Based Scoring  
**Selected K:** K=25 (Pareto optimal)  

---

## 1. Candidate Distribution Metrics

| Metric | Measured Value |
| :--- | :--- |
| **Total Source 1 Entities** | **1,732,544** |
| **Total Candidate Pairs** | **41,738,875** |
| **Mean Candidates / S1** | **24.09** |
| **Median Candidates / S1** | **25** |
| **90th Percentile (P90)** | **25** |
| **95th Percentile (P95)** | **25** |
| **99th Percentile (P99)** | **25** |
| **Maximum Candidates / S1** | **25** |
| **Zero-Candidate Entities** | **829 (0.048%)** |
| **Reduction Ratio (Country-Partitioned)** | **99.999379%** |
| **Reduction Ratio (Global Search Space)** | **99.999758%** |

---

## 2. Partition Breakdown

### By Country:
- **India:** 19,752,116 candidate pairs across 809,986 S1 entities (avg **24.39** cands/S1)
- **US:** 16,009,747 candidate pairs across 663,106 S1 entities (avg **24.14** cands/S1)
- **France:** 5,977,012 candidate pairs across 259,452 S1 entities (avg **23.04** cands/S1)

### By Target Source:
- **Source 2 (`S2-`):** 33,459,684 (80.16%)
- **Source 3 (`S3-`):** 8,279,191 (19.84%)

---

## 3. Official Validator Verification

```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1,732,544
  valid S2/S3 match IDs: 9,969,589
  candidate_pairs.tsv: 1,732,544 rows (829 empty, 1,731,715 non-empty).
PASS — no blocking issues found. Safe to submit.
```

- **ID Existence Check:** 100% verified across all 9,969,589 test target IDs (`--check-ids` PASS).
- **Format Integrity:** Strict tab-separation, comma-separated candidate IDs, zero self-matches, zero duplicate rows, zero duplicate candidate IDs within lists.
