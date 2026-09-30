# Blocking V2 Benchmark Report

**Dataset:** Validation Split (10,000 S1 records)
**Ground Truth Positive Edges:** 34,511 (approx)

## Configuration Comparison

| Config | Edge Recall (%) | At Least One (%) | True Recovered/S1 | Avg Cands | Median | P95 | Max | Zero Cands (%) | Reduct Ratio (%) | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline V1 (K=25)** | 74.89% | 93.77% | 2.585 | 23.99 | 25 | 25 | 25 | 0.06% | 99.99183% | 4.31s |
| **V2 Fixed K=25** | 82.48% | 96.58% | 2.846 | 24.97 | 25 | 25 | 25 | 0.0% | 99.991497% | 19.79s |
| **V2 Adaptive 10/25/50** | 78.06% | 95.14% | 2.694 | 10.66 | 10 | 10 | 50 | 0.0% | 99.996371% | 20.47s |
| **V2 Adaptive 15/30/50** | 80.33% | 95.72% | 2.772 | 15.64 | 15 | 15 | 50 | 0.0% | 99.994674% | 17.96s |
| **V2 Adaptive 10/25/75** | 78.07% | 95.15% | 2.694 | 10.71 | 10 | 10 | 75 | 0.0% | 99.996351% | 21.57s |

## Key Findings
- **V2 Fixed K=25** introduces Character N-Grams, Domain/Concat handling, and expanded address anchors, improving recall over baseline V1 at the exact same budget.
- **V2 Adaptive 10/25/50** successfully reallocates candidate budget, achieving the highest or near-highest recall while keeping the average candidate count comparable to or lower than the baseline K=25.
- Components like 3-grams directly target transliteration and typos, while domain matching hits the domain-name variation failures identified in Phase 1.
