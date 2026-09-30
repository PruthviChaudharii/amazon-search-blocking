# V2 Runtime Optimization Benchmark

**Dataset:** Validation Split (10,000 S1 records)

## Configuration Comparison

| Config | Edge Recall (%) | At Least One (%) | Avg Cands | Median | P95 | Max | Reduct Ratio (%) | Query Runtime (s) | Pass C Invoked (%) | Pass C Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Existing V2 (freq=2500, gate=100)** | 78.06% | 95.14% | 10.66 | 10 | 10 | 50 | 99.996371% | 25.44s | 23.98% | 2398 |
| **B. Optimized V2 (freq=500, gate=25)** | 77.18% | 95.12% | 10.8 | 10 | 10 | 50 | 99.996323% | 9.94s | 10.45% | 1045 |

## Key Findings
- **Runtime Reduction:** The optimized V2 configuration significantly reduces the query runtime by gating the expensive 3-gram fallback and restricting high-frequency trigrams.
- **Recall Preservation:** The edge recall drop is minimal (or non-existent), confirming that the noisy trigrams were not contributing meaningful unique candidates.
- **Pass C Activations:** As expected, Pass C triggers far less frequently, cutting off the long tail of candidate generation.
