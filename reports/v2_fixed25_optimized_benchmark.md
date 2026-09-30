# V2 Fixed K=25 Optimization Benchmark

**Dataset:** Validation Split (10,000 S1 records)

## Configuration Comparison

| Config | Edge Recall (%) | Entity Coverage (%) | Avg Cands | Median | P95 | Max | Index Build (s) | Query Runtime (s) | Pass C Invoked (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Fixed K=25 (freq=2500, gate=100)** | 82.48% | 62.89% | 24.97 | 25 | 25 | 25 | 13.93s | 22.54s | 0.0% |
| **B. Fixed K=25 (freq=500, gate=25)** | 82.37% | 62.60% | 24.92 | 25 | 25 | 25 | 14.42s | 11.62s | 0.0% |

## Decision
**RECOMMENDATION:** Optimized Fixed K=25 (freq=500, gate=25) retains >=81.5% recall while substantially lowering runtime. It is safe to promote for full-test generation.
