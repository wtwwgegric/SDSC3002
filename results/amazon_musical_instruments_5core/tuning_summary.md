# Amazon LSH Tuning Summary

Dataset: Amazon Musical Instruments 5-core

Main finding:
- No tested LSH configuration beat the sparse exact baseline on full-data runtime.

Useful tuning conclusions:
- Fast but low-recall region: `r >= 3` is the only region that can be faster than the sparse exact baseline, but recall is extremely low (for example, `k60_b20_r3` gives recall@10 = 0.018 at 0.435s).
- Middle region: `r = 2` improves recall to roughly `0.18-0.45`, but runtime rises to `1.1-2.6s`, which is still slower than the exact baseline (`0.640s`).
- High-recall region: `r = 1` pushes recall to `0.74-0.96`, but runtime becomes `1.62-5.46s`, much slower than sparse exact.

Representative configurations:
- Best recall among tested configs: `k200_b200_r1`, recall@10 = 0.963, total time = 5.461s.
- Best high-recall tradeoff: `k20_b20_r1`, recall@10 = 0.742, total time = 1.620s.
- Best speedup over sparse exact: `k60_b20_r3`, total time = 0.435s, but recall@10 = 0.018.

Interpretation:
- On this dataset, improving recall requires dramatically increasing the number of LSH candidates.
- Once enough candidates are generated to recover most true neighbors, the LSH overhead dominates and the method loses its speed advantage.
- This is strong evidence that dataset sparsity, not just dataset size, determines whether LSH is worthwhile.