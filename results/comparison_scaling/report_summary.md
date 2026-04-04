# Three-tier Comparison Summary

Dataset: MovieLens 1M binary-preference artifacts (`artifacts/ml1m_binary`)

Configuration:
- top-k = 10
- MinHash hashes = 100
- LSH bands = 50
- rows per band = 2
- seed = 42

## Main finding

For this configuration, the measured crossover point where LSH first becomes faster than the sparse-exact baseline is around the `900-1000` item range.

A stable wording for the report is:

> On MovieLens 1M, the runtime crossover between optimized exact search and LSH occurs at roughly 900-1000 items in the sampled subset. From 1200 items onward, LSH is consistently and clearly faster than the sparse-exact baseline.

## Key results

| subset size | naive (s) | sparse-exact (s) | LSH (s) | LSH recall@10 | LSH vs sparse-exact |
|---|---:|---:|---:|---:|---:|
| 800  | 0.464 | 0.152 | 0.168 | 0.447 | 0.90x |
| 900  | 0.507 | 0.197 | 0.184 | 0.428 | 1.07x |
| 1000 | 0.666 | 0.231 | 0.200 | 0.449 | 1.15x |
| 1200 | 1.053 | 0.343 | 0.273 | 0.479 | 1.26x |
| 1500 | 1.519 | 0.523 | 0.365 | 0.489 | 1.43x |
| 3000 | 6.491 | 2.099 | 0.975 | 0.565 | 2.15x |

Interpretation:
- Naive brute-force becomes quickly impractical as subset size grows.
- Sparse-exact is a very strong engineering baseline on sparse data.
- LSH does not dominate at very small scale because signature generation and verification have fixed overhead.
- Once the subset is large enough, LSH reduces total work enough to surpass sparse-exact.
- The speed gain is paid for by approximate retrieval quality: recall@10 rises from about 0.34 to about 0.57 over the tested range, but remains well below exact search.

## Recommended figures for the report

1. `figures/runtime_vs_subset_size.png`
   - Best main figure.
   - Shows the three curves directly and marks the crossover.

2. `figures/speedup_vs_subset_size.png`
   - Best supporting figure.
   - Shows when the LSH speedup ratio crosses 1.0.

3. `figures/lsh_quality_vs_subset_size.png`
   - Best tradeoff figure.
   - Shows recall@k together with candidate ratio as subset size grows.

## Suggested report wording

A concise result paragraph:

> The naive brute-force baseline scales quadratically and becomes inefficient quickly. The sparse-matrix exact baseline is substantially stronger and remains competitive on moderate-size subsets. LSH only begins to outperform this optimized exact method when the subset size reaches roughly 900-1000 items, and from 1200 items onward the speed advantage becomes clear. At 3000 items, LSH is about 2.15x faster than sparse-exact and about 6.66x faster than naive brute-force, while achieving recall@10 of 0.565.
