# Figure Captions and Discussions

The following captions and discussion paragraphs are written in a report-ready style. Figure numbers can be adjusted to match the final ordering in the report.

## Figure 1. Speedup vs Subset Size

Caption:

Figure 1 shows the speedup of MinHash+LSH relative to the naive brute-force baseline and the sparse exact baseline as the subset size increases. The speedup ratio is defined as the runtime of the reference method divided by the runtime of LSH. A value greater than 1 indicates that LSH is faster, while a value smaller than 1 indicates that LSH is slower.

Discussion:

This figure is intended to highlight relative performance rather than absolute runtime. The curve labeled LSH speedup vs naive is computed as $T_{naive} / T_{LSH}$, and the curve labeled LSH speedup vs sparse-exact is computed as $T_{exact} / T_{LSH}$. As a result, the horizontal line at 1 serves as a direct threshold between improvement and regression. On MovieLens 1M, the speedup against the sparse exact baseline crosses 1 near the 900-1000 item range, which marks the crossover point where LSH begins to outperform the optimized exact method. On Amazon, the corresponding curve remains below 1 for all tested subset sizes, indicating that LSH never overtakes the sparse exact baseline under the tested configuration. Compared with the runtime plot, this figure makes the crossover behavior easier to read because it directly answers whether LSH is faster and by how much.

## Figure 2. LSH Quality vs Subset Size

Caption:

Figure 2 presents the retrieval quality and search budget of MinHash+LSH across subset size. Recall@k measures how many exact top-k neighbors are recovered by LSH, while candidate ratio measures the fraction of all possible item pairs that are generated as LSH candidates and passed to exact verification.

Discussion:

This figure summarizes the central accuracy-efficiency tradeoff of LSH. Recall@k is defined as $\sum_i |E_i \cap A_i| / \sum_i |E_i|$, where $E_i$ denotes the exact top-k neighbor set for item $i$ and $A_i$ denotes the approximate top-k set returned by LSH. Candidate ratio is defined as $|\mathcal{C}| / \binom{n}{2}$, where $|\mathcal{C}|$ is the number of candidate pairs produced by LSH and $\binom{n}{2}$ is the total number of possible item pairs in the subset. A higher recall indicates that LSH recovers more true neighbors, whereas a lower candidate ratio indicates a smaller search budget. The figure is therefore useful for judging whether a quality improvement is achieved efficiently or only by examining many more pairs. In the Amazon experiments, the candidate ratio remains extremely small but recall also remains low, which indicates that the tested LSH configuration prunes the search space too aggressively. More permissive banding improves recall, but only at the cost of substantially more candidate verification and longer runtime.

## Figure 3. MovieLens Pair-Space Reduction

Caption:

Figure 3 compares the full all-pairs search space, the sparse exact nonzero-pair ratio, and the LSH candidate ratio on MovieLens 1M. The y-axis uses a logarithmic scale to reveal how much of the theoretical pair space remains relevant after sparsity is taken into account and how much further it is reduced by LSH.

Discussion:

This figure explains why LSH eventually becomes competitive on MovieLens 1M. Although the sparse exact baseline avoids evaluating all theoretical item pairs, the proportion of item pairs with nonzero overlap remains relatively large on this dataset. In other words, even after sparsity is exploited, the exact method still needs to process a substantial number of meaningful pairs. By contrast, LSH reduces the candidate set much more aggressively, which leads to a visible reduction in verification work. As the subset size grows, this candidate reduction becomes large enough to offset the fixed overhead of signature generation and bucket construction, producing the runtime crossover observed in the speedup plot.

## Figure 4. Amazon Pair-Space Reduction

Caption:

Figure 4 compares the full all-pairs search space, the sparse exact nonzero-pair ratio, and the LSH candidate ratio on Amazon Musical Instruments 5-core. The logarithmic y-axis emphasizes the large gap between the theoretical pair space, the truly overlapping pairs, and the final LSH candidate set.

Discussion:

This figure explains why the sparse exact baseline remains unusually strong on Amazon. The nonzero-pair ratio is already very small because the dataset is extremely sparse, which means that exact sparse computation only needs to process a small fraction of the theoretical pair space. LSH does reduce the candidate space even further, but the additional reduction is not large enough to compensate for the overhead of signature computation, bucket formation, and candidate verification when reasonable recall is desired. As a result, LSH remains faster than naive brute-force search, but it does not surpass the sparse exact baseline.

## Figure 5. Pair-Space Reduction Comparison: MovieLens 1M vs Amazon

Caption:

Figure 5 places the pair-space reduction curves of MovieLens 1M and Amazon Musical Instruments 5-core side by side. Each panel shows the all-pairs baseline, the sparse exact nonzero-pair ratio, and the LSH candidate ratio on a logarithmic scale.

Discussion:

This comparison figure provides the most direct visual explanation of why the two datasets produce different crossover behavior. On MovieLens 1M, the nonzero-pair ratio remains relatively high, so the sparse exact method still has to handle a large set of truly overlapping pairs. LSH therefore provides a meaningful additional reduction in search space, which eventually leads to a runtime advantage. On Amazon, however, the nonzero-pair ratio is already very low, so sparsity alone removes most of the theoretical search burden. In that setting, LSH has much less room to improve upon the exact sparse method. This is why MovieLens exhibits a crossover while Amazon does not, even though both datasets show the expected growth in naive brute-force cost as subset size increases.