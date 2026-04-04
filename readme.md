# SDSC3002 Project: LSH-Accelerated Item-Based Collaborative Filtering

This repository implements a reproducible pipeline for accelerating item-item similarity search in a recommender system using MinHash + LSH.

## Project Scope

- Datasets: MovieLens 1M under `data/ml-1m/` and optional Amazon review subsets under `data/amazon/`
- Task: build top-k similar items for item-based collaborative filtering
- Exact metric: Jaccard similarity on binarized user preference sets
- Approximation: MinHash signatures + LSH banding + exact verification on candidates
- Main evaluation: total runtime, candidate count, recall@k against the exact baseline

Ratings are binarized with `rating >= 4` treated as a positive interaction.

## Repository Layout

```text
SDSC3002/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── baseline.py
│   └── lsh.py
├── experiments/
│   ├── run_preprocess.py
│   ├── run_baseline.py
│   ├── run_lsh.py
│   ├── run_sweep.py
│   └── plot_sweep.py
├── artifacts/
├── results/
├── data/
│   └── ml-1m/
├── data_explore.ipynb
├── demo.py
├── requirements.txt
└── README.md
```

## Environment Setup

Create a Python 3.10+ environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The code expects MovieLens 1M to exist at `data/ml-1m/ratings.dat`.

Optional extension dataset: Amazon Review Data 2018 category subsets in JSON.gz format. A practical next step is to use a 5-core category file such as `Video_Games_5.json.gz` or `Musical_Instruments_5.json.gz` under `data/amazon/`.

Important: the MovieLens 1M README states that redistribution requires separate permission. For public course deliverables, check the dataset license before re-uploading the raw files. A safer default is to publish the code, experiment outputs, and the official dataset download link.

## Run Order

### 1. Preprocess the dataset

```bash
python experiments/run_preprocess.py \
	--ratings data/ml-1m/ratings.dat \
	--dataset movielens \
	--artifacts-dir artifacts/ml1m_binary
```

This produces:

- `artifacts/ml1m_binary/item_user_matrix.npz`
- `artifacts/ml1m_binary/metadata.json`

### 2. Build the exact baseline

```bash
python experiments/run_baseline.py \
	--artifacts-dir artifacts/ml1m_binary \
	--results-dir results/baseline \
	--top-k 10
```

This computes exact item-item Jaccard similarity and saves top-k neighbors plus timing metrics.

Implementation note: the current exact baseline computes all-pairs shared-user counts with sparse matrix multiplication, so it is still exact, but much faster than a naive nested Python loop. This is useful for correctness and recall evaluation, but it also means LSH may not beat the exact baseline on MovieLens 1M unless you switch to a larger dataset or compare against a simpler brute-force implementation.

### 3. Run the LSH pipeline

```bash
python experiments/run_lsh.py \
	--artifacts-dir artifacts/ml1m_binary \
	--results-dir results/lsh_k120_b30_r4 \
	--num-hashes 120 \
	--bands 30 \
	--rows-per-band 4 \
	--top-k 10 \
	--verify-batch-size 2048 \
	--baseline results/baseline/exact_topk.json
```

This produces:

- MinHash signature quality summary
- LSH candidate count and timing metrics
- approximate top-k neighbors
- recall@k against the exact baseline when `--baseline` is provided

Verification note: candidate checking now uses batched sparse dot products grouped by source item. This reduces Python overhead substantially when LSH produces many candidates.

## Amazon Subset Extension

Recommended categories for a larger-but-still-manageable experiment:

- `Musical_Instruments_5.json.gz`: medium-sized 5-core subset
- `Video_Games_5.json.gz`: larger 5-core subset with clearer scale effect

Place the downloaded file under `data/amazon/`, then preprocess it with:

```bash
python experiments/run_preprocess.py \
	--ratings data/amazon/Video_Games_5.json.gz \
	--dataset amazon \
	--artifacts-dir artifacts/amazon_video_games_5core
```

Then run the baseline and LSH pipeline on the Amazon artifacts exactly as before:

```bash
python experiments/run_baseline.py \
	--artifacts-dir artifacts/amazon_video_games_5core \
	--results-dir results/amazon_video_games_5core/baseline \
	--top-k 10

python experiments/run_lsh.py \
	--artifacts-dir artifacts/amazon_video_games_5core \
	--results-dir results/amazon_video_games_5core/lsh_k120_b30_r4 \
	--num-hashes 120 \
	--bands 30 \
	--rows-per-band 4 \
	--top-k 10 \
	--verify-batch-size 2048 \
	--baseline results/amazon_video_games_5core/baseline/exact_topk.json
```

If the exact baseline becomes too slow on the Amazon subset, use the previously validated MovieLens exact baseline methodology for correctness on a smaller subset and position Amazon as the scale experiment for runtime and candidate-growth behavior.

## Suggested Experiment Grid

Use the same preprocessing artifacts and sweep the following parameter groups:

- `num_hashes = 80, 120, 160`
- `bands x rows = 20 x 6, 30 x 4, 60 x 2`
- `top_k = 10`

Recommended interpretation of these settings:

- `20 x 6`: very strict candidate generation, usually fast but recall can collapse.
- `30 x 4`: balanced setting for an initial report figure.
- `60 x 2`: much higher recall, but candidate verification can become slower than the exact baseline on MovieLens 1M.

For each run, record:

- signature generation time
- candidate generation time
- verification time
- total runtime
- candidate count
- candidate ratio
- recall@10

### 4. Run an automated parameter sweep

```bash
python experiments/run_sweep.py \
	--artifacts-dir artifacts/ml1m_binary \
	--results-dir results/sweeps/default_grid \
	--baseline results/baseline/exact_topk.json \
	--baseline-metrics results/baseline/baseline_metrics.json \
	--num-hashes 100 200 \
	--bands 10 20 30 40 50 60
```

The sweep runner automatically skips invalid combinations where `bands` does not divide `num_hashes` exactly.

This produces:

- `sweep_summary.csv`
- `sweep_summary.json`
- per-run metrics under `results/sweeps/default_grid/runs/`

### 5. Plot the sweep results

```bash
python experiments/plot_sweep.py \
	--summary-csv results/sweeps/default_grid/sweep_summary.csv \
	--output-dir results/sweeps/default_grid/plots
```

This produces three report-ready figures:

- `recall_vs_time.png`
- `recall_vs_candidate_ratio.png`
- `verification_vs_candidates.png`

## Main Output Files

- `metadata.json`: dataset statistics and index-to-ID mappings
- `exact_topk.json`: exact baseline neighbors
- `approx_topk.json`: approximate LSH neighbors
- `baseline_metrics.json` and `lsh_metrics.json`: timing and evaluation summaries

## Report Mapping

1. Group members and contributions: fill manually based on your team split.
2. Dataset and task: cite MovieLens 1M and describe item-based CF similarity search.
3. Metrics: use runtime, candidate count, recall@k.
4. Technique and expected benefit: describe MinHash, LSH banding, and reduced pairwise verification.
5. Experimental results: compare exact baseline versus LSH under multiple parameters.
6. Additional technical details: discuss hash count, bucket distribution, memory/runtime trade-offs.
7. Public URL: add your final repository or shared folder link here before submission.