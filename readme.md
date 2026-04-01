# SDSC3002 Project: LSH-Accelerated Item-Based Collaborative Filtering

This repository implements a reproducible pipeline for accelerating item-item similarity search in a recommender system using MinHash + LSH.

## Project Scope

- Dataset: MovieLens 1M under `data/ml-1m/`
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
│   └── run_lsh.py
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

Important: the MovieLens 1M README states that redistribution requires separate permission. For public course deliverables, check the dataset license before re-uploading the raw files. A safer default is to publish the code, experiment outputs, and the official dataset download link.

## Run Order

### 1. Preprocess the dataset

```bash
python experiments/run_preprocess.py \
	--ratings data/ml-1m/ratings.dat \
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
	--baseline results/baseline/exact_topk.json
```

This produces:

- MinHash signature quality summary
- LSH candidate count and timing metrics
- approximate top-k neighbors
- recall@k against the exact baseline when `--baseline` is provided

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
- recall@10

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