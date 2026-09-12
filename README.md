# C10-team-marula
# Team Marula — Optimizing RAG Document Retrieval for Agronomic Advice

## 1. Dataset

We use the dataset provided through the Kaggle competition "Agricultural Extension RAG: Smart Retrieval for Farmers." It contains:

- 695 agricultural documents (avg. 66.1 words/doc, range 20–180 words)
- 308 training queries (avg. 9.4 words, range 6–16 words)
- 4,194 query–document relevance judgments (0–3 scale: 2,724 at 0, 728 at 1, 186 at 2, 556 at 3)
- 200 test queries (held out, no relevance labels)

Document fields: `document_id`, `title`, `text`, `source`, `crop`, `country`, `origin`, `source_url`, `license`.

The data is loaded directly from the Kaggle competition dataset (`documents.csv`, `train_queries.csv`, `qrels_train.csv`, `test_queries.csv`) — see `src/data_loading.py`. No additional scraping or manual labeling was performed. Some documents are marked as synthetic/CC0 in the source metadata.

## 2. Training Pipeline

We compare four retrieval approaches, in increasing order of complexity:

1. **TF-IDF baseline** (`src/tfidf_retrieval.py`) — title+body text vectorized with `TfidfVectorizer` (uni+bigrams, `min_df=2`, sublinear TF, English stopwords removed), cosine similarity via normalized dot product.
2. **BM25** (`src/bm25_retrieval.py`) — a from-scratch BM25 implementation (`k1=1.5`, `b=0.75`) with regex tokenization and title terms duplicated to upweight title matches.
3. **Dense retrieval** (`src/dense_retrieval.py`) — mean-pooled, L2-normalized sentence embeddings. Two variants are provided:
   - single-encoder dense retrieval (one transformer model),
   - a 3-model ensemble (MiniLM, BAAI/BGE, E5) whose cosine similarities are averaged.
4. **Hybrid retrieval** (`src/hybrid_retrieval.py`) — BM25 top-20 and dense-ensemble top-20 candidates combined via Reciprocal Rank Fusion (RRF, `alpha=60`).

No hyperparameter search beyond the values above was performed for this submission; BM25's `k1`/`b` and RRF's `alpha` use standard literature defaults. Model/design choices were driven by the nature of the mismatch described in the problem statement: farmers' plain-language queries vs. technical extension-document vocabulary, which lexical-only retrieval (TF-IDF/BM25) cannot always bridge.

## 3. Evaluation

All methods are evaluated with **nDCG@5** (Normalized Discounted Cumulative Gain at rank 5), the metric specified by the competition, computed against `qrels_train.csv` (`src/metrics.py`). Results on the training queries:

| Method | nDCG@5 |
|---|---|
| TF-IDF (baseline) | 0.5131 |
| Dense retrieval | 0.7856 |
| BM25 | see `scripts/run_bm25.py` output |
| Hybrid (BM25 + Dense ensemble, RRF) | see `scripts/run_hybrid_submission.py` output |

Evaluation is restricted to the retrieval/ranking stage — it does not assess the quality of any downstream generated (LLM) response.

## 4. Reproduction

### Setup

```bash
pip install -r requirements.txt
```

### Data

Point `src/config.py`'s `DATA_DIR` at the folder containing `documents.csv`, `train_queries.csv`, `qrels_train.csv`, `test_queries.csv`, `sample_submission.csv`. On Kaggle this is auto-detected under `/kaggle/input`; locally, download the competition data from Kaggle and set the path.

### Run, in order

```bash
# 1. TF-IDF baseline — sanity check + first submission.csv
python scripts/run_tfidf_baseline.py

# 2. BM25 — sparse retrieval, stronger than TF-IDF
python scripts/run_bm25.py

# 3. Dense ensemble — MiniLM + BGE + E5 averaged similarity
python scripts/run_dense_ensemble.py

# 4. Hybrid (final submission) — BM25 + Dense ensemble via RRF
python scripts/run_hybrid_submission.py
```

Each script prints its training-set nDCG@5 and writes `submission.csv` (or a named variant) in the working directory, in the exact `QueryId,DocumentId` format required by the competition.

The final Kaggle/CodaBench submission was produced by `scripts/run_hybrid_submission.py`. This matches `notebooks/team_marula_submission_first.ipynb`, which is included as the original development notebook.

## 5. Appendix — Contributors & Mentors

### Team Members (Team Marula)
- Obipehin Ridwanullah Adisa
- Patrick Ireoluwa
- Silas Emmanuel
- Simphiwe Paul Ndlovu
- Opateye Goodness Oluwamayokun

### Mentors
- Samuel Taiwo
- Seun Ajayi
