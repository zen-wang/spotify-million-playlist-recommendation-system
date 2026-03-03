# Spotify Million Playlist Dataset — PySpark Recommendation Pipeline

An end-to-end music recommendation system built with **Apache Spark/PySpark**, processing **1 million playlists** and **66 million playlist-track interactions** from Spotify's Million Playlist Dataset (RecSys Challenge 2018). The pipeline implements three recommendation models — popularity baseline, ALS collaborative filtering, and Word2Vec track embeddings — combined into a hybrid ensemble that achieves **R-Precision 0.18** and **NDCG 0.33**, a **287% improvement** over the popularity baseline.

## Highlights

- **Scale**: 33GB raw JSON → 1.8GB optimized Parquet (18× compression) across 1M playlists, 2.2M unique tracks, 295K artists
- **Best Model**: Hybrid ensemble (ALS + Word2Vec + Popularity) achieves R-Precision 0.1816 at 10 seed tracks
- **Spark vs Pandas**: Benchmarked identical operations — Spark completed a Join + GroupBy **5× faster** (15.9s vs 81.2s) while pandas consumed **9GB RAM** for the same operation
- **Full Evaluation**: Custom offline evaluation framework implementing R-Precision, NDCG, and Recommended Songs Clicks across 5 challenge scenarios (0 to 100 seed tracks)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Data Ingestion                                            │
│  1,000 JSON slices → Explicit Schema → Explode/Flatten → Parquet   │
│  Tables: playlists (1M) │ tracks (2.2M) │ playlist_tracks (66M)    │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2: EDA & Feature Engineering                                 │
│  Spark SQL queries │ Track/Artist popularity │ Co-occurrence (15M    │
│  pairs via broadcast self-join on top 10K tracks)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3: Modeling                                                  │
│  Popularity Baseline │ ALS (rank=64, implicit) │ Word2Vec (dim=128) │
│  Hybrid Ensemble: 0.5×ALS + 0.4×W2V + 0.1×Popularity              │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 4: Offline Evaluation                                        │
│  1,343 held-out playlists │ 5 scenarios │ R-Precision, NDCG, Clicks │
│  Vectorized NumPy scoring (aligned arrays, argpartition)            │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 5: Spark vs Pandas Benchmark                                 │
│  Load │ GroupBy │ Join+GroupBy │ Self-Join │ Memory comparison       │
└─────────────────────────────────────────────────────────────────────┘
```

## Results

### Model Evaluation (1,343 held-out playlists, 500 recommendations each)

| Scenario | Model | R-Precision | NDCG | Clicks |
|----------|-------|-------------|------|--------|
| Title only (0 seeds) | Popularity | 0.0503 | 0.1017 | 12.45 |
| First 5 tracks | **Hybrid** | **0.1795** | **0.3227** | **1.58** |
| First 10 tracks | **Hybrid** | **0.1816** | **0.3309** | **1.43** |
| First 25 tracks | **Hybrid** | **0.1727** | **0.3282** | **1.37** |
| First 100 tracks | **Hybrid** | **0.0820** | **0.2266** | **6.34** |

**Key findings:**
- Hybrid ensemble beats every individual model across all scenarios with seed tracks
- ALS is the strongest single model (collaborative patterns carry the most signal)
- Word2Vec has lower R-Precision but better Clicks scores — it surfaces relevant tracks near the top of the list, complementing ALS's broader recall
- For context, top AICrowd submissions achieved R-Precision ~0.22 using gradient boosting ensembles and graph neural networks

### Improvement Over Popularity Baseline

| Scenario | ALS | Word2Vec | Hybrid |
|----------|-----|----------|--------|
| First 5 | +252.6% | +135.1% | +270.1% |
| First 10 | +269.1% | +142.1% | +287.5% |
| First 25 | +288.4% | +145.1% | +304.9% |
| First 100 | +270.0% | +121.4% | +275.5% |

### Spark vs Pandas Benchmark (66M rows, local laptop)

| Operation | Spark | Pandas | Winner |
|-----------|-------|--------|--------|
| Load 66M rows from Parquet | 13.2s | 5.2s | Pandas |
| GroupBy (track popularity) | 5.8s | 5.4s | Pandas |
| Join + GroupBy (top artists) | 15.9s | 81.2s | **Spark (5×)** |
| Self-join co-occurrence (1%) | 24.1s | 21.7s | Pandas |
| Memory for 66M rows | ~2-4 GB | 3,290 MB | Spark |

**Takeaway:** Pandas wins on simple operations (less JVM overhead), but Spark dominates on complex operations. The Join + GroupBy shows why: pandas consumed **9GB RAM** for a single operation, while Spark streamed the data in partitions. At full dataset scale (33GB raw), pandas would crash on most machines.

## Tech Stack

- **Apache Spark / PySpark 3.5** — Distributed data processing, Spark SQL, MLlib
- **Spark MLlib ALS** — Implicit feedback collaborative filtering (rank=64, 10 iterations)
- **Spark MLlib Word2Vec** — Track embedding learning from playlist sequences (dim=128, window=5)
- **NumPy** — Vectorized model scoring and evaluation (aligned arrays, matrix multiplication, argpartition)
- **Parquet** — Columnar storage with Snappy compression (33GB JSON → 1.8GB Parquet)
- **Python** — Pipeline orchestration, metric computation

## PySpark Concepts Demonstrated

| Concept | Where Used | Why It Matters |
|---------|-----------|----------------|
| Explicit schema definition | Stage 1 (JSON ingestion) | Avoids schema inference overhead on 1,000 files |
| `explode()` for nested JSON | Stage 1 (flattening playlists→tracks) | Core technique for denormalizing nested structures |
| Lazy evaluation | Throughout | Spark builds execution plans without loading data until actions trigger |
| `cache()` / `unpersist()` | Stages 2-3 | Keeps hot DataFrames in memory; frees when done |
| Broadcast joins | Stage 2 (co-occurrence) | Sends small table to all workers, avoiding shuffle of 66M-row table |
| Window functions (`RANK() OVER`) | Stage 2 (popularity ranking) | Assigns ranks without collapsing rows |
| `left_anti` join | Stage 3 (train/val split) | Removes validation playlists from training data |
| `collect_list()` + ordering | Stage 3 (Word2Vec sequences) | Builds ordered track sequences per playlist |
| ALS implicit feedback | Stage 3 | Binary interaction data (in playlist or not), not explicit ratings |
| `coalesce()` for output control | Stage 1 (Parquet writes) | Controls number of output files |

## Project Structure

```
├── stage1_ingest.py          # JSON → Parquet pipeline
├── stage2_eda.py             # Spark SQL EDA + feature engineering
├── stage3_models.py          # Popularity, ALS, Word2Vec training
├── stage4_evaluate.py        # Offline evaluation (R-Prec, NDCG, Clicks)
├── stage5_benchmark.py       # Spark vs pandas benchmark
├── generate_submission.py    # AICrowd submission generator (10K × 500 recs)
├── output/
│   ├── eda/                  # EDA visualizations
│   ├── evaluation_results.txt
│   └── benchmark_results.txt
├── spotify_million_playlist_dataset/   # ← not included (request from AICrowd)
│   └── data/                 # 1,000 MPD JSON slices
└── spotify_million_playlist_dataset_challenge/  # ← not included
    └── challenge_set.json    # 10K incomplete playlists
```

## How to Run

### Prerequisites
- Python 3.10+
- Java 17 (required by Spark)
- ~6GB disk space for Parquet output

### Setup
```bash
# Install Java (macOS)
brew install openjdk@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyspark==3.5.4 numpy pandas pyarrow
```

### Run Pipeline
```bash
# Stage 1: Ingest JSON → Parquet (~15 min)
python stage1_ingest.py

# Stage 2: EDA + Feature Engineering (~15 min)
python stage2_eda.py

# Stage 3: Train Models (~30 min, ALS is the bottleneck)
python stage3_models.py

# Stage 4: Evaluate All Models (~5 min)
python stage4_evaluate.py

# Stage 5: Spark vs Pandas Benchmark (~10 min)
python stage5_benchmark.py
```

Total pipeline runtime: ~75 minutes on a laptop with 24GB RAM.

## Dataset

The [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) (MPD) was released for the RecSys Challenge 2018. Key statistics:

| Metric | Value |
|--------|-------|
| Playlists | 1,000,000 |
| Total track appearances | 66,346,428 |
| Unique tracks | 2,262,292 |
| Unique artists | 295,860 |
| Unique albums | 734,684 |
| Avg playlist length | 66.3 tracks |
| Most popular track | HUMBLE. by Kendrick Lamar (46,574 playlists) |
| Most popular artist | Drake (847,160 track appearances) |

## Design Decisions

**Why Spark over pandas?** The raw dataset is 33GB of JSON. While pandas can technically load the Parquet output (~1.8GB), complex operations like joins and co-occurrence computation consumed 9GB+ of RAM in benchmarks. Spark processes data in partitions, handles the full pipeline gracefully on a single laptop, and would scale linearly to a cluster — which is the production-relevant skill.

**Why ALS over deep learning?** Spark MLlib's ALS is purpose-built for large-scale implicit feedback collaborative filtering. It trains on 66M interactions in ~6 minutes on a single machine. Neural CF models (e.g., NCF, LightGCN) would require significant infrastructure for this dataset size and are harder to distribute. ALS was the right tool for demonstrating Spark-native ML at scale.

**Why a hybrid ensemble?** Each model captures different signals — ALS learns collaborative patterns ("playlists like yours also contain..."), Word2Vec captures sequential context ("tracks near each other in playlists are related"), and popularity provides a reliable cold-start fallback. Combining them with weighted scoring consistently outperforms any single model.

**Why offline evaluation instead of AICrowd?** The AICrowd grading infrastructure is no longer functional (all submissions since late 2024 show failed status). Building a custom evaluation framework that simulates all challenge scenarios was both more instructive and more portable as a demonstration of evaluation design skills.

## References

- Chen, C.W., Lamere, P., Schedl, M., & Zamani, H. (2018). *RecSys Challenge 2018: Automatic Music Playlist Continuation.* Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18).
- Spotify Million Playlist Dataset — [AICrowd Challenge Page](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

## License

This project uses the Spotify Million Playlist Dataset, subject to its [challenge rules and license](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/challenge_rules). Code in this repository is for educational and portfolio purposes.