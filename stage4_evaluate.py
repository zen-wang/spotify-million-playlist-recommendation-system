"""
Stage 4: Offline Evaluation (Optimized)
=========================================
Evaluates all models using R-Precision, NDCG, and Recommended Songs Clicks.

This version uses fully vectorized NumPy operations instead of Python dict
loops, making scoring ~100x faster.

Usage:
    python stage4_evaluate.py

Input:  ./output/parquet/*, ./output/models/*
Output: ./output/evaluation_results.txt
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import time
import os


# ===========================================================================
# 1. SPARK SESSION + LOAD EVERYTHING
# ===========================================================================

spark = (
    SparkSession.builder
    .appName("MPD-Stage4-Evaluate")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

PARQUET_DIR = "output/parquet"
MODEL_DIR = "output/models"

print("Loading data and models...")
t0 = time.time()

tracks = spark.read.parquet(f"{PARQUET_DIR}/tracks")
pop_baseline = spark.read.parquet(f"{PARQUET_DIR}/pop_baseline")
track_id_map = spark.read.parquet(f"{PARQUET_DIR}/track_id_map")
val_playlists = spark.read.parquet(f"{PARQUET_DIR}/validation_playlists")
val_tracks = spark.read.parquet(f"{PARQUET_DIR}/validation_tracks")
als_item_factors = spark.read.parquet(f"{PARQUET_DIR}/als_item_factors")
w2v_vectors = spark.read.parquet(f"{PARQUET_DIR}/w2v_vectors")

print(f"  Loaded in {time.time() - t0:.1f}s")


# ===========================================================================
# 2. BUILD UNIFIED INDEX
# ===========================================================================
# KEY OPTIMIZATION from the slow version:
#
# Before: Each model had separate dicts. Hybrid looped through 2.2M Python
#         dict entries per playlist -> billions of operations -> 5+ hours.
#
# Now:    ONE unified index. Every track gets a single integer index.
#         All scores are aligned NumPy arrays. Combining them is just:
#
#             hybrid = w1 * pop + w2 * als_scores + w3 * w2v_scores
#
#         That's ONE vectorized operation. NumPy runs this in C, so it
#         takes ~10ms instead of minutes per playlist.
# ===========================================================================

print("\nBuilding unified track index...")
t0 = time.time()

# Collect popularity data
pop_rows = pop_baseline.select("track_uri", "playlist_count").collect()
pop_dict = {row.track_uri: row.playlist_count for row in pop_rows}

# All unique track URIs = our candidate set
all_uris = sorted(pop_dict.keys())
uri_to_idx = {uri: i for i, uri in enumerate(all_uris)}
num_candidates = len(all_uris)
print(f"  {num_candidates:,} candidate tracks indexed")

# --- Popularity array (aligned) ---
pop_array = np.zeros(num_candidates, dtype=np.float32)
for uri, count in pop_dict.items():
    pop_array[uri_to_idx[uri]] = count
pop_array /= pop_array.max()  # normalize to 0-1
print(f"  Popularity array: done")

# --- ALS item factor matrix (aligned) ---
als_rows = (
    als_item_factors
    .join(track_id_map, als_item_factors.id == track_id_map.track_id)
    .select("track_uri", "features")
    .collect()
)
als_dim = len(als_rows[0].features)
als_matrix = np.zeros((num_candidates, als_dim), dtype=np.float32)
als_coverage = 0
for row in als_rows:
    idx = uri_to_idx.get(row.track_uri)
    if idx is not None:
        als_matrix[idx] = row.features
        als_coverage += 1
print(f"  ALS matrix: {als_coverage:,} tracks (dim={als_dim})")

# --- Word2Vec matrix (aligned, pre-normalized for cosine similarity) ---
w2v_rows = w2v_vectors.select("word", "vector").collect()
w2v_dim = len(w2v_rows[0].vector)
w2v_matrix = np.zeros((num_candidates, w2v_dim), dtype=np.float32)
w2v_coverage = 0
for row in w2v_rows:
    idx = uri_to_idx.get(row.word)
    if idx is not None:
        w2v_matrix[idx] = row.vector
        w2v_coverage += 1

# Pre-normalize rows for cosine similarity
w2v_norms = np.linalg.norm(w2v_matrix, axis=1, keepdims=True)
w2v_norms[w2v_norms == 0] = 1.0
w2v_matrix_normed = w2v_matrix / w2v_norms
print(f"  W2V matrix: {w2v_coverage:,} tracks (dim={w2v_dim})")

# --- Validation data ---
val_rows = val_tracks.orderBy("pid", "pos").collect()
val_data = {}
for row in val_rows:
    if row.pid not in val_data:
        val_data[row.pid] = []
    val_data[row.pid].append(row.track_uri)

val_name_rows = val_playlists.select("pid", "name").collect()
val_names = {row.pid: row.name for row in val_name_rows}

# Pre-compute index arrays for each validation playlist
val_data_indices = {}
for pid, track_list in val_data.items():
    indices = [uri_to_idx[uri] for uri in track_list if uri in uri_to_idx]
    val_data_indices[pid] = indices

# Done with Spark — everything is in NumPy now
spark.stop()
print(f"\nAll data prepared in {time.time() - t0:.1f}s")
print(f"  Validation playlists: {len(val_data)}")
print()


# ===========================================================================
# 3. METRICS
# ===========================================================================

NUM_RECS = 500


def r_precision(recommended_indices, holdout_set):
    """R-Precision: fraction of ground truth found in top-R recs."""
    R = len(holdout_set)
    if R == 0:
        return 0.0
    hits = sum(1 for idx in recommended_indices[:R] if idx in holdout_set)
    return hits / R


def ndcg(recommended_indices, holdout_set):
    """NDCG: rewards relevant items appearing earlier."""
    R = len(holdout_set)
    if R == 0:
        return 0.0
    dcg = 0.0
    for i, idx in enumerate(recommended_indices[:NUM_RECS]):
        if idx in holdout_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(R, NUM_RECS)))
    return dcg / idcg if idcg > 0 else 0.0


def clicks(recommended_indices, holdout_set):
    """Clicks: pages of 10 until first relevant track."""
    for i, idx in enumerate(recommended_indices[:NUM_RECS]):
        if idx in holdout_set:
            return i // 10
    return NUM_RECS // 10 + 1


# ===========================================================================
# 4. VECTORIZED RECOMMENDATION FUNCTIONS
# ===========================================================================
# Pattern:
#   1. Compute score array of shape (num_candidates,) via matrix multiply
#   2. Set seed scores to -inf (exclude from results)
#   3. np.argpartition to get top-500 (O(n), faster than full sort O(n log n))
#   4. Sort only those 500
# ===========================================================================

def get_top_n(scores, seed_indices, n=NUM_RECS):
    """Get top-n indices from score array, excluding seeds."""
    scores = scores.copy()
    if len(seed_indices) > 0:
        scores[seed_indices] = -np.inf

    if len(scores) <= n:
        return np.argsort(-scores)

    partitioned = np.argpartition(-scores, n)[:n]
    return partitioned[np.argsort(-scores[partitioned])]


def score_popularity(seed_indices):
    return get_top_n(pop_array, seed_indices)


def score_als(seed_indices):
    if len(seed_indices) == 0:
        return get_top_n(pop_array, seed_indices)

    seed_vecs = als_matrix[seed_indices]
    nonzero = np.any(seed_vecs != 0, axis=1)
    if not nonzero.any():
        return get_top_n(pop_array, seed_indices)

    user_vec = seed_vecs[nonzero].mean(axis=0)
    scores = als_matrix @ user_vec
    return get_top_n(scores, seed_indices)


def score_w2v(seed_indices):
    if len(seed_indices) == 0:
        return get_top_n(pop_array, seed_indices)

    seed_vecs = w2v_matrix[seed_indices]
    nonzero = np.any(seed_vecs != 0, axis=1)
    if not nonzero.any():
        return get_top_n(pop_array, seed_indices)

    user_vec = seed_vecs[nonzero].mean(axis=0)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm
    scores = w2v_matrix_normed @ user_vec
    return get_top_n(scores, seed_indices)


def score_hybrid(seed_indices, w_pop=0.1, w_als=0.5, w_w2v=0.4):
    """
    Hybrid: weighted sum of normalized scores.

    This is now ~3 matrix multiplies + array addition.
    Total: ~30ms per playlist instead of ~3 minutes.
    """
    hybrid = w_pop * pop_array.copy()

    if len(seed_indices) > 0:
        # ALS component
        seed_vecs_als = als_matrix[seed_indices]
        nonzero_als = np.any(seed_vecs_als != 0, axis=1)
        if nonzero_als.any():
            user_vec_als = seed_vecs_als[nonzero_als].mean(axis=0)
            als_scores = als_matrix @ user_vec_als
            als_min, als_max = als_scores.min(), als_scores.max()
            if als_max > als_min:
                als_scores = (als_scores - als_min) / (als_max - als_min)
            hybrid += w_als * als_scores

        # W2V component
        seed_vecs_w2v = w2v_matrix[seed_indices]
        nonzero_w2v = np.any(seed_vecs_w2v != 0, axis=1)
        if nonzero_w2v.any():
            user_vec_w2v = seed_vecs_w2v[nonzero_w2v].mean(axis=0)
            norm = np.linalg.norm(user_vec_w2v)
            if norm > 0:
                user_vec_w2v /= norm
            w2v_scores = w2v_matrix_normed @ user_vec_w2v
            w2v_scores = (w2v_scores + 1) / 2  # normalize -1..1 to 0..1
            hybrid += w_w2v * w2v_scores

    return get_top_n(hybrid, seed_indices)


# ===========================================================================
# 5. RUN EVALUATION
# ===========================================================================

print("=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

models = {
    "Popularity": score_popularity,
    "ALS": score_als,
    "Word2Vec": score_w2v,
    "Hybrid": score_hybrid,
}

scenarios = {
    "title_only (0)": 0,
    "first_5": 5,
    "first_10": 10,
    "first_25": 25,
    "first_100": 100,
}

results = {}

for scenario_name, num_seeds in scenarios.items():
    print(f"\n--- Scenario: {scenario_name} ({num_seeds} seed tracks) ---")
    results[scenario_name] = {}

    for model_name, score_fn in models.items():
        t_model = time.time()
        r_precs = []
        ndcgs = []
        clicks_list = []

        for pid, idx_list in val_data_indices.items():
            if num_seeds == 0:
                seed_idx = np.array([], dtype=int)
                holdout_idx = set(idx_list)
            else:
                seed_idx = np.array(idx_list[:num_seeds], dtype=int)
                holdout_idx = set(idx_list[num_seeds:])

            if len(holdout_idx) == 0:
                continue

            # 0-seed: everyone uses popularity
            if num_seeds == 0:
                rec_indices = get_top_n(pop_array, seed_idx)
            else:
                rec_indices = score_fn(seed_idx)

            r_precs.append(r_precision(rec_indices, holdout_idx))
            ndcgs.append(ndcg(rec_indices, holdout_idx))
            clicks_list.append(clicks(rec_indices, holdout_idx))

        avg_rprec = np.mean(r_precs) if r_precs else 0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0
        avg_clicks = np.mean(clicks_list) if clicks_list else 0

        results[scenario_name][model_name] = {
            "r_precision": avg_rprec,
            "ndcg": avg_ndcg,
            "clicks": avg_clicks,
            "evaluated": len(r_precs),
        }

        elapsed = time.time() - t_model
        print(f"  {model_name:12s} | R-Prec: {avg_rprec:.4f} | NDCG: {avg_ndcg:.4f} | "
              f"Clicks: {avg_clicks:.2f} | {elapsed:.1f}s")


# ===========================================================================
# 6. RESULTS SUMMARY
# ===========================================================================

print()
print("=" * 70)
print("EVALUATION RESULTS SUMMARY")
print("=" * 70)

print(f"\n{'Scenario':<18} {'Model':<12} {'R-Precision':>11} {'NDCG':>8} {'Clicks':>8}")
print("-" * 60)

for scenario_name in scenarios:
    for model_name in models:
        r = results[scenario_name][model_name]
        print(f"{scenario_name:<18} {model_name:<12} {r['r_precision']:>11.4f} "
              f"{r['ndcg']:>8.4f} {r['clicks']:>8.2f}")
    print()


# ===========================================================================
# 7. BEST MODEL PER SCENARIO
# ===========================================================================

print("=" * 70)
print("BEST MODEL PER SCENARIO")
print("=" * 70)

for scenario_name in scenarios:
    best_model = max(
        results[scenario_name].items(),
        key=lambda x: x[1]["r_precision"]
    )
    r = best_model[1]
    print(f"  {scenario_name:<18} -> {best_model[0]:<12} "
          f"(R-Prec: {r['r_precision']:.4f}, NDCG: {r['ndcg']:.4f})")


# ===========================================================================
# 8. IMPROVEMENT OVER BASELINE
# ===========================================================================

print()
print("=" * 70)
print("IMPROVEMENT OVER POPULARITY BASELINE")
print("=" * 70)

for scenario_name in scenarios:
    pop_rprec = results[scenario_name]["Popularity"]["r_precision"]
    for model_name in ["ALS", "Word2Vec", "Hybrid"]:
        model_rprec = results[scenario_name][model_name]["r_precision"]
        if pop_rprec > 0:
            improvement = ((model_rprec - pop_rprec) / pop_rprec) * 100
            print(f"  {scenario_name:<18} {model_name:<12} "
                  f"{'+' if improvement >= 0 else ''}{improvement:.1f}% vs popularity")
        else:
            print(f"  {scenario_name:<18} {model_name:<12} N/A (baseline = 0)")
    print()


# ===========================================================================
# 9. SAVE RESULTS
# ===========================================================================

output_path = "output/evaluation_results.txt"
with open(output_path, "w") as f:
    f.write("MPD Recommendation Pipeline - Evaluation Results\n")
    f.write(f"{'=' * 60}\n\n")
    f.write(f"Validation set: {len(val_data)} playlists\n")
    f.write(f"Recommendations per playlist: {NUM_RECS}\n\n")

    f.write(f"{'Scenario':<18} {'Model':<12} {'R-Precision':>11} {'NDCG':>8} {'Clicks':>8}\n")
    f.write(f"{'-' * 60}\n")

    for scenario_name in scenarios:
        for model_name in models:
            r = results[scenario_name][model_name]
            f.write(f"{scenario_name:<18} {model_name:<12} {r['r_precision']:>11.4f} "
                    f"{r['ndcg']:>8.4f} {r['clicks']:>8.2f}\n")
        f.write("\n")

    f.write(f"\nBest model per scenario:\n")
    for scenario_name in scenarios:
        best = max(results[scenario_name].items(), key=lambda x: x[1]["r_precision"])
        f.write(f"  {scenario_name:<18} -> {best[0]}\n")

print(f"\nResults saved to: {os.path.abspath(output_path)}")

print()
print("=" * 60)
print("STAGE 4 COMPLETE")
print("=" * 60)
print()
print("Key takeaways to look for:")
print("  1. More seed tracks -> better scores across all models")
print("  2. Hybrid should beat or tie individual models")
print("  3. 0-seed: all models = popularity (no personalization possible)")
print("  4. ALS and Word2Vec shine with 25+ seeds")
print()
print("For reference, top AICrowd submissions scored:")
print("  R-Precision: ~0.22, NDCG: ~0.39")
print()
print("Next: python stage5_benchmark.py")