"""
Generate AICrowd Submission
=============================
Loads the 10,000 challenge set playlists, runs the hybrid model on each,
and outputs a properly formatted submission CSV.

Usage:
    python generate_submission.py

Input:  ./spotify_million_playlist_dataset_challenge/challenge_set.json
        ./output/parquet/* (model artifacts from Stage 3)
Output: ./output/submission.csv
        ./output/submission.csv.gz (ready to upload)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import json
import gzip
import time
import os
import sys


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

CHALLENGE_SET_PATH = "spotify_million_playlist_dataset_challenge/challenge_set.json"
PARQUET_DIR = "output/parquet"
OUTPUT_CSV = "output/submission.csv"
OUTPUT_GZ = "output/submission.csv.gz"
NUM_RECS = 500

# Your team info for the submission header
TEAM_NAME = "wei_an_wang"
TEAM_EMAIL = "your_email@asu.edu"  # UPDATE THIS with your real email

if not os.path.exists(CHALLENGE_SET_PATH):
    print(f"ERROR: Challenge set not found at {CHALLENGE_SET_PATH}")
    print("Make sure you run this from the project root.")
    sys.exit(1)


# ===========================================================================
# 2. LOAD CHALLENGE SET
# ===========================================================================

print("Loading challenge set...")
t0 = time.time()

with open(CHALLENGE_SET_PATH, "r") as f:
    challenge = json.load(f)

challenge_playlists = challenge["playlists"]
print(f"  {len(challenge_playlists)} challenge playlists loaded in {time.time() - t0:.1f}s")

# Summarize the challenge scenarios
scenario_counts = {}
for pl in challenge_playlists:
    n_seeds = pl["num_samples"]
    has_name = "name" in pl
    key = f"{n_seeds} seeds, {'with' if has_name else 'no'} title"
    scenario_counts[key] = scenario_counts.get(key, 0) + 1

print("  Challenge scenarios:")
for key, count in sorted(scenario_counts.items()):
    print(f"    {key}: {count} playlists")
print()


# ===========================================================================
# 3. LOAD MODEL ARTIFACTS (same as Stage 4)
# ===========================================================================

print("Loading model artifacts via Spark...")
t0 = time.time()

spark = (
    SparkSession.builder
    .appName("MPD-Submission")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

pop_baseline = spark.read.parquet(f"{PARQUET_DIR}/pop_baseline")
track_id_map = spark.read.parquet(f"{PARQUET_DIR}/track_id_map")
als_item_factors = spark.read.parquet(f"{PARQUET_DIR}/als_item_factors")
w2v_vectors = spark.read.parquet(f"{PARQUET_DIR}/w2v_vectors")

print(f"  Loaded Spark tables in {time.time() - t0:.1f}s")


# ===========================================================================
# 4. BUILD UNIFIED NUMPY INDEX (same approach as optimized Stage 4)
# ===========================================================================

print("Building unified track index...")
t0 = time.time()

# Popularity
pop_rows = pop_baseline.select("track_uri", "playlist_count").collect()
pop_dict = {row.track_uri: row.playlist_count for row in pop_rows}

all_uris = sorted(pop_dict.keys())
uri_to_idx = {uri: i for i, uri in enumerate(all_uris)}
num_candidates = len(all_uris)

pop_array = np.zeros(num_candidates, dtype=np.float32)
for uri, count in pop_dict.items():
    pop_array[uri_to_idx[uri]] = count
pop_array /= pop_array.max()

# ALS
als_rows = (
    als_item_factors
    .join(track_id_map, als_item_factors.id == track_id_map.track_id)
    .select("track_uri", "features")
    .collect()
)
als_dim = len(als_rows[0].features)
als_matrix = np.zeros((num_candidates, als_dim), dtype=np.float32)
for row in als_rows:
    idx = uri_to_idx.get(row.track_uri)
    if idx is not None:
        als_matrix[idx] = row.features

# Word2Vec
w2v_rows = w2v_vectors.select("word", "vector").collect()
w2v_dim = len(w2v_rows[0].vector)
w2v_matrix = np.zeros((num_candidates, w2v_dim), dtype=np.float32)
for row in w2v_rows:
    idx = uri_to_idx.get(row.word)
    if idx is not None:
        w2v_matrix[idx] = row.vector

w2v_norms = np.linalg.norm(w2v_matrix, axis=1, keepdims=True)
w2v_norms[w2v_norms == 0] = 1.0
w2v_matrix_normed = w2v_matrix / w2v_norms

# Done with Spark
spark.stop()
print(f"  {num_candidates:,} tracks indexed in {time.time() - t0:.1f}s")
print()


# ===========================================================================
# 5. RECOMMENDATION FUNCTIONS
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


def recommend_hybrid(seed_uris, w_pop=0.1, w_als=0.5, w_w2v=0.4):
    """
    Hybrid recommendation returning track URIs.
    Handles all scenarios including 0-seed (title-only).
    """
    # Map seed URIs to indices
    seed_indices = np.array(
        [uri_to_idx[uri] for uri in seed_uris if uri in uri_to_idx],
        dtype=int
    )
    seed_set = set(seed_uris)  # for filtering, includes URIs not in our index

    # Start with popularity
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

        # Word2Vec component
        seed_vecs_w2v = w2v_matrix[seed_indices]
        nonzero_w2v = np.any(seed_vecs_w2v != 0, axis=1)
        if nonzero_w2v.any():
            user_vec_w2v = seed_vecs_w2v[nonzero_w2v].mean(axis=0)
            norm = np.linalg.norm(user_vec_w2v)
            if norm > 0:
                user_vec_w2v /= norm
            w2v_scores = w2v_matrix_normed @ user_vec_w2v
            w2v_scores = (w2v_scores + 1) / 2
            hybrid += w_w2v * w2v_scores

    top_indices = get_top_n(hybrid, seed_indices)

    # Convert indices back to URIs, excluding any seed URIs
    # (seed_indices only covers seeds found in our index;
    #  seed_set catches any seeds not in our training data)
    result = []
    for idx in top_indices:
        uri = all_uris[idx]
        if uri not in seed_set:
            result.append(uri)
        if len(result) >= NUM_RECS:
            break

    # If we don't have enough (rare edge case), pad with popularity
    if len(result) < NUM_RECS:
        pop_ranked = np.argsort(-pop_array)
        for idx in pop_ranked:
            uri = all_uris[idx]
            if uri not in seed_set and uri not in set(result):
                result.append(uri)
            if len(result) >= NUM_RECS:
                break

    return result[:NUM_RECS]


# ===========================================================================
# 6. GENERATE PREDICTIONS FOR ALL 10,000 CHALLENGE PLAYLISTS
# ===========================================================================

print("=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

submission_lines = []
# Header line
submission_lines.append(f"team_info,{TEAM_NAME},{TEAM_EMAIL}")

total = len(challenge_playlists)
t0 = time.time()

for i, pl in enumerate(challenge_playlists):
    pid = pl["pid"]
    seed_uris = [track["track_uri"] for track in pl["tracks"]]

    # Generate 500 recommendations
    recs = recommend_hybrid(seed_uris)

    # Format: pid, uri1, uri2, ..., uri500
    line = f"{pid}, " + ", ".join(recs)
    submission_lines.append(line)

    # Progress reporting
    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (total - i - 1) / rate
        print(f"  {i+1:,}/{total:,} playlists ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

elapsed = time.time() - t0
print(f"  Done! {total:,} playlists in {elapsed:.1f}s ({total/elapsed:.1f} playlists/sec)")
print()


# ===========================================================================
# 7. WRITE CSV AND GZIP
# ===========================================================================

print("Writing submission files...")

# Write plain CSV
with open(OUTPUT_CSV, "w") as f:
    for line in submission_lines:
        f.write(line + "\n")
csv_size = os.path.getsize(OUTPUT_CSV) / (1024**2)
print(f"  {OUTPUT_CSV}: {csv_size:.1f} MB")

# Gzip it
with open(OUTPUT_CSV, "rb") as f_in:
    with gzip.open(OUTPUT_GZ, "wb") as f_out:
        f_out.write(f_in.read())
gz_size = os.path.getsize(OUTPUT_GZ) / (1024**2)
print(f"  {OUTPUT_GZ}: {gz_size:.1f} MB")
print()


# ===========================================================================
# 8. VERIFY SUBMISSION FORMAT
# ===========================================================================

print("Verifying submission format...")

# Quick self-check (matching what verify_submission.py does)
errors = 0
seen_pids = set()
expected_pids = set(pl["pid"] for pl in challenge_playlists)

# Build seed tracks per playlist for checking
seed_tracks_map = {}
for pl in challenge_playlists:
    seed_tracks_map[pl["pid"]] = set(t["track_uri"] for t in pl["tracks"])

has_team_info = False
for line_no, line in enumerate(submission_lines):
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    if not has_team_info:
        if line.startswith("team_info"):
            has_team_info = True
        else:
            print(f"  ERROR: Missing team_info at line {line_no}")
            errors += 1
        continue

    fields = [f.strip() for f in line.split(",")]
    pid = int(fields[0])
    tracks = fields[1:]
    seen_pids.add(pid)

    if pid not in expected_pids:
        print(f"  ERROR: Bad pid {pid} at line {line_no}")
        errors += 1

    if len(tracks) != NUM_RECS:
        print(f"  ERROR: Wrong track count ({len(tracks)}) for pid {pid}")
        errors += 1

    if len(set(tracks)) != NUM_RECS:
        print(f"  ERROR: Duplicate tracks for pid {pid}")
        errors += 1

    # Check no seed tracks in submission
    overlap = seed_tracks_map.get(pid, set()) & set(tracks)
    if overlap:
        print(f"  ERROR: {len(overlap)} seed tracks found in submission for pid {pid}")
        errors += 1

if len(seen_pids) != len(expected_pids):
    print(f"  ERROR: Wrong playlist count ({len(seen_pids)}, expected {len(expected_pids)})")
    errors += 1

if errors == 0:
    print("  ✅ Submission format verified — ready to upload!")
else:
    print(f"  ❌ {errors} errors found — fix before submitting")

print()
print("=" * 60)
print("SUBMISSION READY")
print("=" * 60)
print()
print(f"Upload this file to AICrowd: {os.path.abspath(OUTPUT_GZ)}")
print()
print("Steps:")
print("  1. Go to https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge")
print("  2. Click 'Submissions' → 'New Submission'")
print(f"  3. Upload: {OUTPUT_GZ}")
print("  4. Cross your fingers 🤞")
print()
print(f"NOTE: Update TEAM_EMAIL in this script before submitting!")
print(f"      Currently set to: {TEAM_EMAIL}")
