"""
Stage 3: Recommendation Models
================================
Builds four recommender models of increasing sophistication:

    3A. Popularity Baseline     — recommend most popular tracks (simple but strong)
    3B. ALS Collaborative Filter — matrix factorization on playlist-track interactions
    3C. Word2Vec Embeddings     — treat playlists as "sentences" of track IDs
from pyspark.sql.window import Window
    3D. Hybrid Ensemble         — weighted combination of all three

Before modeling, we create a validation set by holding out tracks from
real playlists — simulating the challenge scenarios.

Usage:
    python stage3_models.py

Input:  ./output/parquet/{playlists, tracks, playlist_tracks, track_popularity}
Output: ./output/models/{als_model, word2vec_model}
        ./output/parquet/validation_set
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import Word2Vec
import time
import os


# ===========================================================================
# 1. SPARK SESSION + LOAD DATA
# ===========================================================================

spark = (
    SparkSession.builder
    .appName("MPD-Stage3-Models")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

PARQUET_DIR = "output/parquet"
MODEL_DIR = "output/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading tables...")
t0 = time.time()

playlist_tracks = spark.read.parquet(f"{PARQUET_DIR}/playlist_tracks")
tracks = spark.read.parquet(f"{PARQUET_DIR}/tracks")
track_popularity = spark.read.parquet(f"{PARQUET_DIR}/track_popularity")
playlists = spark.read.parquet(f"{PARQUET_DIR}/playlists")

playlist_tracks.cache()
playlist_tracks.count()

print(f"  Loaded in {time.time() - t0:.1f}s")
print()


# ===========================================================================
# 2. CREATE VALIDATION SET
# ===========================================================================
# To evaluate our models, we simulate the challenge:
#   - Pick 2,000 playlists (200 per scenario)
#   - For each, split into "seed" tracks (given) and "holdout" tracks (predict)
#
# We simulate 5 scenarios (the most distinct ones):
#   Scenario A: 0 seed tracks   (title only — hardest)
#   Scenario B: 5 seed tracks   (first 5)
#   Scenario C: 10 seed tracks  (first 10)
#   Scenario D: 25 seed tracks  (first 25)
#   Scenario E: 100 seed tracks (first 100, only playlists with 100+ tracks)
#
# We keep these playlists SEPARATE from training data so we don't cheat.
#
# NEW CONCEPT — sample() and randomSplit():
#   sample(fraction) returns a random subset of rows.
#   But for reproducibility we use a seed parameter so the same rows
#   are selected every time you run the script.
# ===========================================================================

print("=" * 60)
print("CREATING VALIDATION SET")
print("=" * 60)

# Step 1: Reserve 2,000 playlists for validation
# Only consider playlists with 100+ tracks (so all scenarios work)
long_playlists = playlists.filter(F.col("num_tracks") >= 100).select("pid", "name")
val_playlists = long_playlists.sample(fraction=0.006, seed=42).limit(2000)
val_pids = val_playlists.select("pid")
val_pids.cache()
val_pid_count = val_pids.count()
print(f"  Reserved {val_pid_count} playlists for validation")

# Step 2: Get all tracks for these playlists
val_tracks = (
    playlist_tracks
    .join(val_pids, "pid")
    .orderBy("pid", "pos")
)
val_tracks.cache()
val_tracks.count()

# Step 3: For each playlist, split into seed and holdout
# We'll store all tracks with a flag for whether they're seed or holdout
# For now, we create one split: first 5 tracks as seed, rest as holdout

scenarios = {
    "title_only": 0,
    "first_5": 5,
    "first_10": 10,
    "first_25": 25,
    "first_100": 100,
}

# Save the full validation playlist tracks for Stage 4
val_tracks.write.mode("overwrite").parquet(f"{PARQUET_DIR}/validation_tracks")
val_playlists.write.mode("overwrite").parquet(f"{PARQUET_DIR}/validation_playlists")
print(f"  Saved validation set to {PARQUET_DIR}/validation_tracks")

# Step 4: Remove validation playlists from training data
# This is critical — if val playlists stay in training, evaluation is meaningless
train_pt = playlist_tracks.join(val_pids, "pid", "left_anti")
train_pt.cache()
train_count = train_pt.count()
print(f"  Training set: {train_count:,} playlist-track pairs")
print(f"  (Removed {playlist_tracks.count() - train_count:,} validation pairs)")
print()


# ===========================================================================
# 3A. POPULARITY BASELINE
# ===========================================================================
# The simplest possible model: recommend the most popular tracks that
# aren't already in the playlist.
#
# Why start with this? Two reasons:
#   1. It's a surprisingly strong baseline — popular tracks are popular
#      for a reason. Many competition teams couldn't beat it by much.
#   2. You need a baseline to know if your fancy model is actually helping.
#      "My ALS model got R-Precision 0.15" means nothing without knowing
#      that popularity alone gets 0.10.
#
# Implementation: just sort track_popularity by rank. At recommendation time,
# filter out tracks already in the playlist.
# ===========================================================================

print("=" * 60)
print("MODEL 3A: POPULARITY BASELINE")
print("=" * 60)

from pyspark.sql.window import Window

# Recompute popularity from TRAINING data only (excluding validation playlists)
pop_baseline = (
    train_pt
    .groupBy("track_uri")
    .agg(F.count("*").alias("playlist_count"))
    .withColumn("pop_rank", F.rank().over(
        Window.orderBy(F.desc("playlist_count"))
    ))
)

# Need to import Window


# Recompute with proper import
pop_baseline = (
    train_pt
    .groupBy("track_uri")
    .agg(F.count("*").alias("playlist_count"))
    .withColumn("pop_rank", F.rank().over(
        Window.orderBy(F.desc("playlist_count"))
    ))
)

pop_baseline.cache()
pop_baseline.count()

print("Top 10 tracks (training set):")
(
    pop_baseline
    .join(tracks, "track_uri")
    .orderBy("pop_rank")
    .select("pop_rank", "track_name", "artist_name", "playlist_count")
    .limit(10)
    .show(truncate=False)
)

pop_baseline.write.mode("overwrite").parquet(f"{PARQUET_DIR}/pop_baseline")
print("  Saved popularity baseline.")
print()


# ===========================================================================
# 3B. ALS COLLABORATIVE FILTERING
# ===========================================================================
# ALS (Alternating Least Squares) is the go-to matrix factorization method
# in Spark MLlib. Here's the intuition:
#
# Imagine a giant matrix:
#   Rows    = playlists (1M)
#   Columns = tracks (2.2M)
#   Cell    = 1 if track is in playlist, 0 otherwise
#
# This matrix is 99.997% zeros (sparse!). ALS learns two smaller matrices:
#   - Playlist factors: each playlist gets a vector of size `rank`
#   - Track factors:    each track gets a vector of size `rank`
#
# The dot product of a playlist vector and track vector predicts how well
# that track fits that playlist. High score = good recommendation.
#
# KEY PARAMETERS:
#   rank       = dimension of latent factors (bigger = more expressive but slower)
#   regParam   = regularization to prevent overfitting
#   alpha      = confidence parameter for implicit feedback
#   maxIter    = training iterations
#   implicitPrefs = True because our data is binary (in playlist or not),
#                   not explicit ratings (1-5 stars)
#
# IMPORTANT GOTCHA — ALS needs INTEGER IDs, not string URIs.
# We need to map track_uri → integer index. Spark's StringIndexer can do
# this, but a simple dense mapping is more transparent.
# ===========================================================================

print("=" * 60)
print("MODEL 3B: ALS COLLABORATIVE FILTERING")
print("=" * 60)

# Step 1: Create integer ID mappings
# pid is already an integer (0 to 999,999) — perfect for ALS.
# track_uri needs mapping to integers.

print("Creating track ID mapping...")
t0 = time.time()

# Give each unique track a sequential integer ID
track_id_map = (
    train_pt
    .select("track_uri")
    .distinct()
    .withColumn("track_id", F.monotonically_increasing_id())
    # monotonically_increasing_id() gives unique IDs but they can be large
    # and non-contiguous. For ALS, we want dense IDs starting from 0.
)

# For dense IDs, use row_number (this requires a Window but gives 0,1,2,...)
track_id_map = (
    train_pt
    .select("track_uri")
    .distinct()
    .coalesce(1)  # single partition so row_number is globally sequential
    .withColumn("track_id",
        (F.row_number().over(Window.orderBy("track_uri")) - 1).cast(IntegerType())
    )
)

track_id_map.cache()
num_tracks = track_id_map.count()
print(f"  Mapped {num_tracks:,} unique tracks to integer IDs in {time.time() - t0:.1f}s")

# Save the mapping — we need it later to convert predictions back to URIs
track_id_map.write.mode("overwrite").parquet(f"{PARQUET_DIR}/track_id_map")

# Step 2: Prepare ALS input
# ALS expects: (user_id, item_id, rating)
# For us:      (pid,     track_id, 1.0)
# Every row has rating=1.0 (implicit: track IS in playlist)

print("Preparing ALS training data...")
t0 = time.time()

als_input = (
    train_pt
    .join(track_id_map, "track_uri")
    .select(
        F.col("pid").alias("user_id"),       # playlist = "user"
        F.col("track_id").alias("item_id"),  # track = "item"
    )
    .withColumn("rating", F.lit(1.0).cast(FloatType()))
)

als_input.cache()
als_input.count()
print(f"  Prepared {als_input.count():,} interactions in {time.time() - t0:.1f}s")

# Step 3: Train ALS model
# Starting with moderate hyperparameters. We can tune these in Stage 4.
#
# rank=64:     64-dimensional latent space. Good starting point.
#              Lower (16-32) is faster, higher (128-256) might capture more
#              nuance but risks overfitting and is much slower.
#
# regParam=0.1: Standard regularization. Prevents factors from growing huge.
#
# alpha=1.0:   Confidence scaling for implicit feedback. Higher alpha means
#              Spark trusts the 1's more (presence = strong signal).
#
# maxIter=10:  10 passes over the data. Usually converges enough.
#              More iterations = marginally better but slower.
#
# coldStartStrategy="drop": When predicting for unseen users/items,
#              return NaN instead of crashing. We handle cold start separately.

print("\nTraining ALS model (this will take several minutes)...")
print("  rank=64, regParam=0.1, alpha=1.0, maxIter=10")
t0 = time.time()

als = ALS(
    rank=64,
    maxIter=10,
    regParam=0.1,
    alpha=1.0,
    implicitPrefs=True,
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    seed=42,
)

als_model = als.fit(als_input)
print(f"  ALS training complete in {time.time() - t0:.1f}s")

# Save the model
als_model.write().overwrite().save(f"{MODEL_DIR}/als_model")
print(f"  Saved ALS model to {MODEL_DIR}/als_model")

# Quick sanity check: recommend tracks for a random training playlist
print("\nSanity check — top 10 ALS recommendations for pid=0:")
pid0_recs = (
    als_model
    .recommendForUserSubset(
        spark.createDataFrame([(0,)], ["user_id"]),
        10
    )
    .select(F.explode("recommendations").alias("rec"))
    .select(
        F.col("rec.item_id").alias("track_id"),
        F.col("rec.rating").alias("als_score"),
    )
    .join(track_id_map, "track_id")
    .join(tracks, "track_uri")
    .select("track_name", "artist_name", "als_score")
)
pid0_recs.show(truncate=False)
print()


# ===========================================================================
# 3C. WORD2VEC TRACK EMBEDDINGS
# ===========================================================================
# Here's a clever idea from NLP applied to music:
#
# In NLP, Word2Vec learns word meanings by looking at which words appear
# near each other in sentences. "King" and "Queen" get similar vectors
# because they appear in similar contexts.
#
# We do the same with playlists:
#   - A playlist is a "sentence"
#   - Each track is a "word"
#   - Tracks that appear in similar playlists get similar vectors
#
# This captures something different from ALS:
#   - ALS learns from which playlists contain which tracks
#   - Word2Vec also captures ORDERING (nearby tracks in a playlist
#     are more related than distant ones)
#
# After training, we can find "similar tracks" by comparing vectors.
# Given seed tracks in a playlist, we recommend tracks with the most
# similar embeddings.
#
# IMPLEMENTATION:
#   Spark MLlib's Word2Vec expects input as arrays of strings (one per
#   "sentence"). We collect track_uris per playlist, ordered by position.
# ===========================================================================

print("=" * 60)
print("MODEL 3C: WORD2VEC TRACK EMBEDDINGS")
print("=" * 60)

# Step 1: Build "sentences" — ordered track lists per playlist
print("Building playlist sequences...")
t0 = time.time()

playlist_sequences = (
    train_pt
    .orderBy("pid", "pos")
    .groupBy("pid")
    .agg(
        # collect_list preserves the order from orderBy
        F.collect_list("track_uri").alias("track_sequence")
    )
    # Filter out very short playlists (not enough context for Word2Vec)
    .filter(F.size("track_sequence") >= 5)
)

playlist_sequences.cache()
seq_count = playlist_sequences.count()
print(f"  {seq_count:,} playlist sequences in {time.time() - t0:.1f}s")

# Step 2: Train Word2Vec
# vectorSize=128:   dimension of track embeddings (similar to ALS rank)
# windowSize=5:     how many neighboring tracks define "context"
#                   (5 means: look at 5 tracks before and after)
# minCount=5:       ignore tracks that appear in fewer than 5 playlists
#                   (too rare to learn good embeddings)
# maxIter=5:        training passes over the data
# numPartitions=8:  parallelism for training

print("\nTraining Word2Vec model (this will take several minutes)...")
print("  vectorSize=128, windowSize=5, minCount=5, maxIter=5")
t0 = time.time()

word2vec = Word2Vec(
    vectorSize=128,
    windowSize=5,
    minCount=5,
    maxIter=5,
    numPartitions=8,
    inputCol="track_sequence",
    outputCol="track_embedding",
    seed=42,
)

w2v_model = word2vec.fit(playlist_sequences)
print(f"  Word2Vec training complete in {time.time() - t0:.1f}s")

# Save the model
w2v_model.write().overwrite().save(f"{MODEL_DIR}/word2vec_model")
print(f"  Saved Word2Vec model to {MODEL_DIR}/word2vec_model")

# Step 3: Sanity check — find tracks similar to HUMBLE. by Kendrick Lamar
humble_uri = "spotify:track:7x9aauaA9cu6tyfpHnqDLo"

print(f"\nSanity check — tracks most similar to 'HUMBLE.':")
try:
    similar = w2v_model.findSynonyms(humble_uri, 10)
    (
        similar
        .join(tracks, similar.word == tracks.track_uri)
        .select("track_name", "artist_name", "similarity")
        .show(truncate=False)
    )
except Exception as e:
    # HUMBLE's URI might not match exactly — try finding it
    print(f"  Could not find exact URI. Looking up HUMBLE...")
    humble_row = (
        tracks
        .filter(F.col("track_name") == "HUMBLE.")
        .filter(F.col("artist_name") == "Kendrick Lamar")
        .select("track_uri")
        .first()
    )
    if humble_row:
        similar = w2v_model.findSynonyms(humble_row.track_uri, 10)
        (
            similar
            .join(tracks, similar.word == tracks.track_uri)
            .select("track_name", "artist_name", "similarity")
            .show(truncate=False)
        )
    else:
        print("  Could not find HUMBLE. in track data — skipping sanity check")

print()


# ===========================================================================
# 3D. HYBRID RECOMMENDATION FUNCTION
# ===========================================================================
# Each model captures different signals:
#   - Popularity: what's globally popular (no personalization)
#   - ALS: collaborative patterns (playlists with similar taste)
#   - Word2Vec: sequential/contextual patterns (nearby tracks)
#
# A hybrid combines all three. The simplest approach: weighted scoring.
#
# For a given playlist's seed tracks, we:
#   1. Get popularity scores for all candidate tracks
#   2. Get ALS predicted scores for this playlist
#   3. Get Word2Vec similarity to the seed tracks
#   4. Combine: hybrid_score = w1*pop + w2*als + w3*w2v
#
# We'll build the full hybrid pipeline here and tune weights in Stage 4.
# For now, we just save all the components and define the combination logic.
# ===========================================================================

print("=" * 60)
print("MODEL 3D: HYBRID ENSEMBLE SETUP")
print("=" * 60)

# The Word2Vec model gives us a vocabulary of track vectors.
# Extract them as a DataFrame for easy joining later.
print("Extracting Word2Vec track vectors...")
t0 = time.time()

w2v_vectors = w2v_model.getVectors()
w2v_vectors.write.mode("overwrite").parquet(f"{PARQUET_DIR}/w2v_vectors")
print(f"  {w2v_vectors.count():,} track vectors saved in {time.time() - t0:.1f}s")

# Save ALS item factors for later use
# These are the learned track vectors from ALS
als_item_factors = als_model.itemFactors
als_item_factors.write.mode("overwrite").parquet(f"{PARQUET_DIR}/als_item_factors")
als_user_factors = als_model.userFactors
als_user_factors.write.mode("overwrite").parquet(f"{PARQUET_DIR}/als_user_factors")
print(f"  Saved ALS factors ({als_item_factors.count():,} items, {als_user_factors.count():,} users)")

print()


# ===========================================================================
# SUMMARY
# ===========================================================================

print("=" * 60)
print("STAGE 3 COMPLETE — Summary")
print("=" * 60)
print()
print("Models trained:")
print("  ✅ 3A Popularity Baseline  — sorted by playlist count")
print("  ✅ 3B ALS (rank=64)        — collaborative filtering on 66M interactions")
print("  ✅ 3C Word2Vec (dim=128)   — track embeddings from playlist sequences")
print("  ✅ 3D Hybrid setup         — all components saved for ensemble")
print()
print("Validation set:")
print(f"  {val_pid_count} playlists reserved for evaluation")
print()
print("Saved to:")
print(f"  Models:  {os.path.abspath(MODEL_DIR)}/")
print(f"  Tables:  {os.path.abspath(PARQUET_DIR)}/")
print()
print("Model sizes:")
for name in ["als_model", "word2vec_model"]:
    path = f"{MODEL_DIR}/{name}"
    if os.path.exists(path):
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(path)
            for f in fnames
        )
        print(f"  {name}: {total / (1024**2):.1f} MB")

print()
print("Next: python stage4_evaluate.py")

spark.stop()
