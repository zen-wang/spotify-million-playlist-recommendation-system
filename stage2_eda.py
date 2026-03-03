"""
Stage 2: EDA & Feature Engineering
====================================
Loads the Parquet tables from Stage 1 and runs exploratory queries +
builds feature tables needed for modeling in Stage 3.

This stage teaches:
  - Spark SQL (writing SQL directly against DataFrames)
  - GroupBy aggregations
  - Window functions
  - Caching (keeping hot data in memory)
  - Building feature tables for downstream ML

Usage:
    python stage2_eda.py

Input:  ./output/parquet/{playlists, tracks, playlist_tracks}
Output: ./output/parquet/{track_popularity, artist_popularity, cooccurrence}
        ./output/eda/ (printed stats and summaries)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import time
import os

# ===========================================================================
# 1. SPARK SESSION + LOAD PARQUET
# ===========================================================================
# Notice how fast Parquet loads compared to Stage 1's JSON reading.
# Parquet stores schema inside the files, so no inference needed.
# ===========================================================================

spark = (
    SparkSession.builder
    .appName("MPD-Stage2-EDA")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

PARQUET_DIR = "output/parquet"
OUTPUT_DIR = "output/eda"
FEATURE_DIR = "output/parquet"  # feature tables go alongside the base tables

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Parquet tables...")
t0 = time.time()

playlists = spark.read.parquet(f"{PARQUET_DIR}/playlists")
tracks = spark.read.parquet(f"{PARQUET_DIR}/tracks")
playlist_tracks = spark.read.parquet(f"{PARQUET_DIR}/playlist_tracks")

print(f"  Loaded in {time.time() - t0:.1f}s")


# ===========================================================================
# 2. REGISTER AS SQL TABLES
# ===========================================================================
# createOrReplaceTempView lets you write raw SQL against DataFrames.
# This is powerful because:
#   - You already know SQL (most people do)
#   - Spark optimizes SQL and DataFrame operations the same way
#   - Interviewers often ask "can you write Spark SQL?"
#
# Under the hood, Spark SQL and DataFrame API produce the SAME execution
# plan — there's no performance difference. Use whichever is clearer.
# ===========================================================================

playlists.createOrReplaceTempView("playlists")
tracks.createOrReplaceTempView("tracks")
playlist_tracks.createOrReplaceTempView("playlist_tracks")


# ===========================================================================
# 3. CACHE THE PLAYLIST_TRACKS TABLE
# ===========================================================================
# cache() tells Spark: "keep this DataFrame in memory after the first time
# you compute it." Since we'll query playlist_tracks many times in this
# script, caching avoids re-reading Parquet from disk each time.
#
# IMPORTANT: cache() is lazy — it doesn't load into memory immediately.
# The data gets cached on the first ACTION (count, show, etc.).
#
# On your 24GB machine with 12GB for Spark, the ~1.5GB playlist_tracks
# table fits easily in memory.
# ===========================================================================

playlist_tracks.cache()
# Trigger the cache by running a count
pt_count = playlist_tracks.count()
print(f"  Cached playlist_tracks: {pt_count:,} rows")
print()


# ===========================================================================
# PART A: EXPLORATORY DATA ANALYSIS
# ===========================================================================

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ---------------------------------------------------------------------------
# A1. Playlist Length Distribution
# ---------------------------------------------------------------------------
# How long are playlists? This matters because:
#   - Very short playlists give less signal for collaborative filtering
#   - Very long playlists might be "dump" playlists (less curated)
#   - The challenge has scenarios from 0 to 100 seed tracks
# ---------------------------------------------------------------------------

print("\n--- A1: Playlist Length Distribution ---")

length_stats = spark.sql("""
    SELECT
        MIN(num_tracks)    AS min_length,
        MAX(num_tracks)    AS max_length,
        AVG(num_tracks)    AS avg_length,
        PERCENTILE(num_tracks, 0.25) AS p25,
        PERCENTILE(num_tracks, 0.50) AS median,
        PERCENTILE(num_tracks, 0.75) AS p75,
        PERCENTILE(num_tracks, 0.95) AS p95
    FROM playlists
""")
length_stats.show(truncate=False)

# Histogram buckets
print("Playlist length buckets:")
spark.sql("""
    SELECT
        CASE
            WHEN num_tracks <= 10  THEN '5-10'
            WHEN num_tracks <= 25  THEN '11-25'
            WHEN num_tracks <= 50  THEN '26-50'
            WHEN num_tracks <= 100 THEN '51-100'
            WHEN num_tracks <= 150 THEN '101-150'
            ELSE '151-250'
        END AS length_bucket,
        COUNT(*) AS num_playlists,
        ROUND(COUNT(*) * 100.0 / 1000000, 2) AS pct
    FROM playlists
    GROUP BY 1
    ORDER BY MIN(num_tracks)
""").show(truncate=False)


# ---------------------------------------------------------------------------
# A2. Top 20 Most Popular Tracks
# ---------------------------------------------------------------------------
# "Popularity" = how many playlists a track appears in.
# This is the simplest recommendation signal and our baseline model.
# ---------------------------------------------------------------------------

print("--- A2: Top 20 Tracks ---")

# Here's the Spark SQL way vs DataFrame API — both produce identical results.
# SQL version:
top_tracks_sql = spark.sql("""
    SELECT
        t.track_name,
        t.artist_name,
        COUNT(*) AS playlist_count
    FROM playlist_tracks pt
    JOIN tracks t ON pt.track_uri = t.track_uri
    GROUP BY t.track_name, t.artist_name
    ORDER BY playlist_count DESC
    LIMIT 20
""")

# DataFrame API version (equivalent — pick whichever you prefer):
# top_tracks_df = (
#     playlist_tracks
#     .groupBy("track_uri").count()
#     .join(tracks, "track_uri")
#     .select("track_name", "artist_name", F.col("count").alias("playlist_count"))
#     .orderBy(F.desc("playlist_count"))
#     .limit(20)
# )

top_tracks_sql.show(truncate=False)


# ---------------------------------------------------------------------------
# A3. Top 20 Most Popular Artists
# ---------------------------------------------------------------------------

print("--- A3: Top 20 Artists ---")

spark.sql("""
    SELECT
        t.artist_name,
        COUNT(*) AS track_appearances
    FROM playlist_tracks pt
    JOIN tracks t ON pt.track_uri = t.track_uri
    GROUP BY t.artist_name
    ORDER BY track_appearances DESC
    LIMIT 20
""").show(truncate=False)


# ---------------------------------------------------------------------------
# A4. Playlist Followers Distribution
# ---------------------------------------------------------------------------
# Most playlists have very few followers (long tail distribution).
# This is useful context for understanding user engagement.
# ---------------------------------------------------------------------------

print("--- A4: Follower Distribution ---")

spark.sql("""
    SELECT
        CASE
            WHEN num_followers = 1  THEN '1'
            WHEN num_followers = 2  THEN '2'
            WHEN num_followers <= 5 THEN '3-5'
            WHEN num_followers <= 10 THEN '6-10'
            WHEN num_followers <= 50 THEN '11-50'
            ELSE '50+'
        END AS follower_bucket,
        COUNT(*) AS num_playlists,
        ROUND(COUNT(*) * 100.0 / 1000000, 2) AS pct
    FROM playlists
    GROUP BY 1
    ORDER BY MIN(num_followers)
""").show(truncate=False)


# ---------------------------------------------------------------------------
# A5. Playlist Title Analysis
# ---------------------------------------------------------------------------
# Titles are a key signal — the challenge includes "title only" scenarios.
# Knowing common titles helps us build a title-based recommender later.
# ---------------------------------------------------------------------------

print("--- A5: Top 20 Playlist Titles (normalized) ---")

# Normalize: lowercase, strip punctuation — same as the MPD's normalize_name
spark.sql("""
    SELECT
        LOWER(TRIM(name)) AS title_normalized,
        COUNT(*) AS count
    FROM playlists
    GROUP BY 1
    ORDER BY count DESC
    LIMIT 20
""").show(truncate=False)

# How many playlists have descriptions?
desc_count = spark.sql("""
    SELECT COUNT(*) AS with_description
    FROM playlists
    WHERE description IS NOT NULL
""").collect()[0]["with_description"]
print(f"Playlists with descriptions: {desc_count:,} ({desc_count/10000:.1f}%)")
print()


# ---------------------------------------------------------------------------
# A6. Track Frequency Distribution (long tail analysis)
# ---------------------------------------------------------------------------
# How many tracks appear in only 1 playlist vs many playlists?
# This tells us about data sparsity — a key challenge for RecSys.
# ---------------------------------------------------------------------------

print("--- A6: Track Frequency Distribution ---")

spark.sql("""
    WITH track_freq AS (
        SELECT track_uri, COUNT(*) AS freq
        FROM playlist_tracks
        GROUP BY track_uri
    )
    SELECT
        CASE
            WHEN freq = 1       THEN '1 (appears once)'
            WHEN freq <= 5      THEN '2-5'
            WHEN freq <= 20     THEN '6-20'
            WHEN freq <= 100    THEN '21-100'
            WHEN freq <= 1000   THEN '101-1000'
            WHEN freq <= 10000  THEN '1001-10000'
            ELSE '10000+'
        END AS frequency_bucket,
        COUNT(*) AS num_tracks,
        ROUND(COUNT(*) * 100.0 / 2262292, 2) AS pct_of_tracks
    FROM track_freq
    GROUP BY 1
    ORDER BY MIN(freq)
""").show(truncate=False)


# ===========================================================================
# PART B: FEATURE ENGINEERING
# ===========================================================================
# Build feature tables that Stage 3 (modeling) will consume.
# ===========================================================================

print()
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ---------------------------------------------------------------------------
# B1. Track Popularity Table
# ---------------------------------------------------------------------------
# For each track: how many playlists it appears in, and its popularity rank.
#
# NEW CONCEPT — Window Functions:
#   RANK() OVER (ORDER BY ...) assigns a rank to each row based on a
#   sort order, without collapsing rows like GROUP BY does. It's like
#   adding a new column that says "this is the #5 most popular track."
#
#   Window functions are a Spark SQL superpower and a common interview topic.
# ---------------------------------------------------------------------------

print("\n--- B1: Building track_popularity table ---")
t0 = time.time()

track_popularity = spark.sql("""
    SELECT
        track_uri,
        COUNT(*) AS playlist_count,
        RANK() OVER (ORDER BY COUNT(*) DESC) AS popularity_rank
    FROM playlist_tracks
    GROUP BY track_uri
""")

track_popularity.write.mode("overwrite").parquet(f"{FEATURE_DIR}/track_popularity")
tp_count = track_popularity.count()
print(f"  {tp_count:,} tracks with popularity scores in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# B2. Artist Popularity Table
# ---------------------------------------------------------------------------
# For each artist: total appearances, unique tracks, unique playlists.
# Useful for artist-based features and cold-start fallback.
# ---------------------------------------------------------------------------

print("--- B2: Building artist_popularity table ---")
t0 = time.time()

artist_popularity = spark.sql("""
    SELECT
        t.artist_uri,
        t.artist_name,
        COUNT(*)                        AS total_appearances,
        COUNT(DISTINCT pt.track_uri)    AS unique_tracks,
        COUNT(DISTINCT pt.pid)          AS unique_playlists,
        RANK() OVER (ORDER BY COUNT(*) DESC) AS popularity_rank
    FROM playlist_tracks pt
    JOIN tracks t ON pt.track_uri = t.track_uri
    GROUP BY t.artist_uri, t.artist_name
""")

artist_popularity.write.mode("overwrite").parquet(f"{FEATURE_DIR}/artist_popularity")
ap_count = artist_popularity.count()
print(f"  {ap_count:,} artists with popularity scores in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# B3. Track Co-occurrence (Top Pairs)
# ---------------------------------------------------------------------------
# Co-occurrence = two tracks that frequently appear in the same playlist.
# This is one of the strongest signals for playlist continuation:
#   "If track A and track B appear together in many playlists,
#    and a new playlist has track A, recommend track B."
#
# IMPORTANT — this is a SELF-JOIN on playlist_tracks, which is expensive:
#   Every pair of tracks in a playlist generates a row. A 100-track playlist
#   produces ~5,000 pairs. Across 1M playlists this is MASSIVE.
#
# To keep it manageable, we:
#   1. Only consider tracks in the top 10,000 by popularity
#   2. This still captures the most useful co-occurrence patterns
#
# NEW CONCEPT — Broadcast Join:
#   When one side of a join is small enough to fit in memory (our top_tracks
#   list), Spark can "broadcast" it to every worker instead of shuffling
#   the big table. This is dramatically faster.
#   F.broadcast(small_df) hints Spark to use this strategy.
# ---------------------------------------------------------------------------

print("--- B3: Building track co-occurrence table ---")
print("  (This is the most expensive operation — may take a few minutes)")
t0 = time.time()

# Step 1: Get top 10K most popular tracks
top_track_uris = (
    spark.read.parquet(f"{FEATURE_DIR}/track_popularity")
    .filter(F.col("popularity_rank") <= 10000)
    .select("track_uri")
)

# Step 2: Filter playlist_tracks to only these popular tracks
# broadcast() tells Spark: "this small DataFrame (10K rows) should be sent
# to every worker, so the big table (66M rows) doesn't need to shuffle."
pt_filtered = (
    playlist_tracks
    .join(F.broadcast(top_track_uris), "track_uri")
    .select("pid", "track_uri")
)
pt_filtered.cache()
pt_filtered.count()  # trigger cache

# Step 3: Self-join to find all pairs that share a playlist
# t1.track_uri < t2.track_uri avoids counting (A,B) and (B,A) separately
cooccurrence = (
    pt_filtered.alias("t1")
    .join(pt_filtered.alias("t2"),
          (F.col("t1.pid") == F.col("t2.pid")) &
          (F.col("t1.track_uri") < F.col("t2.track_uri")))
    .groupBy(
        F.col("t1.track_uri").alias("track_uri_1"),
        F.col("t2.track_uri").alias("track_uri_2"),
    )
    .agg(F.count("*").alias("co_count"))
    # Only keep pairs that appear together in at least 10 playlists
    .filter(F.col("co_count") >= 10)
)

cooccurrence.write.mode("overwrite").parquet(f"{FEATURE_DIR}/cooccurrence")
cooc_count = cooccurrence.count()
print(f"  {cooc_count:,} co-occurring track pairs in {time.time() - t0:.1f}s")

pt_filtered.unpersist()  # free cache memory


# ---------------------------------------------------------------------------
# B4. Show sample co-occurrence results
# ---------------------------------------------------------------------------

print("\n--- Top 20 Track Pairs (most co-occurring) ---")

(
    cooccurrence
    .join(tracks.alias("t1"), cooccurrence.track_uri_1 == F.col("t1.track_uri"))
    .join(tracks.alias("t2"), cooccurrence.track_uri_2 == F.col("t2.track_uri"))
    .select(
        F.col("t1.track_name").alias("track_1"),
        F.col("t1.artist_name").alias("artist_1"),
        F.col("t2.track_name").alias("track_2"),
        F.col("t2.artist_name").alias("artist_2"),
        "co_count",
    )
    .orderBy(F.desc("co_count"))
    .limit(20)
    .show(truncate=40)
)


# ===========================================================================
# SUMMARY
# ===========================================================================

print()
print("=" * 60)
print("STAGE 2 COMPLETE — Summary of outputs")
print("=" * 60)

# Show feature table sizes
print("\nFeature tables:")
for table in ["track_popularity", "artist_popularity", "cooccurrence"]:
    path = f"{FEATURE_DIR}/{table}"
    if os.path.exists(path):
        total_bytes = sum(
            os.path.getsize(os.path.join(path, f))
            for f in os.listdir(path)
            if f.endswith(".parquet")
        )
        print(f"  {table}: {total_bytes / (1024**2):.1f} MB")

print()
print("These feature tables feed into Stage 3 (modeling):")
print("  - track_popularity  → Popularity baseline model")
print("  - artist_popularity → Cold-start fallback + artist features")
print("  - cooccurrence      → Co-occurrence based recommendations")
print()
print("Next: python stage3_models.py")

spark.stop()
