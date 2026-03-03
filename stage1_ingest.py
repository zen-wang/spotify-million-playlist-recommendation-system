"""
Stage 1: Data Ingestion — JSON → Parquet
=========================================
Reads the 1,000 MPD JSON slice files, flattens the nested playlist→tracks
structure, and writes three optimized Parquet tables:

    - playlists:       1M rows  (one per playlist)
    - tracks:          ~2.2M rows (one per unique track, deduplicated)
    - playlist_tracks: ~66M rows (every playlist-track pair with position)

Usage:
    python stage1_ingest.py

Expected runtime: ~10-20 minutes on a laptop with 24GB RAM.
Output: ./output/parquet/{playlists, tracks, playlist_tracks}
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    BooleanType, ArrayType
)
import time
import os


# ===========================================================================
# 1. SPARK SESSION SETUP
# ===========================================================================
# SparkSession is the single entry point to all Spark functionality.
# Think of it like a "connection" — you need one before doing anything.
#
# Key configs for local mode:
#   - master("local[*]")  → use all CPU cores. [4] would use 4 cores.
#   - driver.memory       → how much RAM Spark can use. We give it 12GB
#                           out of your 24GB, leaving room for the OS.
#   - shuffle.partitions  → when Spark rearranges data (joins, groupBys),
#                           it splits into this many chunks. Default is 200
#                           which is overkill for local mode. 8 is enough
#                           and avoids creating tons of tiny files.
#   - parquet settings    → control output file size and compression.
# ===========================================================================

spark = (
    SparkSession.builder
    .appName("MPD-Stage1-Ingest")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "8")
    # Write fewer, larger parquet files (easier to manage)
    .config("spark.sql.files.maxRecordsPerFile", 0)  # no per-file limit
    .getOrCreate()
)

# Suppress noisy INFO logs — only show warnings and errors
spark.sparkContext.setLogLevel("WARN")

print(f"Spark version: {spark.version}")
print(f"Spark UI: http://localhost:4040 (open this in browser to monitor jobs!)")
print()


# ===========================================================================
# 2. DEFINE PATHS
# ===========================================================================
# Relative to where you run the script (project root)

DATA_DIR = "spotify_million_playlist_dataset/data"
OUTPUT_DIR = "output/parquet"

# Verify data directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        f"Make sure you run this script from: Spotify_Million_Playlist_Dataset_Challenge/"
    )


# ===========================================================================
# 3. DEFINE SCHEMA (instead of letting Spark infer it)
# ===========================================================================
# WHY define schema explicitly?
#   - Performance: Spark doesn't need to scan files to figure out types.
#   - Correctness: "collaborative" is stored as string "true"/"false" in the
#     JSON, not a real boolean. We can handle this ourselves.
#   - Resume talking point: "I defined explicit schemas to avoid schema
#     inference overhead on 1,000 files."
#
# This schema matches the MPD JSON structure from the README:
# { "info": {...}, "playlists": [ { ..., "tracks": [ {...}, ... ] } ] }
# ===========================================================================

track_schema = StructType([
    StructField("pos", IntegerType()),
    StructField("artist_name", StringType()),
    StructField("track_uri", StringType()),
    StructField("artist_uri", StringType()),
    StructField("track_name", StringType()),
    StructField("album_uri", StringType()),
    StructField("duration_ms", IntegerType()),
    StructField("album_name", StringType()),
])

playlist_schema = StructType([
    StructField("name", StringType()),
    StructField("collaborative", StringType()),  # "true"/"false" string
    StructField("pid", IntegerType()),
    StructField("modified_at", IntegerType()),
    StructField("num_albums", IntegerType()),
    StructField("num_tracks", IntegerType()),
    StructField("num_followers", IntegerType()),
    StructField("num_edits", IntegerType()),
    StructField("duration_ms", LongType()),  # Long because total ms can exceed int range
    StructField("num_artists", IntegerType()),
    StructField("description", StringType()),  # optional field — will be null if missing
    StructField("tracks", ArrayType(track_schema)),
])

# Top-level JSON schema
mpd_schema = StructType([
    StructField("info", StructType([
        StructField("generated_on", StringType()),
        StructField("slice", StringType()),
        StructField("version", StringType()),
    ])),
    StructField("playlists", ArrayType(playlist_schema)),
])


# ===========================================================================
# 4. READ ALL JSON FILES
# ===========================================================================
# spark.read.json() can take a directory path and read all JSON files in it.
#
# IMPORTANT PySpark concept — LAZY EVALUATION:
#   This line does NOT load 33GB into memory. Spark just records "I need to
#   read these files" in its execution plan. Data only moves when you trigger
#   an ACTION (like .count(), .write(), .show()).
#
# multiLine=True because each JSON file is one big object, not one JSON per line.
# ===========================================================================

print("Reading JSON files...")
t0 = time.time()

raw_df = (
    spark.read
    .schema(mpd_schema)
    .option("multiLine", True)
    .json(DATA_DIR)
)

# Quick check — this IS an action, so Spark actually reads files here
file_count = raw_df.count()
print(f"  Read {file_count} slice files in {time.time() - t0:.1f}s")


# ===========================================================================
# 5. FLATTEN PLAYLISTS
# ===========================================================================
# The raw JSON has structure:  { playlists: [ {playlist1}, {playlist2}, ... ] }
# Each file has ~1000 playlists packed into one array.
#
# explode() is a key Spark function:
#   - Input:  1 row with an array of 1000 playlists
#   - Output: 1000 rows, one per playlist
#
# Then we select the individual fields from the struct.
# ===========================================================================

print("Flattening playlists...")
t0 = time.time()

playlists_df = (
    raw_df
    # explode turns each array element into its own row
    .select(F.explode("playlists").alias("pl"))
    # now "pl" is a struct — pull out each field
    .select(
        F.col("pl.pid"),
        F.col("pl.name"),
        F.col("pl.description"),
        # Convert "true"/"false" strings to real booleans
        (F.col("pl.collaborative") == "true").alias("collaborative"),
        F.col("pl.modified_at"),
        F.col("pl.num_tracks"),
        F.col("pl.num_albums"),
        F.col("pl.num_artists"),
        F.col("pl.num_followers"),
        F.col("pl.num_edits"),
        F.col("pl.duration_ms"),
        # Keep tracks array for the next step
        F.col("pl.tracks"),
    )
)

playlist_count = playlists_df.count()
print(f"  {playlist_count:,} playlists in {time.time() - t0:.1f}s")


# ===========================================================================
# 6. BUILD PLAYLIST_TRACKS TABLE (the big one: ~66M rows)
# ===========================================================================
# This is the core interaction table: which track is in which playlist,
# and at what position.
#
# We explode again — this time the tracks array inside each playlist.
# One playlist with 50 tracks becomes 50 rows.
#
# This table is what ALS collaborative filtering will train on.
# ===========================================================================

print("Building playlist_tracks table...")
t0 = time.time()

playlist_tracks_df = (
    playlists_df
    .select(
        F.col("pid"),
        F.explode("tracks").alias("track"),
    )
    .select(
        F.col("pid"),
        F.col("track.track_uri"),
        F.col("track.pos"),
    )
)

pt_count = playlist_tracks_df.count()
print(f"  {pt_count:,} playlist-track pairs in {time.time() - t0:.1f}s")


# ===========================================================================
# 7. BUILD DEDUPLICATED TRACKS TABLE (~2.2M unique tracks)
# ===========================================================================
# The same track appears in many playlists. We want one row per unique track
# with its metadata (name, artist, album, duration).
#
# dropDuplicates(["track_uri"]) keeps the first occurrence.
#
# WHY deduplicate? So we have a clean lookup table. When the model predicts
# "recommend track_uri X", we can join back to get the track name, artist, etc.
# ===========================================================================

print("Building deduplicated tracks table...")
t0 = time.time()

tracks_df = (
    playlists_df
    .select(F.explode("tracks").alias("track"))
    .select(
        F.col("track.track_uri"),
        F.col("track.track_name"),
        F.col("track.artist_uri"),
        F.col("track.artist_name"),
        F.col("track.album_uri"),
        F.col("track.album_name"),
        F.col("track.duration_ms"),
    )
    .dropDuplicates(["track_uri"])
)

track_count = tracks_df.count()
print(f"  {track_count:,} unique tracks in {time.time() - t0:.1f}s")


# ===========================================================================
# 8. DROP THE NESTED TRACKS COLUMN FROM PLAYLISTS
# ===========================================================================
# The playlists table doesn't need the full tracks array anymore —
# that info is now in playlist_tracks. Keeping it would waste space.
# ===========================================================================

playlists_clean_df = playlists_df.drop("tracks")


# ===========================================================================
# 9. WRITE TO PARQUET
# ===========================================================================
# Parquet is a columnar storage format — it's what the industry uses for
# big data. Compared to the raw JSON:
#   - 33GB JSON → ~5GB Parquet (columnar + compression)
#   - Much faster to read (Spark can skip columns it doesn't need)
#   - Preserves schema (no re-inference needed)
#
# coalesce(N) controls how many output files we get. Fewer files = easier
# to manage on a laptop. We don't need 200 tiny files.
#
# mode("overwrite") replaces existing output — safe for re-running.
# ===========================================================================

print("Writing Parquet files...")

# Playlists: 1M rows is small — one file is fine
print("  Writing playlists...")
t0 = time.time()
(
    playlists_clean_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{OUTPUT_DIR}/playlists")
)
print(f"    Done in {time.time() - t0:.1f}s")

# Tracks: 2.2M rows, still small — one file is fine
print("  Writing tracks...")
t0 = time.time()
(
    tracks_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(f"{OUTPUT_DIR}/tracks")
)
print(f"    Done in {time.time() - t0:.1f}s")

# Playlist-tracks: 66M rows — use more files for parallelism in later stages
# 8 files ≈ 8M rows each, good balance for local mode
print("  Writing playlist_tracks...")
t0 = time.time()
(
    playlist_tracks_df
    .coalesce(8)
    .write
    .mode("overwrite")
    .parquet(f"{OUTPUT_DIR}/playlist_tracks")
)
print(f"    Done in {time.time() - t0:.1f}s")


# ===========================================================================
# 10. VALIDATION — compare against known stats
# ===========================================================================
# From stats.txt we know the ground truth. Let's verify our pipeline is correct.
# This is a critical step — if these numbers don't match, something went wrong.
# ===========================================================================

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

# Re-read from Parquet to verify the written data
p_df = spark.read.parquet(f"{OUTPUT_DIR}/playlists")
t_df = spark.read.parquet(f"{OUTPUT_DIR}/tracks")
pt_df = spark.read.parquet(f"{OUTPUT_DIR}/playlist_tracks")

results = {
    "playlists":           (p_df.count(),  1_000_000),
    "total_tracks":        (pt_df.count(), 66_346_428),
    "unique_tracks":       (t_df.count(),  2_262_292),
    "unique_artists":      (t_df.select("artist_uri").distinct().count(), 295_860),
    "unique_albums":       (t_df.select("album_uri").distinct().count(), 734_684),
}

all_passed = True
for name, (actual, expected) in results.items():
    status = "✅" if actual == expected else "❌"
    if actual != expected:
        all_passed = False
    print(f"  {status} {name}: {actual:,} (expected {expected:,})")

# Check average playlist length
avg_len = pt_df.count() / p_df.count()
print(f"  {'✅' if abs(avg_len - 66.346428) < 0.001 else '❌'} avg playlist length: {avg_len:.6f} (expected 66.346428)")

print()
if all_passed:
    print("All validations passed! Stage 1 complete. 🎉")
else:
    print("⚠️  Some validations failed — check the output above.")

# Show Parquet file sizes
print()
print("Output file sizes:")
for table in ["playlists", "tracks", "playlist_tracks"]:
    path = f"{OUTPUT_DIR}/{table}"
    total_bytes = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".parquet")
    )
    print(f"  {table}: {total_bytes / (1024**2):.1f} MB")

print()
print(f"Parquet output directory: {os.path.abspath(OUTPUT_DIR)}/")

# Clean up
spark.stop()
