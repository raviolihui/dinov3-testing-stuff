#!/usr/bin/env python3
"""Compute per-band statistics for Core-S2L2A by sampling random tiles and pixels.

This script reads a random subset of parquet part files from the Major-TOM Core-S2L2A
`images/` directory, decodes the embedded GeoTIFF bytes for each band, scales
reflectances by 10000.0 (S2 L2A convention), and computes per-band mean/std/min/max.

It samples a configurable number of tiles (rows) and a configurable number of
pixels per tile per band to keep memory and runtime bounded.

Example:
    python scripts/compute_core_s2l2a_band_stats.py \
        --root /data/databases/Core-S2L2A/images \
        --num-tiles 200 --max-files 20 --pixels-per-tile 2000

"""
from __future__ import annotations

import argparse
import glob
import math
import random
import sys
from typing import List

import numpy as np
import pyarrow.parquet as pq
from rasterio.io import MemoryFile


S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B09",
    "B11",
    "B12",
]
S2_SCALE = 10000.0


def decode_band_blob(blob: bytes) -> np.ndarray:
    with MemoryFile(blob) as mem:
        with mem.open() as src:
            arr = src.read(1)
    return arr.astype(np.float32)


def sample_tile_pixels(arr: np.ndarray, n_samples: int) -> np.ndarray:
    """Return up to n_samples pixel values sampled uniformly from arr (2D).

    If arr.size <= n_samples, return all values flattened.
    """
    flat = arr.ravel()
    N = flat.size
    if N <= n_samples:
        return flat
    # choose without replacement
    idx = np.random.choice(N, size=n_samples, replace=False)
    return flat[idx]


def compute_stats(root: str, num_tiles: int = 200, max_files: int = 50, pixels_per_tile: int = 2000, seed: int = 42):
    files = sorted(glob.glob(f"{root}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    files = files[:max_files]
    random.Random(seed).shuffle(files)

    # accumulators per band
    C = len(S2_BANDS)
    counts = np.zeros(C, dtype=np.int64)
    sums = np.zeros(C, dtype=np.float64)
    sumsqs = np.zeros(C, dtype=np.float64)
    mins = np.full(C, np.inf, dtype=np.float64)
    maxs = np.full(C, -np.inf, dtype=np.float64)

    tiles_processed = 0

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.num_row_groups):
            # read this row group, but only the band columns
            try:
                table = pf.read_row_group(rg, columns=S2_BANDS)
            except Exception as e:
                print(f"Warning: could not read row group {rg} in {fpath}: {e}", file=sys.stderr)
                continue
            n_rows = table.num_rows
            # iterate rows in this row group
            for r in range(n_rows):
                row = {col: table.column(col)[r].as_py() for col in S2_BANDS}
                # for each band, decode and sample
                for bi, bname in enumerate(S2_BANDS):
                    blob = row[bname]
                    if blob is None:
                        continue
                    try:
                        arr = decode_band_blob(blob)
                    except Exception as e:
                        print(f"Warning: failed decoding band {bname} in {fpath} rg{rg} row{r}: {e}", file=sys.stderr)
                        continue
                    # scale
                    arr = arr.astype(np.float64) / S2_SCALE
                    # subsample pixels
                    pixels = sample_tile_pixels(arr, pixels_per_tile)
                    # update accumulators
                    n = pixels.size
                    counts[bi] += n
                    sums[bi] += pixels.sum()
                    sumsqs[bi] += (pixels ** 2).sum()
                    mins[bi] = min(mins[bi], float(pixels.min()))
                    maxs[bi] = max(maxs[bi], float(pixels.max()))

                tiles_processed += 1
                if tiles_processed >= num_tiles:
                    break
            if tiles_processed >= num_tiles:
                break
        if tiles_processed >= num_tiles:
            break

    # compute final stats
    means = sums / counts
    variances = sumsqs / counts - means ** 2
    stds = np.sqrt(np.maximum(variances, 0.0))

    # Print results
    print(f"Tiles processed: {tiles_processed}")
    print("Per-band statistics (bands order):", S2_BANDS)
    for i, b in enumerate(S2_BANDS):
        print(
            f"{b}: count={counts[i]:d}  mean={means[i]:.6f}  std={stds[i]:.6f}  min={mins[i]:.6f}  max={maxs[i]:.6f}"
        )

    return {
        "bands": S2_BANDS,
        "counts": counts,
        "means": means,
        "stds": stds,
        "mins": mins,
        "maxs": maxs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/databases/Core-S2L2A/images")
    p.add_argument("--num-tiles", type=int, default=200)
    p.add_argument("--max-files", type=int, default=50)
    p.add_argument("--pixels-per-tile", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    compute_stats(
        root=args.root,
        num_tiles=args.num_tiles,
        max_files=args.max_files,
        pixels_per_tile=args.pixels_per_tile,
        seed=args.seed,
    )
