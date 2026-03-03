#!/usr/bin/env python3
"""Smoke-test for CoreS2L2A dataset adapter.

Usage:
    python scripts/test_core_s2l2a.py
    python scripts/test_core_s2l2a.py --root /data/databases/Core-S2L2A/images --rows 5
"""

import argparse
import sys
import time
import os

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/data/databases/Core-S2L2A/images",
        help="Path to the directory containing part_*.parquet files",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of samples to read during the test",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=2,
        help="Limit index building to this many parquet files (faster for smoke test)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CoreS2L2A Dataset Smoke Test")
    print("=" * 60)

    # Import here so we catch missing deps early with a clear message
    try:
        from dinov3.data.datasets.core_s2l2a import CoreS2L2A, S2L2A_BAND_NAMES, TARGET_SIZE
    except ImportError as e:
        print(f"[FAIL] Could not import CoreS2L2A: {e}")
        sys.exit(1)

    print(f"Band names ({len(S2L2A_BAND_NAMES)}): {S2L2A_BAND_NAMES}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Root: {args.root}")

    # Limit number of parquet files for a quick test
    import glob
    all_files = sorted(glob.glob(os.path.join(args.root, "*.parquet")))
    if not all_files:
        print(f"[FAIL] No parquet files found in {args.root}")
        sys.exit(1)
    files_to_use = all_files[: args.max_files]
    print(f"\nUsing {len(files_to_use)} of {len(all_files)} parquet file(s):")
    for f in files_to_use:
        print(f"  {f}")

    # Build dataset
    print("\nBuilding dataset index...")
    t0 = time.perf_counter()
    ds = CoreS2L2A(parquet_files=files_to_use)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.2f}s — {ds}")

    assert len(ds) > 0, "Dataset is empty!"
    n_test = min(args.rows, len(ds))
    print(f"\nReading {n_test} sample(s)...")

    for i in range(n_test):
        t0 = time.perf_counter()
        img, target = ds[i]
        t1 = time.perf_counter()

        assert isinstance(img, torch.Tensor), f"Expected torch.Tensor, got {type(img)}"
        assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}"
        assert img.shape == (len(S2L2A_BAND_NAMES), TARGET_SIZE, TARGET_SIZE), (
            f"Expected ({len(S2L2A_BAND_NAMES)}, {TARGET_SIZE}, {TARGET_SIZE}), got {img.shape}"
        )
        assert img.min() >= 0.0, f"Min value below 0: {img.min()}"
        assert img.max() <= 1.0, f"Max value above 1: {img.max()}"

        per_band_mean = img.mean(dim=(1, 2))
        per_band_std  = img.std(dim=(1, 2))
        print(
            f"  [{i}] shape={tuple(img.shape)}  dtype={img.dtype}  "
            f"min={img.min():.4f}  max={img.max():.4f}  "
            f"read_time={t1-t0:.2f}s"
        )
        print(f"       per-band mean: {per_band_mean.tolist()}")
        print(f"       per-band std:  {per_band_std.tolist()}")

    print("\n[PASS] All assertions passed.")

    # Optional: test DataLoader throughput
    print("\nDataLoader throughput test (batch=2, workers=2) ...")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)
    t0 = time.perf_counter()
    batch_img, batch_tgt = next(iter(loader))
    t1 = time.perf_counter()
    print(f"  Batch shape: {tuple(batch_img.shape)}  target: {batch_tgt}")
    print(f"  First batch read in {t1-t0:.2f}s")
    print("\n[PASS] DataLoader test passed.")


if __name__ == "__main__":
    main()
