#!/usr/bin/env python3
"""Verify that every experiment used the same leakage-free train/val/test split.

Each run writes a `split_proof` JSON holding the SHA256 of its sorted train / val /
test file lists. This script checks two things for every dataset:

  1. all runs on that dataset (full model and every ablation) share identical hashes;
  2. those hashes start with the prefixes published in the paper (Table II).

Usage (from the repository root):
    python scripts/verify_splits.py
Exit code is 0 if everything matches and 1 otherwise.
"""
import json
import sys
from pathlib import Path

PUBLISHED = {  # first 12 hex characters of each SHA256, as printed in the paper
    "gopro":     {"train": "7d659adfcd1f", "val": "063ab299a04b", "test": "7018c8b387e9"},
    "hide":      {"train": "e62b33858f1b", "val": "34a1110dae60", "test": "e216a016ecbd"},
    "realblurj": {"train": "af1027140302", "val": "8431800b03e0", "test": "2a98ebbf6075"},
}
EXPECTED_COUNTS = {
    "gopro":     {"train": 1893, "val": 210, "test": 1111},
    "hide":      {"train": 5727, "val": 636, "test": 1329},
    "realblurj": {"train": 3383, "val": 375, "test": 980},
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "results" / "split_proofs"
    files = sorted(root.glob("*.json"))
    if not files:
        print(f"No split proofs found in {root}")
        return 1

    ok = True
    for dataset, prefixes in PUBLISHED.items():
        runs = [f for f in files if f.name.startswith(dataset)]
        print(f"\n{dataset.upper()}  ({len(runs)} runs)")
        reference = None
        for f in runs:
            proof = json.loads(f.read_text(encoding="utf-8"))
            hashes = proof["sha256"]
            counts = proof["counts"]
            if reference is None:
                reference = hashes
            same = all(hashes[k] == reference[k] for k in ("train", "val", "test"))
            match = all(hashes[k].startswith(prefixes[k]) for k in ("train", "val", "test"))
            sized = all(counts[k] == EXPECTED_COUNTS[dataset][k] for k in ("train", "val", "test"))
            good = same and match and sized
            ok &= good
            print(f"  [{'OK ' if good else 'FAIL'}] {f.stem:32s} "
                  f"train {hashes['train'][:12]}  identical: {same}  matches paper: {match}  counts: {sized}")

    print("\nRESULT:", "all runs share the published, identical splits" if ok else "MISMATCH FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
