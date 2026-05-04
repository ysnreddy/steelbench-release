#!/usr/bin/env python3
"""Recompute BRISQUE quality scores for extracted clips.

Reads representative frames from output/frames/ and updates the manifest
CSV with correct BRISQUE scores. Run after extract_clips.py if BRISQUE
model files were missing during extraction.

Usage:
    python recompute_brisque.py
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).parent)
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "output", "metadata", "clips_manifest.csv")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "output", "frames")
MODEL_PATH = os.path.join(PROJECT_ROOT, "brisque_model_live.yml")
RANGE_PATH = os.path.join(PROJECT_ROOT, "brisque_range_live.yml")


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"No manifest at {MANIFEST_PATH}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(RANGE_PATH):
        print(f"BRISQUE model files not found: {MODEL_PATH}, {RANGE_PATH}")
        sys.exit(1)

    brisque = cv2.quality.QualityBRISQUE_create(MODEL_PATH, RANGE_PATH)

    # Read manifest
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    needs_update = sum(1 for r in rows if r.get("brisque_score") == "-1.0")
    print(f"Total clips: {len(rows)}, need BRISQUE update: {needs_update}")

    if needs_update == 0:
        print("All clips already have BRISQUE scores.")
        return

    updated = 0
    for row in tqdm(rows, desc="Computing BRISQUE"):
        if row.get("brisque_score") != "-1.0":
            continue

        clip_id = row["clip_id"]
        frames_dir = os.path.join(FRAMES_DIR, clip_id)

        if not os.path.exists(frames_dir):
            continue

        scores = []
        for fname in sorted(os.listdir(frames_dir)):
            if fname.endswith(".jpg"):
                frame = cv2.imread(os.path.join(frames_dir, fname))
                if frame is not None:
                    try:
                        score = brisque.compute(frame)
                        if isinstance(score, tuple):
                            scores.append(score[0])
                        else:
                            scores.append(float(score))
                    except Exception:
                        pass

        if scores:
            mean_brisque = round(float(np.mean(scores)), 2)
            row["brisque_score"] = str(mean_brisque)
            updated += 1

    # Write updated manifest
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} clips with BRISQUE scores.")
    print(f"Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
