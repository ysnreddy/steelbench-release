#!/usr/bin/env python3
"""Re-extract 8 representative frames from existing clip MP4s.

Skips clips that already have 8 frames (e.g., pilot batch).
Much faster than re-running the full extraction pipeline.

Usage:
    python reextract_frames.py                    # all clips
    python reextract_frames.py --dry-run          # show what would be done
    python reextract_frames.py --workers 8        # parallel workers
"""

import argparse
import cv2
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


FRAMES_DIR = "output/frames"
CLIPS_DIR = "output/clips"
NUM_FRAMES = 8
JPEG_QUALITY = 95


def find_clip_mp4(clip_id):
    """Find the MP4 file for a given clip_id by searching clips dir."""
    for root, dirs, files in os.walk(CLIPS_DIR):
        for f in files:
            if f == f"{clip_id}.mp4":
                return os.path.join(root, f)
    return None


def extract_8_frames(clip_mp4, output_dir, num_frames=8, quality=95):
    """Extract num_frames evenly spaced frames from a clip MP4."""
    cap = cv2.VideoCapture(clip_mp4)
    if not cap.isOpened():
        return False, f"Cannot open {clip_mp4}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        # Use all available frames
        offsets = list(range(total_frames))
    else:
        offsets = [int(total_frames * i / (num_frames - 1)) for i in range(num_frames)]
        # Clamp last frame
        offsets[-1] = min(offsets[-1], total_frames - 1)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, frame_idx in enumerate(offsets):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1

    cap.release()
    return saved == len(offsets), f"Saved {saved}/{len(offsets)} frames"


def process_clip(clip_id):
    """Process a single clip: find MP4, extract 8 frames, replace old ones."""
    frame_dir = os.path.join(FRAMES_DIR, clip_id)

    # Find clip MP4
    clip_mp4 = find_clip_mp4(clip_id)
    if not clip_mp4:
        return clip_id, False, "MP4 not found"

    # Remove old frames
    if os.path.exists(frame_dir):
        for f in os.listdir(frame_dir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(frame_dir, f))

    # Extract new frames
    ok, msg = extract_8_frames(clip_mp4, frame_dir, NUM_FRAMES, JPEG_QUALITY)
    return clip_id, ok, msg


def main():
    parser = argparse.ArgumentParser(description="Re-extract 8 frames from clip MP4s")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only")
    args = parser.parse_args()

    # Build index: clip_id -> MP4 path (one pass over clips dir)
    print("Building clip MP4 index...")
    clip_mp4_index = {}
    for root, dirs, files in os.walk(CLIPS_DIR):
        for f in files:
            if f.endswith('.mp4'):
                clip_id = f.replace('.mp4', '')
                clip_mp4_index[clip_id] = os.path.join(root, f)
    print(f"Found {len(clip_mp4_index)} clip MP4s")

    # Find clips that need re-extraction (not already 8 frames)
    needs_work = []
    already_ok = 0
    no_frame_dir = 0

    for clip_id in clip_mp4_index:
        frame_dir = os.path.join(FRAMES_DIR, clip_id)
        if os.path.exists(frame_dir):
            n_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            if n_frames == NUM_FRAMES:
                already_ok += 1
                continue
        else:
            no_frame_dir += 1
        needs_work.append(clip_id)

    print(f"Already 8 frames: {already_ok}")
    print(f"Need re-extraction: {len(needs_work)} ({no_frame_dir} have no frame dir)")

    if args.dry_run:
        print("Dry run — nothing changed.")
        return

    if not needs_work:
        print("Nothing to do!")
        return

    # Process in parallel
    print(f"\nExtracting 8 frames from {len(needs_work)} clips using {args.workers} workers...")
    success = 0
    failed = 0
    failed_clips = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for clip_id in needs_work:
            # Pass clip MP4 path directly instead of searching each time
            futures[executor.submit(process_clip_fast, clip_id, clip_mp4_index[clip_id])] = clip_id

        for i, future in enumerate(as_completed(futures), 1):
            clip_id, ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                failed_clips.append((clip_id, msg))

            if i % 200 == 0 or i == len(needs_work):
                print(f"  [{i}/{len(needs_work)}] success={success}, failed={failed}")

    print(f"\nDone. Success: {success}, Failed: {failed}")
    if failed_clips:
        print(f"\nFailed clips ({len(failed_clips)}):")
        for clip_id, msg in failed_clips[:20]:
            print(f"  {clip_id}: {msg}")
        if len(failed_clips) > 20:
            print(f"  ... and {len(failed_clips) - 20} more")


def process_clip_fast(clip_id, clip_mp4):
    """Process a single clip with known MP4 path."""
    frame_dir = os.path.join(FRAMES_DIR, clip_id)

    # Remove old frames
    if os.path.exists(frame_dir):
        for f in os.listdir(frame_dir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(frame_dir, f))

    # Extract new frames
    ok, msg = extract_8_frames(clip_mp4, frame_dir, NUM_FRAMES, JPEG_QUALITY)
    return clip_id, ok, msg


if __name__ == "__main__":
    main()
