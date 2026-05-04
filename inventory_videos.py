#!/usr/bin/env python3
"""Generate source video inventory and metadata summary.

Scans raw_data/ directory and produces a comprehensive inventory of all
source videos with metadata. Used for paper Section 3.5 (Dataset Statistics).

Also identifies Source A (curated multi-camera) vs Source B (continuous
fixed-camera) footage for temporal analysis.

Usage:
    python inventory_videos.py
    python inventory_videos.py --input-dir raw_data/ --output output/metadata/
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video_processor import get_video_info, parse_video_filename

DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "raw_data")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "metadata")


def discover_videos(input_dir):
    """Find all video files recursively."""
    videos = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".avi", ".mkv")):
                videos.append(os.path.join(root, f))
    return videos


def classify_source_type(video_path, parsed):
    """Classify video as Source A (curated) or Source B (continuous).

    Source B = continuous recordings from fixed cameras (200+ hours).
    Heuristic: if a camera has multiple consecutive recordings totaling
    significant duration, it's likely Source B.

    For now, use naming convention and folder structure:
    - Videos following {Site}_{Work Area}_main_{timestamp}.mp4 with @duration
      suffix in folder name → Source A (curated from CCTV NVR)
    - Videos with IP-address folders or long-duration continuous naming
      → Source B (continuous recording)
    """
    folder_name = os.path.basename(os.path.dirname(video_path))

    # Check for @duration suffix in folder name (NVR export marker)
    if "@" in folder_name:
        return "A"

    # Check for IP-address-style naming (continuous recording cameras)
    parts = folder_name.split("_")
    if any(part.count(".") == 3 for part in parts):
        return "B"

    # Default: Source A (curated multi-camera)
    return "A"


def analyze_video(video_path, raw_dir):
    """Extract metadata from a single video file."""
    filename = os.path.basename(video_path)
    rel_path = os.path.relpath(video_path, raw_dir)
    date_folder = rel_path.split(os.sep)[0] if os.sep in rel_path else ""

    # Parse filename
    parsed = parse_video_filename(filename, parent_dir=os.path.dirname(video_path))

    # Get video info
    try:
        info = get_video_info(video_path)
    except RuntimeError:
        info = {"fps": 0, "total_frames": 0, "duration_sec": 0,
                "width": 0, "height": 0}

    # File size
    file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 1)

    # Source type
    source_type = classify_source_type(video_path, parsed)

    return {
        "filename": filename,
        "relative_path": rel_path,
        "date_folder": date_folder,
        "site": parsed.get("site", ""),
        "work_area": parsed.get("work_area", ""),
        "camera_id": parsed.get("camera_id", ""),
        "date": parsed.get("date", ""),
        "start_timestamp": parsed.get("start_timestamp", ""),
        "fps": info["fps"],
        "total_frames": info["total_frames"],
        "duration_sec": round(info["duration_sec"], 1),
        "width": info["width"],
        "height": info["height"],
        "resolution": f"{info['width']}x{info['height']}",
        "file_size_mb": file_size_mb,
        "source_type": source_type,
    }


def generate_summary(inventory):
    """Generate summary statistics from video inventory."""
    total_duration_hrs = sum(v["duration_sec"] for v in inventory) / 3600
    total_size_gb = sum(v["file_size_mb"] for v in inventory) / 1024

    sites = Counter(v["site"] for v in inventory if v["site"])
    cameras = Counter(v["camera_id"] for v in inventory if v["camera_id"])
    resolutions = Counter(v["resolution"] for v in inventory)
    date_folders = Counter(v["date_folder"] for v in inventory)
    source_types = Counter(v["source_type"] for v in inventory)

    # Per-site breakdown
    per_site = defaultdict(lambda: {
        "video_count": 0, "duration_hours": 0, "work_areas": set(), "cameras": set()
    })
    for v in inventory:
        s = v["site"] or "unknown"
        per_site[s]["video_count"] += 1
        per_site[s]["duration_hours"] += v["duration_sec"] / 3600
        if v["work_area"]:
            per_site[s]["work_areas"].add(v["work_area"])
        if v["camera_id"]:
            per_site[s]["cameras"].add(v["camera_id"])

    per_site_serializable = {}
    for s, info in per_site.items():
        per_site_serializable[s] = {
            "video_count": info["video_count"],
            "duration_hours": round(info["duration_hours"], 2),
            "work_areas": sorted(info["work_areas"]),
            "num_work_areas": len(info["work_areas"]),
            "cameras": sorted(info["cameras"]),
            "num_cameras": len(info["cameras"]),
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "total_videos": len(inventory),
        "total_duration_hours": round(total_duration_hrs, 1),
        "total_size_gb": round(total_size_gb, 1),
        "unique_sites": len(sites),
        "unique_cameras": len(cameras),
        "sites": dict(sites.most_common()),
        "cameras": dict(cameras.most_common()),
        "resolutions": dict(resolutions.most_common()),
        "date_folders": dict(date_folders.most_common()),
        "source_types": dict(source_types),
        "source_a_count": source_types.get("A", 0),
        "source_b_count": source_types.get("B", 0),
        "fps_distribution": dict(Counter(
            round(v["fps"]) for v in inventory if v["fps"] > 0
        )),
        "per_site_breakdown": per_site_serializable,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate source video inventory and metadata"
    )
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_RAW_DIR,
        help="Raw video directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for inventory files"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Raw data directory not found: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning {args.input_dir} for videos...")
    video_paths = discover_videos(args.input_dir)
    print(f"Found {len(video_paths)} video files")

    # Analyze each video
    inventory = []
    for i, vp in enumerate(video_paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Analyzing video {i + 1}/{len(video_paths)}: "
                  f"{os.path.basename(vp)}")
        entry = analyze_video(vp, args.input_dir)
        inventory.append(entry)

    # Write inventory CSV
    csv_path = os.path.join(args.output_dir, "video_inventory.csv")
    if inventory:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
            writer.writeheader()
            writer.writerows(inventory)
        print(f"\nInventory CSV: {csv_path}")

    # Generate and write summary
    summary = generate_summary(inventory)
    summary_path = os.path.join(args.output_dir, "video_inventory_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"VIDEO INVENTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos:     {summary['total_videos']}")
    print(f"Total duration:   {summary['total_duration_hours']:.1f} hours")
    print(f"Total size:       {summary['total_size_gb']:.1f} GB")
    print(f"Unique sites:     {summary['unique_sites']}")
    print(f"Unique cameras:   {summary['unique_cameras']}")
    print(f"Source A (curated): {summary['source_a_count']}")
    print(f"Source B (continuous): {summary['source_b_count']}")
    print(f"\nSites: {summary['sites']}")
    print(f"Resolutions: {summary['resolutions']}")
    print(f"Date folders: {summary['date_folders']}")


if __name__ == "__main__":
    main()
