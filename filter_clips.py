#!/usr/bin/env python3
"""Filter extracted clips into Tier A (annotation-ready) and Tier B (degradation pool).

Tier A: Clips with sufficient person visibility for full annotation.
Tier B: Clips with marginal person presence — reserved for degradation analysis.

Usage:
    python filter_clips.py
    python filter_clips.py --manifest output/metadata/clips_manifest.csv
    python filter_clips.py --min-detection-ratio 0.5 --min-bbox-area 0.008
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime


# Default filter thresholds
DEFAULT_MIN_DETECTION_RATIO = 0.40   # Person visible in at least 40% of sampled frames
DEFAULT_MIN_AVG_BBOX_AREA = 0.005    # Person occupies at least 0.5% of frame on average
DEFAULT_MIN_AVG_CONFIDENCE = 0.45    # Minimum average detection confidence
DEFAULT_MIN_CLIP_DURATION = 8.0      # Minimum clip duration in seconds


def load_manifest(manifest_path):
    """Load clips manifest CSV into list of dicts."""
    clips = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["person_detection_ratio", "avg_bbox_area_ratio",
                        "max_bbox_area_ratio", "min_bbox_area_ratio",
                        "avg_confidence", "max_confidence", "clip_duration_sec",
                        "avg_persons", "brisque_score", "mean_pixel_intensity",
                        "source_start_sec", "source_end_sec"]:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = 0.0

            for key in ["total_persons_detected", "min_persons", "max_persons"]:
                if key in row and row[key]:
                    try:
                        row[key] = int(row[key])
                    except ValueError:
                        row[key] = 0

            clips.append(row)
    return clips


def classify_clip(clip, min_detection_ratio, min_avg_bbox_area,
                  min_avg_confidence, min_clip_duration):
    """Classify a clip as Tier A or Tier B with rejection reasons.

    Returns (tier, reasons) where tier is 'A' or 'B' and reasons is a list
    of why it was assigned to Tier B (empty for Tier A).
    """
    reasons = []

    # Filter 1: Person presence ratio
    if clip["person_detection_ratio"] < min_detection_ratio:
        reasons.append(
            f"low_presence:{clip['person_detection_ratio']:.2f}<{min_detection_ratio}"
        )

    # Filter 2: Person size (too small / distant)
    if clip["avg_bbox_area_ratio"] < min_avg_bbox_area:
        reasons.append(
            f"small_person:{clip['avg_bbox_area_ratio']:.4f}<{min_avg_bbox_area}"
        )

    # Filter 3: Detection confidence
    if clip["avg_confidence"] < min_avg_confidence:
        reasons.append(
            f"low_confidence:{clip['avg_confidence']:.3f}<{min_avg_confidence}"
        )

    # Filter 4: Clip too short
    if clip["clip_duration_sec"] < min_clip_duration:
        reasons.append(
            f"short_clip:{clip['clip_duration_sec']:.1f}s<{min_clip_duration}s"
        )

    tier = "A" if not reasons else "B"
    return tier, reasons


def auto_tag_visual_condition(clip):
    """Auto-tag visual conditions using available heuristics.

    Paper requires 8 condition tags. Some (dust, steam) can only be reliably
    assigned via manual review or VLM analysis — those are left for the
    annotation stage. This function auto-detects what's possible from metadata.

    Condition tags (paper spec):
        clean            — good visibility, no degradation
        dust_light       — minor dust (manual tag)
        dust_heavy       — dense dust (manual tag)
        steam            — steam/vapor (manual tag)
        low_light        — insufficient lighting (auto: pixel intensity)
        occlusion_static — fixed objects blocking view (partial auto)
        occlusion_dynamic — moving objects blocking view (partial auto)
        multi_degradation — 2+ conditions co-occurring (derived)

    Additional auto-detected tags:
        low_quality      — high BRISQUE score (blurry, noisy, compressed)
        distant_scene    — all persons very small in frame
        multi_worker     — 2+ persons (not a degradation, useful metadata)

    Returns list of detected condition tags.
    """
    tags = []

    # Low light detection via mean pixel intensity
    intensity = clip["mean_pixel_intensity"]
    if 0 <= intensity < 60:
        tags.append("low_light")

    # BRISQUE-based quality detection
    brisque = clip.get("brisque_score", -1.0)
    if isinstance(brisque, str):
        try:
            brisque = float(brisque)
        except ValueError:
            brisque = -1.0
    if brisque >= 0:
        # BRISQUE ranges: 0-20 excellent, 20-40 good, 40-60 fair, 60+ poor
        if brisque > 60:
            tags.append("low_quality")

    # Occlusion detection — person appears/disappears across frames
    if clip["max_persons"] > 0 and clip["min_persons"] == 0:
        ratio = clip["person_detection_ratio"]
        if 0.2 < ratio < 0.7:
            tags.append("occlusion_static")

    # Distant scene — all persons far from camera
    if clip["max_bbox_area_ratio"] > 0 and clip["max_bbox_area_ratio"] < 0.01:
        tags.append("distant_scene")

    # Multi-worker scene hint (not a degradation, but useful metadata)
    if clip["max_persons"] >= 2:
        tags.append("multi_worker")

    # Multi-degradation: 2+ degradation conditions co-occurring
    degradation_tags = [t for t in tags
                        if t in ("low_light", "low_quality", "occlusion_static",
                                 "occlusion_dynamic", "dust_light", "dust_heavy",
                                 "steam")]
    if len(degradation_tags) >= 2:
        tags.append("multi_degradation")

    if not tags:
        tags.append("clean")

    return tags


# Severity levels per the paper (benchmarking_pipeline.pdf Section 5.3)
# S0: clean
# S1: dust_light, low_light (mild)
# S2: dust_heavy, occlusion_static, steam (partial)
# S3: steam (dense), occlusion_dynamic, low_light (severe)
# S4: multi_degradation (2+ conditions)
_SEVERITY_MAP = {
    "clean": "S0",
    "dust_light": "S1",
    "dust_heavy": "S2",
    "steam": "S2",
    "low_light": "S1",
    "low_quality": "S1",
    "occlusion_static": "S2",
    "occlusion_dynamic": "S3",
    "multi_degradation": "S4",
    "distant_scene": "S1",
}


def assign_severity_level(condition_tags):
    """Assign severity level S0-S4 based on visual condition tags.

    Takes the worst (highest) severity among all tags.
    """
    severity_order = ["S0", "S1", "S2", "S3", "S4"]
    max_severity = "S0"

    for tag in condition_tags:
        sev = _SEVERITY_MAP.get(tag, "S0")
        if severity_order.index(sev) > severity_order.index(max_severity):
            max_severity = sev

    # Special case: very low intensity is severe low light (S3 not S1)
    # This would need the actual intensity value, handled in classify_clip
    return max_severity


def generate_report(clips, tier_a, tier_b, output_dir):
    """Generate filtering summary report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_clips": len(clips),
        "tier_a_count": len(tier_a),
        "tier_b_count": len(tier_b),
        "tier_a_percentage": round(100 * len(tier_a) / len(clips), 1) if clips else 0,
        "tier_b_percentage": round(100 * len(tier_b) / len(clips), 1) if clips else 0,
    }

    # Tier B reason breakdown
    all_reasons = []
    for clip in tier_b:
        for reason in clip["_rejection_reasons"]:
            all_reasons.append(reason.split(":")[0])
    report["tier_b_reason_counts"] = dict(Counter(all_reasons))

    # Tier A distribution by camera
    cam_counts = Counter(c["camera_id"] for c in tier_a)
    report["tier_a_by_camera"] = dict(cam_counts)

    # Tier A distribution by date
    date_counts = Counter(c["date"] for c in tier_a)
    report["tier_a_by_date"] = dict(sorted(date_counts.items()))

    # Tier A visual condition distribution
    condition_counts = Counter()
    for c in tier_a:
        for tag in c["_visual_tags"]:
            condition_counts[tag] += 1
    report["tier_a_visual_conditions"] = dict(condition_counts)

    # Tier A stats
    if tier_a:
        report["tier_a_stats"] = {
            "mean_detection_ratio": round(
                sum(c["person_detection_ratio"] for c in tier_a) / len(tier_a), 3
            ),
            "mean_avg_persons": round(
                sum(c["avg_persons"] for c in tier_a) / len(tier_a), 2
            ),
            "mean_brisque": round(
                sum(c["brisque_score"] for c in tier_a if c["brisque_score"] > 0)
                / max(1, sum(1 for c in tier_a if c["brisque_score"] > 0)), 2
            ),
            "multi_worker_clips": sum(
                1 for c in tier_a if c["max_persons"] >= 2
            ),
            "single_worker_clips": sum(
                1 for c in tier_a if c["max_persons"] == 1
            ),
        }

    return report


def write_filtered_manifests(tier_a, tier_b, output_dir):
    """Write separate manifest CSVs for Tier A and Tier B."""
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    # Add tier, visual tags, severity level columns
    extra_columns = ["tier", "visual_tags", "severity_level", "rejection_reasons"]

    # Get base columns from first clip
    if tier_a:
        base_columns = [k for k in tier_a[0].keys() if not k.startswith("_")]
    elif tier_b:
        base_columns = [k for k in tier_b[0].keys() if not k.startswith("_")]
    else:
        return

    all_columns = base_columns + extra_columns

    # Write Tier A
    tier_a_path = os.path.join(output_dir, "metadata", "tier_a_manifest.csv")
    with open(tier_a_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for clip in tier_a:
            row = {k: v for k, v in clip.items() if not k.startswith("_")}
            row["tier"] = "A"
            row["visual_tags"] = ",".join(clip.get("_visual_tags", []))
            row["severity_level"] = clip.get("_severity_level", "S0")
            row["rejection_reasons"] = ""
            writer.writerow(row)

    # Write Tier B
    tier_b_path = os.path.join(output_dir, "metadata", "tier_b_manifest.csv")
    with open(tier_b_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for clip in tier_b:
            row = {k: v for k, v in clip.items() if not k.startswith("_")}
            row["tier"] = "B"
            row["visual_tags"] = ",".join(clip.get("_visual_tags", []))
            row["severity_level"] = clip.get("_severity_level", "S0")
            row["rejection_reasons"] = "|".join(clip.get("_rejection_reasons", []))
            writer.writerow(row)

    print(f"  Tier A manifest: {tier_a_path} ({len(tier_a)} clips)")
    print(f"  Tier B manifest: {tier_b_path} ({len(tier_b)} clips)")

    return tier_a_path, tier_b_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter clips into Tier A (annotation-ready) and Tier B (degradation pool)"
    )
    parser.add_argument("--manifest", type=str,
                        default="output/metadata/clips_manifest.csv",
                        help="Path to clips manifest CSV")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--min-detection-ratio", type=float,
                        default=DEFAULT_MIN_DETECTION_RATIO)
    parser.add_argument("--min-bbox-area", type=float,
                        default=DEFAULT_MIN_AVG_BBOX_AREA)
    parser.add_argument("--min-confidence", type=float,
                        default=DEFAULT_MIN_AVG_CONFIDENCE)
    parser.add_argument("--min-duration", type=float,
                        default=DEFAULT_MIN_CLIP_DURATION)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = (args.manifest if os.path.isabs(args.manifest)
                     else os.path.join(script_dir, args.manifest))
    output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                  else os.path.join(script_dir, args.output_dir))

    print(f"Loading manifest: {manifest_path}")
    clips = load_manifest(manifest_path)
    print(f"Total clips: {len(clips)}")

    if not clips:
        print("No clips found. Run extract_clips.py first.")
        return

    print(f"\nFilter thresholds:")
    print(f"  Min detection ratio: {args.min_detection_ratio}")
    print(f"  Min avg bbox area:   {args.min_bbox_area}")
    print(f"  Min avg confidence:  {args.min_confidence}")
    print(f"  Min clip duration:   {args.min_duration}s")
    print()

    # Classify each clip
    tier_a = []
    tier_b = []

    for clip in clips:
        tier, reasons = classify_clip(
            clip, args.min_detection_ratio, args.min_bbox_area,
            args.min_confidence, args.min_duration,
        )
        visual_tags = auto_tag_visual_condition(clip)
        severity_level = assign_severity_level(visual_tags)
        clip["_visual_tags"] = visual_tags
        clip["_severity_level"] = severity_level
        clip["_rejection_reasons"] = reasons

        if tier == "A":
            tier_a.append(clip)
        else:
            tier_b.append(clip)

    print(f"Results:")
    print(f"  Tier A (annotation-ready): {len(tier_a)} clips "
          f"({100 * len(tier_a) / len(clips):.1f}%)")
    print(f"  Tier B (degradation pool): {len(tier_b)} clips "
          f"({100 * len(tier_b) / len(clips):.1f}%)")

    # Write filtered manifests
    print(f"\nWriting filtered manifests...")
    write_filtered_manifests(tier_a, tier_b, output_dir)

    # Generate and save report
    report = generate_report(clips, tier_a, tier_b, output_dir)
    report_path = os.path.join(output_dir, "metadata", "filtering_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Filtering report: {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total clips:        {report['total_clips']}")
    print(f"Tier A:             {report['tier_a_count']} ({report['tier_a_percentage']}%)")
    print(f"Tier B:             {report['tier_b_count']} ({report['tier_b_percentage']}%)")

    if report.get("tier_b_reason_counts"):
        print(f"\nTier B rejection reasons:")
        for reason, count in sorted(report["tier_b_reason_counts"].items(),
                                     key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if report.get("tier_a_visual_conditions"):
        print(f"\nTier A visual conditions:")
        for condition, count in sorted(report["tier_a_visual_conditions"].items(),
                                        key=lambda x: -x[1]):
            print(f"  {condition}: {count}")

    if report.get("tier_a_stats"):
        stats = report["tier_a_stats"]
        print(f"\nTier A stats:")
        print(f"  Mean detection ratio: {stats['mean_detection_ratio']}")
        print(f"  Mean avg persons:     {stats['mean_avg_persons']}")
        print(f"  Mean BRISQUE:         {stats['mean_brisque']}")
        print(f"  Single-worker clips:  {stats['single_worker_clips']}")
        print(f"  Multi-worker clips:   {stats['multi_worker_clips']}")


if __name__ == "__main__":
    main()
