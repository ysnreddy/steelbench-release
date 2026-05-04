#!/usr/bin/env python3
"""Run a SteelBench annotation experiment from a YAML config.

Single entry point that orchestrates batch curation, safety-critical
identification, role-based assignment, and active_batch generation.

Usage:
    python run_experiment.py --config experiments/pilot_v2.yaml
    python run_experiment.py --config experiments/pilot_v2.yaml --dry-run
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = str(Path(__file__).parent)
sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.experiment_config import (
    load_experiment,
    get_tier1_annotators,
    get_tier2_annotators,
    get_tier3_annotators,
    get_safety_zones,
    get_clip_targets,
    validate_experiment_dir,
)


def load_camera_zones():
    """Load camera-to-zone mapping."""
    zones_path = os.path.join(
        PROJECT_ROOT, "annotation_tool", "config", "camera_zones.yaml"
    )
    if not os.path.exists(zones_path):
        return {}
    with open(zones_path) as f:
        data = yaml.safe_load(f)
    # Build camera_id -> [zones] mapping
    camera_zones = {}
    for cam_id, cam_info in data.get("cameras", {}).items():
        camera_zones[cam_id] = cam_info.get("zones", [])
    return camera_zones


def identify_safety_clips(clips, safety_zones, camera_zones):
    """Identify clips from safety-relevant camera zones.

    Returns set of clip_ids that are safety-critical.
    """
    if not safety_zones:
        return set()

    safety_zone_set = set(safety_zones)
    critical_clips = set()

    for clip in clips:
        cam_id = clip.get("camera_id", "")
        cam_zones = set(camera_zones.get(cam_id, []))
        if cam_zones & safety_zone_set:
            critical_clips.add(clip.get("clip_id", ""))

    return critical_clips


def run_experiment(config, dry_run=False):
    """Execute a full experiment setup from config."""

    print(f"Experiment: {config['experiment_name']}")
    print(f"Description: {config.get('description', '')}")
    print(f"Config: {config.get('_config_path', '')}")
    print()

    # Validate
    issues = validate_experiment_dir(config)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not dry_run:
            print("Fix these before running. Use --dry-run to preview anyway.")
            return

    # --- Step 1: Curate batch ---
    print("=" * 60)
    print("Step 1: Curate Batch")
    print("=" * 60)

    from curate_batch import (
        load_tier_a_manifest,
        load_vlm_annotations,
        load_existing_annotations,
        get_class_distribution,
        curate_balanced_batch,
        build_active_batch,
        print_distribution_report,
    )

    clips = load_tier_a_manifest()
    vlm_actions = load_vlm_annotations()
    annotated = load_existing_annotations()
    existing_dist = get_class_distribution(annotated)

    cs = config["clip_selection"]
    print(f"Tier A clips: {len(clips)}")
    print(f"VLM-classified: {len(vlm_actions)}")
    print(f"Already annotated: {len(annotated)}")
    print(f"Batch size: {cs['size']}")
    print(f"Explicit targets: {cs.get('targets', {})}")
    print(f"Min per class: {cs.get('min_per_class', 0)}")
    print(f"Unclassified budget: {cs.get('unclassified_budget', 0)}")

    selected = curate_balanced_batch(
        clips, vlm_actions, existing_dist, cs["size"],
        prioritize=list(cs.get("targets", {}).keys()) or None,
        exclude_annotated=cs.get("exclude_annotated", True),
        annotated_ids=set(annotated.keys()),
        targets=cs.get("targets", {}),
        min_per_class=cs.get("min_per_class", 0),
        unclassified_budget=cs.get("unclassified_budget", 0),
    )

    print(f"\nSelected {len(selected)} clips")
    print_distribution_report(selected, vlm_actions, existing_dist)

    if not dry_run:
        batch_info = build_active_batch(selected, config["experiment_name"])
        print(f"\nActive batch built: {batch_info['copied_clips']} clips, "
              f"{batch_info['copied_frames']} frames")

    # --- Step 2: Identify safety-critical clips ---
    print(f"\n{'=' * 60}")
    print("Step 2: Safety-Critical Clip Identification")
    print("=" * 60)

    safety_zones = get_safety_zones(config)
    camera_zones = load_camera_zones()

    safety_clips = identify_safety_clips(
        selected, safety_zones, camera_zones
    )
    print(f"Safety zones monitored: {safety_zones}")
    print(f"Safety-critical clips: {len(safety_clips)} / {len(selected)}")

    # --- Step 3: Role-based assignment ---
    print(f"\n{'=' * 60}")
    print("Step 3: Role-Based Assignment")
    print("=" * 60)

    tier1 = get_tier1_annotators(config)
    tier2 = get_tier2_annotators(config)
    tier3 = get_tier3_annotators(config)

    print(f"Tier 1 (annotation): {tier1}")
    print(f"Tier 2 (adjudication): {tier2}")
    print(f"Tier 3 (safety): {tier3}")

    # Load VLM results for routing
    from annotation_tool.routing import route_batch

    vlm_results_path = os.path.join(
        PROJECT_ROOT, "annotation_tool", "data", "vlm_results",
        "vlm_annotations.jsonl"
    )
    vlm_by_clip = {}
    if os.path.exists(vlm_results_path):
        with open(vlm_results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    vlm_by_clip[r["clip_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue

    # Build agreements for routing
    agreements = []
    for clip in selected:
        cid = clip.get("clip_id", "")
        vlm = vlm_by_clip.get(cid)
        if vlm and vlm.get("vlm", {}).get("success") and vlm.get("agreement"):
            composite = vlm["agreement"].get("composite", 0.0)
        else:
            composite = 0.0
        agreements.append({"clip_id": cid, "composite": composite})

    routing = route_batch(agreements)
    routed = routing["routed_clips"]
    summary = routing["summary"]

    print(f"\nRouting: High={summary['high_confidence']['count']}, "
          f"Medium={summary['medium_confidence']['count']}, "
          f"Low={summary['low_confidence']['count']}")

    # Assign to tier_1 annotators (round-robin with double-annotation)
    import random
    double_ratio = config["assignment"]["double_annotate_ratio"]

    all_tier1_clips = (
        routed["low_confidence"] + routed["medium_confidence"] +
        routed["high_confidence"]
    )
    random.shuffle(all_tier1_clips)

    assignments = {}

    # Tier 1 assignments
    for aid in tier1:
        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_1",
            "clips": [],
        }

    for i, clip in enumerate(all_tier1_clips):
        cid = clip["clip_id"]
        route = clip.get("route", "low_confidence")
        task = clip.get("human_effort", "fresh_annotation")

        # Primary annotator
        primary = tier1[i % len(tier1)]
        assignments[primary]["clips"].append({
            "clip_id": cid,
            "route": route,
            "task": task,
        })

        # Double-annotation for IAA
        if random.random() < double_ratio:
            secondary = tier1[(i + 1) % len(tier1)]
            assignments[secondary]["clips"].append({
                "clip_id": cid,
                "route": route,
                "task": task,
                "is_double_annotation": True,
            })

    # Tier 2 — starts empty (fills dynamically as tier_1 flags/disagrees)
    for aid in tier2:
        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_2",
            "clips": [],
            "note": "Queue populated dynamically from flagged clips and disagreements",
        }

    # Tier 3 — all safety-critical clips
    for aid in tier3:
        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_3",
            "clips": [
                {"clip_id": cid, "route": "safety_review", "task": "safety_verification"}
                for cid in safety_clips
            ],
        }

    # Print summary
    for aid, data in assignments.items():
        print(f"  {aid} ({data['role']}): {len(data['clips'])} clips")

    # --- Step 4: Save everything ---
    if not dry_run:
        print(f"\n{'=' * 60}")
        print("Step 4: Save Assignments")
        print("=" * 60)

        assign_dir = os.path.join(
            PROJECT_ROOT, "annotation_tool", "data", "assignments"
        )
        os.makedirs(assign_dir, exist_ok=True)

        output = {
            "generated_at": datetime.now().isoformat(),
            "experiment_name": config["experiment_name"],
            "experiment_config": config.get("_config_path", ""),
            "routing_summary": summary,
            "safety_critical_clips": list(safety_clips),
            "assignments": assignments,
        }

        # Save master JSON
        json_path = os.path.join(assign_dir, "assignments.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {json_path}")

        # Save per-annotator CSVs
        for aid, data in assignments.items():
            csv_path = os.path.join(assign_dir, f"{aid}_clips.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["clip_id", "route", "task", "role"])
                for clip in data["clips"]:
                    writer.writerow([
                        clip["clip_id"], clip.get("route", ""),
                        clip.get("task", ""), data["role"],
                    ])
            print(f"Saved: {csv_path}")

        # Save experiment config copy for reproducibility
        config_copy_path = os.path.join(assign_dir, "experiment_config.yaml")
        with open(config_copy_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved: {config_copy_path}")

    print(f"\n{'=' * 60}")
    print("EXPERIMENT SETUP COMPLETE" if not dry_run else "DRY RUN COMPLETE")
    print("=" * 60)

    if not dry_run:
        print(f"\nNext steps:")
        print(f"  1. Sync to VPS:  ./sync_vps.sh push")
        print(f"  2. Annotators login at: https://<ANNOTATOR_HOST>")
        print(f"  3. Monitor at: https://<DASHBOARD_HOST>")


def main():
    parser = argparse.ArgumentParser(
        description="Run a SteelBench annotation experiment"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()

    config = load_experiment(args.config)
    run_experiment(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
