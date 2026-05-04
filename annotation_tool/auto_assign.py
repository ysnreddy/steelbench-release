#!/usr/bin/env python3
"""Auto-assignment of clips to human annotators based on VLM agreement routing.

Supports role-based assignment (tier_1/tier_2/tier_3) when an experiment config
is provided, or simple round-robin when used standalone.

Usage:
    # With experiment config (recommended)
    python -m annotation_tool.auto_assign --config experiments/pilot_v2.yaml

    # Standalone (backward compatible)
    python -m annotation_tool.auto_assign --annotators 3
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.routing import route_batch, select_verification_sample

# ---------- Paths ----------
OUTPUT_DIR = os.environ.get(
    "STEELBENCH_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
VLM_RESULTS_DIR = os.environ.get(
    "STEELBENCH_VLM_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results"))
ASSIGNMENTS_DIR = os.environ.get(
    "STEELBENCH_ASSIGNMENTS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "assignments"))
TIER_A_MANIFEST = os.environ.get(
    "STEELBENCH_MANIFEST",
    os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv"))


def load_vlm_results(results_path=None):
    """Load VLM annotation results from JSONL."""
    if results_path is None:
        results_path = os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    results = []
    if not os.path.exists(results_path):
        return results
    with open(results_path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def load_manifest_metadata(manifest_path=None):
    """Load clip metadata from Tier A manifest."""
    if manifest_path is None:
        manifest_path = TIER_A_MANIFEST
    metadata = {}
    if not os.path.exists(manifest_path):
        return metadata
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row.get("clip_id", "")] = row
    return metadata


def load_camera_zones():
    """Load camera-to-zone mapping for safety-critical identification."""
    zones_path = os.path.join(
        PROJECT_ROOT, "annotation_tool", "config", "camera_zones.yaml"
    )
    if not os.path.exists(zones_path):
        return {}
    with open(zones_path) as f:
        data = yaml.safe_load(f)
    camera_zones = {}
    for cam_id, cam_info in data.get("cameras", {}).items():
        camera_zones[cam_id] = cam_info.get("zones", [])
    return camera_zones


def build_clip_agreements(vlm_results):
    """Extract clip_id and routing info from VLM results.

    With single-VLM policy, all clips get full human review (VLM is
    prefill only, not a decision-maker). VLM confidence is preserved
    for prioritization — low-confidence clips go to tier_2 experts.
    """
    agreements = []
    for r in vlm_results:
        clip_id = r.get("clip_id", "")
        vlm = r.get("vlm") or {}
        norm = vlm.get("normalized") or {}
        persons = norm.get("persons", [])
        n_workers = len(persons)
        annotation_layer = 2 if n_workers <= 5 else 1

        # Use mean VLM confidence for prioritization (not routing)
        confidences = [p.get("confidence", 0.0) for p in persons
                       if p.get("confidence") is not None]
        mean_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

        agreements.append({
            "clip_id": clip_id,
            "composite": 0.0,  # No inter-model agreement with single VLM
            "vlm_confidence": round(mean_confidence, 3),
            "vlm_success": vlm.get("success", False),
            "annotation_layer": annotation_layer,
        })
    return agreements


def build_prefill_data(vlm_result):
    """Extract VLM pre-fill data for the annotation UI (single VLM)."""
    vlm = vlm_result.get("vlm") or {}
    norm = vlm.get("normalized") if vlm.get("success") else None

    if not norm:
        return None

    return {
        "source": vlm.get("model", "unknown"),
        "annotation": norm,
    }


def identify_safety_clips(clip_ids_with_cameras, safety_zones, camera_zones):
    """Identify clips from safety-relevant camera zones.

    Args:
        clip_ids_with_cameras: list of (clip_id, camera_id) tuples
        safety_zones: list of zone names to flag
        camera_zones: dict {camera_id: [zone_names]}

    Returns:
        set of clip_ids that are safety-critical
    """
    if not safety_zones:
        return set()
    zone_set = set(safety_zones)
    critical = set()
    for clip_id, camera_id in clip_ids_with_cameras:
        cam_zones = set(camera_zones.get(camera_id, []))
        if cam_zones & zone_set:
            critical.add(clip_id)
    return critical


def assign_role_based(routed_clips, experiment_config, manifest_metadata,
                      vlm_by_clip):
    """Role-based assignment using experiment config.

    - tier_1: Normal clips (round-robin with double-annotation for IAA)
    - tier_2: Starts empty (filled dynamically from flags/disagreements)
    - tier_3: All safety-critical clips
    """
    from annotation_tool.experiment_config import (
        get_tier1_annotators, get_tier2_annotators, get_tier3_annotators,
        get_safety_zones,
    )
    from annotation_tool.calibration import CalibrationManager

    tier1_ids = get_tier1_annotators(experiment_config)
    tier2_ids = get_tier2_annotators(experiment_config)
    tier3_ids = get_tier3_annotators(experiment_config)
    safety_zones = get_safety_zones(experiment_config)
    double_ratio = experiment_config.get("assignment", {}).get(
        "double_annotate_ratio", 0.15)
    experts_only = experiment_config.get("assignment", {}).get(
        "experts_only", False)

    # Calibration manager for anchored/blind condition lookup (Phase 2 anchoring bias)
    cal_dir = experiment_config.get("calibration", {}).get(
        "calibration_dir",
        os.path.join(PROJECT_ROOT, "active_batch", "data"),
    )
    cal_mgr = CalibrationManager(cal_dir)

    # Identify safety-critical clips
    camera_zones = load_camera_zones()
    clip_cameras = []
    all_clips = (
        routed_clips.get("low_confidence", []) +
        routed_clips.get("medium_confidence", []) +
        routed_clips.get("high_confidence", [])
    )
    for clip in all_clips:
        cid = clip["clip_id"]
        meta = manifest_metadata.get(cid, {})
        cam_id = meta.get("camera_id", "")
        clip_cameras.append((cid, cam_id))

    safety_clip_ids = identify_safety_clips(clip_cameras, safety_zones, camera_zones)

    # Build assignments
    assignments = {}

    # Experts-only mode (Phase 1.5 GT batch): skip tier_1 / tier_3 entirely
    if experts_only:
        tier1_ids = []
        tier3_ids = []

    # Tier 1 — round-robin with double-annotation
    for aid in tier1_ids:
        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_1",
            "total_clips": 0,
            "clips": [],
        }

    random.shuffle(all_clips)
    for i, clip in enumerate(all_clips):
        cid = clip["clip_id"]
        route = clip.get("route", "low_confidence")

        if route == "low_confidence":
            task = "fresh_annotation"
        elif route == "medium_confidence":
            task = "review_prefill"
        else:
            task = "verify_consensus"

        cal_condition = cal_mgr.get_calibration_condition(cid)

        prefill = None
        if cid in vlm_by_clip:
            prefill = build_prefill_data(vlm_by_clip[cid])

        # Strip prefill server-side for blind calibration clips so it doesn't
        # leak into per-annotator CSVs (defense in depth — app.py also enforces)
        if cal_condition == "blind":
            prefill = None

        entry = {
            "clip_id": cid,
            "route": route,
            "task": task,
            "prefill": prefill,
            "is_safety_critical": cid in safety_clip_ids,
            "calibration_condition": cal_condition,
        }

        # Skip tier_1 assignment if no tier_1 annotators (e.g., experts_only mode)
        if not tier1_ids:
            continue

        # Primary annotator
        primary = tier1_ids[i % len(tier1_ids)]
        assignments[primary]["clips"].append(entry)

        # Double-annotation for IAA
        if random.random() < double_ratio:
            secondary = tier1_ids[(i + 1) % len(tier1_ids)]
            assignments[secondary]["clips"].append({
                **entry,
                "is_double_annotation": True,
            })

    # Update counts
    for aid in tier1_ids:
        assignments[aid]["total_clips"] = len(assignments[aid]["clips"])

    # Tier 2 — experts get:
    # 1. EXPERTS_ONLY mode (Phase 1.5 GT batch): both experts see ALL clips
    # 2. Normal mode: ALL low+med conf (adjudication) + 15% high-conf audit
    # 3. Dynamically flagged clips during tier_1 annotation (handled at runtime)
    if experts_only:
        # Both experts see ALL clips — gold-standard double annotation
        for aid in tier2_ids:
            expert_entries = []
            for clip in all_clips:
                cid = clip["clip_id"]
                prefill = None
                if cid in vlm_by_clip:
                    prefill = build_prefill_data(vlm_by_clip[cid])
                expert_entries.append({
                    "clip_id": cid,
                    "route": clip.get("route", "low_confidence"),
                    "task": "expert_gold",
                    "prefill": prefill,
                    "is_safety_critical": cid in safety_clip_ids,
                    "is_gold": True,
                    "calibration_condition": cal_mgr.get_calibration_condition(cid),
                })
            assignments[aid] = {
                "annotator_id": aid,
                "role": "tier_2",
                "total_clips": len(expert_entries),
                "clips": expert_entries,
                "note": "GT batch — gold-standard double annotation (both experts see all clips).",
            }
        print(f"Experts-only mode: {len(tier2_ids)} experts × {len(all_clips)} clips each")
    else:
        # AUDIT-ONLY expert routing (Phase 2 onward):
        # Tier_1 handles ALL clips as primary. Experts get:
        #   - tier2_audit_ratio  (default 0.25) × high-conf clips
        #   - tier2_medconf_audit_ratio (default 0.15) × medium-conf clips
        #   - tier2_lowconf_audit_ratio (default 0.15) × low-conf clips
        #   (+ dynamic IAA-disagreement escalations handled in app.py at save time)
        low_conf_clips = [c for c in all_clips if c.get("route") == "low_confidence"]
        med_conf_clips = [c for c in all_clips if c.get("route") == "medium_confidence"]
        high_conf_clips = [c for c in all_clips if c.get("route") == "high_confidence"]

        assign_cfg = experiment_config.get("assignment", {})
        audit_ratio = assign_cfg.get("tier2_audit_ratio", 0.25)
        med_ratio = assign_cfg.get("tier2_medconf_audit_ratio", 0.15)
        low_ratio = assign_cfg.get("tier2_lowconf_audit_ratio", 0.15)

        def _sample(pool, ratio, floor=10):
            shuffled = pool[:]
            random.shuffle(shuffled)
            n = max(floor, int(len(pool) * ratio)) if pool else 0
            return shuffled[:n]

        audit_high = _sample(high_conf_clips, audit_ratio, floor=20)
        audit_med = _sample(med_conf_clips, med_ratio, floor=10)
        audit_low = _sample(low_conf_clips, low_ratio, floor=10)
        audit_clip_ids = {c["clip_id"] for c in audit_high + audit_med + audit_low}

        expert_pool = audit_high + audit_med + audit_low
        random.shuffle(expert_pool)
        print(f'Tier 2 AUDIT-ONLY pool: {len(audit_high)} high-audit + {len(audit_med)} med-audit + '
              f'{len(audit_low)} low-audit = {len(expert_pool)} total '
              f'(~{len(expert_pool)//max(len(tier2_ids),1)} per expert; '
              'all 2,709 clips still primary-annotated by tier_1)')

        for i, aid in enumerate(tier2_ids):
            # Round-robin split between experts
            expert_clip_list = [c for j, c in enumerate(expert_pool)
                               if j % len(tier2_ids) == i]

            expert_entries = []
            for clip in expert_clip_list:
                cid = clip["clip_id"]
                prefill = None
                if cid in vlm_by_clip:
                    prefill = build_prefill_data(vlm_by_clip[cid])
                expert_entries.append({
                    "clip_id": cid,
                    "route": clip.get("route", "low_confidence"),
                    "task": "audit",
                    "prefill": prefill,
                    "is_safety_critical": cid in safety_clip_ids,
                    "is_audit": True,
                    "calibration_condition": cal_mgr.get_calibration_condition(cid),
                })

            assignments[aid] = {
                "annotator_id": aid,
                "role": "tier_2",
                "total_clips": len(expert_entries),
                "clips": expert_entries,
                "note": ("Audit-only: samples of high/med/low-conf + dynamic escalations from "
                         "tier_1 IAA disagreements and unsafe-act flags."),
            }

    # Tier 3 — safety officer
    # If safety_officer_reviews_all: safety officer sees ALL clips for safety field
    # Otherwise: only safety-critical clips from flagged zones
    reviews_all = experiment_config.get("safety", {}).get(
        "safety_officer_reviews_all", False)

    for aid in tier3_ids:
        if reviews_all:
            # Safety officer reviews ALL clips — marks safety field for each
            # VLM-flagged unsafe clips are marked as priority
            safety_entries = []
            for clip in all_clips:
                cid = clip["clip_id"]
                is_vlm_unsafe = False
                if cid in vlm_by_clip:
                    vlm_data = vlm_by_clip[cid]
                    vlm_info = vlm_data.get("vlm") or {}
                    norm = vlm_info.get("normalized") or {}
                    for person in norm.get("persons", []):
                        ua = (person.get("unsafe_act") or "").strip().lower()
                        if ua and ua != "none":
                            is_vlm_unsafe = True
                            break

                safety_entries.append({
                    "clip_id": cid,
                    "route": "safety_review",
                    "task": "safety_verification",
                    "is_safety_critical": cid in safety_clip_ids,
                    "vlm_flagged_unsafe": is_vlm_unsafe,
                    "priority": "high" if is_vlm_unsafe or cid in safety_clip_ids else "normal",
                })

            # Sort: high priority first
            safety_entries.sort(key=lambda x: (0 if x["priority"] == "high" else 1))
        else:
            # AUDIT-ONLY safety officer queue (Phase 2 onward).
            # Sample SO_AUDIT_RATIO (default 0.10) of clips where the VLM cited any
            # unsafe_act != "none", stratified by site so every department is audited.
            # Zone-flagged clips (from auto_flag_zones) always included regardless of VLM.
            so_audit_ratio = experiment_config.get("safety", {}).get(
                "safety_officer_random_sample_ratio", 0.10)

            # Build pool of VLM-safety-flagged clips, grouped by site
            from collections import defaultdict as _dd
            flagged_by_site = _dd(list)
            for clip in all_clips:
                cid = clip["clip_id"]
                site = clip.get("site") or "unknown"
                has_vlm_unsafe = False
                if cid in vlm_by_clip:
                    norm = (vlm_by_clip[cid].get("vlm") or {}).get("normalized") or {}
                    for person in norm.get("persons", []):
                        ua = (person.get("unsafe_act") or "").strip().lower()
                        if ua and ua != "none" and ua != "none_visible":
                            has_vlm_unsafe = True
                            break
                if has_vlm_unsafe:
                    flagged_by_site[site].append(cid)

            sampled_ids = set()
            for site, cids in flagged_by_site.items():
                random.shuffle(cids)
                n = max(2, int(len(cids) * so_audit_ratio)) if cids else 0
                for c in cids[:n]:
                    sampled_ids.add(c)

            # Union with zone-flagged critical clips (always audited)
            audit_ids = sampled_ids | set(safety_clip_ids)

            safety_entries = [
                {
                    "clip_id": cid,
                    "route": "safety_review",
                    "task": "safety_audit",
                    "is_safety_critical": cid in safety_clip_ids,
                    "vlm_flagged_unsafe": cid in sampled_ids,
                }
                for cid in sorted(audit_ids)
            ]
            print(f"Tier 3 SO audit queue: {len(sampled_ids)} VLM-unsafe-sampled "
                  f"+ {len(safety_clip_ids)} zone-flagged = {len(safety_entries)} total "
                  f"(audit-only; dynamic additions via tier3_queue.json at save time)")

        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_3",
            "total_clips": len(safety_entries),
            "clips": safety_entries,
            "reviews_all_clips": reviews_all,
        }

    return assignments, safety_clip_ids


def assign_simple(routed_clips, num_annotators=3, annotator_ids=None,
                  vlm_by_clip=None):
    """Simple round-robin assignment (backward compatible, no roles)."""
    if annotator_ids is None:
        annotator_ids = [f"annotator_{i+1}" for i in range(num_annotators)]
    if vlm_by_clip is None:
        vlm_by_clip = {}

    assignments = {}
    for aid in annotator_ids:
        assignments[aid] = {
            "annotator_id": aid,
            "role": "tier_1",
            "total_clips": 0,
            "clips": [],
        }

    # Low confidence — 2 annotators
    low_clips = routed_clips.get("low_confidence", [])
    random.shuffle(low_clips)
    for i, clip in enumerate(low_clips):
        cid = clip["clip_id"]
        entry = {"clip_id": cid, "route": "low_confidence", "task": "fresh_annotation"}
        a1 = annotator_ids[i % num_annotators]
        a2 = annotator_ids[(i + 1) % num_annotators]
        assignments[a1]["clips"].append(entry)
        assignments[a2]["clips"].append({**entry})

    # Medium — 1 annotator with pre-fill
    med_clips = routed_clips.get("medium_confidence", [])
    for i, clip in enumerate(med_clips):
        cid = clip["clip_id"]
        prefill = build_prefill_data(vlm_by_clip.get(cid, {})) if cid in vlm_by_clip else None
        aid = annotator_ids[i % num_annotators]
        assignments[aid]["clips"].append({
            "clip_id": cid, "route": "medium_confidence",
            "task": "review_prefill", "prefill": prefill,
        })

    # High — sample verify
    high_clips = routed_clips.get("high_confidence", [])
    sample = select_verification_sample(high_clips)
    for i, clip in enumerate(sample):
        cid = clip["clip_id"]
        prefill = build_prefill_data(vlm_by_clip.get(cid, {})) if cid in vlm_by_clip else None
        aid = annotator_ids[i % num_annotators]
        assignments[aid]["clips"].append({
            "clip_id": cid, "route": "high_confidence",
            "task": "verify_consensus", "prefill": prefill,
        })

    for aid in annotator_ids:
        assignments[aid]["total_clips"] = len(assignments[aid]["clips"])

    return assignments, set()


def generate_assignments(vlm_results, manifest_metadata, experiment_config=None,
                         num_annotators=3, annotator_ids=None):
    """Main pipeline: VLM results -> routing -> assignment."""
    agreements = build_clip_agreements(vlm_results)
    routing_result = route_batch(agreements)
    routed = routing_result["routed_clips"]
    summary = routing_result["summary"]

    vlm_by_clip = {r["clip_id"]: r for r in vlm_results}

    # Attach prefill and manifest data
    for route_name, clips in routed.items():
        for clip in clips:
            cid = clip["clip_id"]
            if cid in vlm_by_clip:
                clip["prefill"] = build_prefill_data(vlm_by_clip[cid])
            if cid in manifest_metadata:
                clip["manifest"] = manifest_metadata[cid]

    # Route to assignment
    if experiment_config:
        assignments, safety_clips = assign_role_based(
            routed, experiment_config, manifest_metadata, vlm_by_clip
        )
    else:
        assignments, safety_clips = assign_simple(
            routed, num_annotators, annotator_ids, vlm_by_clip
        )

    output = {
        "generated_at": datetime.now().isoformat(),
        "experiment_name": experiment_config.get("experiment_name", "") if experiment_config else "",
        "routing_summary": summary,
        "safety_critical_clips": list(safety_clips),
        "assignments": assignments,
    }

    return output


def save_assignments(output, output_path=None):
    """Save assignment manifest to JSON."""
    if output_path is None:
        output_path = os.path.join(ASSIGNMENTS_DIR, "assignments.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    return output_path


def save_per_annotator_csv(output, output_dir=None):
    """Save per-annotator CSV assignment sheets."""
    if output_dir is None:
        output_dir = ASSIGNMENTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for aid, data in output["assignments"].items():
        csv_path = os.path.join(output_dir, f"{aid}_clips.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_id", "route", "task", "role"])
            for clip in data.get("clips", []):
                writer.writerow([
                    clip["clip_id"], clip.get("route", ""),
                    clip.get("task", ""), data.get("role", "tier_1"),
                ])
        paths.append(csv_path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Auto-assign clips to annotators")
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment YAML config (enables role-based assignment)")
    parser.add_argument("--annotators", type=int, default=3,
                        help="Number of annotators (simple mode, default: 3)")
    parser.add_argument("--annotator-ids", type=str, nargs="+",
                        help="Specific annotator IDs (simple mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--vlm-results", type=str, default=None,
                        help="Path to vlm_annotations.jsonl")
    args = parser.parse_args()

    # Load experiment config if provided
    experiment_config = None
    if args.config:
        from annotation_tool.experiment_config import load_experiment
        experiment_config = load_experiment(args.config)
        print(f"Experiment: {experiment_config['experiment_name']}")

    # Load data
    vlm_results = load_vlm_results(args.vlm_results)
    if not vlm_results:
        print("No VLM results found. Run vlm_annotator first.")
        sys.exit(1)

    manifest_meta = load_manifest_metadata()
    print(f"Loaded {len(vlm_results)} VLM results, "
          f"{len(manifest_meta)} manifest entries")

    # Generate assignments
    output = generate_assignments(
        vlm_results, manifest_meta,
        experiment_config=experiment_config,
        num_annotators=args.annotators,
        annotator_ids=args.annotator_ids,
    )

    # Save
    json_path = save_assignments(output, args.output)
    csv_paths = save_per_annotator_csv(output)

    # Print summary
    summary = output["routing_summary"]
    print("\n" + "=" * 60)
    print("ASSIGNMENT SUMMARY")
    print("=" * 60)
    print(f"Total clips:       {summary['total_clips']}")
    print(f"High confidence:   {summary['high_confidence']['count']} "
          f"({summary['high_confidence']['percentage']}%)")
    print(f"Medium confidence: {summary['medium_confidence']['count']} "
          f"({summary['medium_confidence']['percentage']}%)")
    print(f"Low confidence:    {summary['low_confidence']['count']} "
          f"({summary['low_confidence']['percentage']}%)")

    safety_count = len(output.get("safety_critical_clips", []))
    if safety_count:
        print(f"Safety-critical:   {safety_count}")

    print(f"\nAnnotator assignments:")
    for aid, data in output["assignments"].items():
        role = data.get("role", "tier_1")
        total = data.get("total_clips", len(data.get("clips", [])))
        print(f"  {aid} ({role}): {total} clips")

    print(f"\nSaved to: {json_path}")
    for p in csv_paths:
        print(f"  CSV: {p}")


if __name__ == "__main__":
    main()
