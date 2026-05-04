#!/usr/bin/env python3
"""Curate a class-balanced batch of clips for annotation.

Selects clips from Tier A manifest to maximize action class diversity,
filling gaps in the current annotation distribution.

Usage:
    python curate_batch.py --size 300
    python curate_batch.py --size 50 --prioritize F1,F2,C1,D1
    python curate_batch.py --size 100 --exclude-annotated
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent)
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
VLM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results")
ACTIVE_BATCH_DIR = os.path.join(PROJECT_ROOT, "active_batch")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations")


def load_tier_a_manifest():
    """Load all Tier A clips."""
    manifest_path = os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv")
    clips = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            clips.append(row)
    return clips


def load_tier_b_manifest(path=None):
    """Load all Tier B clips."""
    manifest_path = path or os.path.join(OUTPUT_DIR, "metadata", "tier_b_manifest.csv")
    clips = []
    if not os.path.exists(manifest_path):
        return clips
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            clips.append(row)
    return clips


def stratified_sample_with_floor(clips, n_target, key_fn, min_per=0, max_per=None,
                                  seed=42):
    """Generic stratified sampling with floor and cap per stratum.

    Strategy:
    1. Group clips by key_fn(clip)
    2. Compute proportional allocation per stratum
    3. Apply floor: each stratum gets at least min_per (if pool allows)
    4. Apply cap: each stratum gets at most max_per
    5. Drift correction to hit n_target exactly

    Args:
        clips: list of clip dicts
        n_target: total number of clips to select
        key_fn: callable returning the stratum key for a clip
        min_per: minimum clips per stratum (subject to availability)
        max_per: maximum clips per stratum (None = no cap)
        seed: rng seed

    Returns:
        List of selected clips.
    """
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for c in clips:
        by_stratum[key_fn(c)].append(c)

    total_pool = sum(len(v) for v in by_stratum.values())
    if total_pool == 0 or n_target <= 0:
        return []

    # 1. Proportional allocation
    allocated = {}
    for stratum, pool in by_stratum.items():
        prop = int(round(n_target * len(pool) / total_pool))
        allocated[stratum] = prop

    # 2. Apply floor (subject to pool availability)
    if min_per > 0:
        for stratum, pool in by_stratum.items():
            if allocated[stratum] < min_per:
                allocated[stratum] = min(min_per, len(pool))

    # 3. Apply cap
    if max_per is not None:
        for stratum in allocated:
            allocated[stratum] = min(allocated[stratum], max_per)

    # 4. Drift correction (hit n_target exactly)
    drift = n_target - sum(allocated.values())
    strata_sorted_desc = sorted(by_stratum.keys(),
                                 key=lambda s: -len(by_stratum[s]))
    i = 0
    safety = max(len(by_stratum), 1) * 20
    while drift != 0 and i < safety:
        s = strata_sorted_desc[i % len(strata_sorted_desc)]
        pool_size = len(by_stratum[s])
        cap = max_per if max_per is not None else pool_size
        if drift > 0 and allocated[s] < min(pool_size, cap):
            allocated[s] += 1
            drift -= 1
        elif drift < 0 and allocated[s] > min_per and allocated[s] > 0:
            allocated[s] -= 1
            drift += 1
        i += 1

    # 5. Pick clips per stratum
    selected = []
    for stratum, k in allocated.items():
        pool = by_stratum[stratum][:]
        rng.shuffle(pool)
        selected.extend(pool[:k])

    rng.shuffle(selected)
    return selected[:n_target]


def stratified_tier_b_sample(tier_b_clips, n_target, seed=42, min_per_site=0):
    """Select n_target Tier B clips stratified by (site, severity_level).

    Combines severity stratification (S0/S1/S2) with site coverage. Each site
    gets at least min_per_site clips when possible.
    """
    if not tier_b_clips:
        return []

    # If site floor requested, do 2-pass: first allocate per site, then per
    # severity within site. Otherwise use the original severity-only stratification.
    if min_per_site > 0:
        # Stratify by site (with floor), then balance severity within each site
        sites = defaultdict(list)
        for c in tier_b_clips:
            sites[c.get("site", "?")].append(c)
        n_sites = len(sites)
        # Per-site allocation: floor + proportional
        site_target = max(min_per_site,
                          n_target // max(n_sites, 1))
        selected = []
        for site, pool in sites.items():
            site_n = min(site_target, len(pool))
            picked = stratified_sample_with_floor(
                pool, site_n,
                key_fn=lambda c: c.get("severity_level", "S0") or "S0",
                seed=seed,
            )
            selected.extend(picked)
        # Trim to n_target
        rng = random.Random(seed)
        rng.shuffle(selected)
        if len(selected) > n_target:
            selected = selected[:n_target]
        elif len(selected) < n_target:
            # Fill remaining proportionally (no site constraint)
            picked_ids = {c.get("clip_id") for c in selected}
            remaining = [c for c in tier_b_clips if c.get("clip_id") not in picked_ids]
            extra = stratified_sample_with_floor(
                remaining, n_target - len(selected),
                key_fn=lambda c: c.get("severity_level", "S0") or "S0",
                seed=seed + 1,
            )
            selected.extend(extra)
        return selected[:n_target]

    # Severity-only stratification (original behavior)
    return stratified_sample_with_floor(
        tier_b_clips, n_target,
        key_fn=lambda c: c.get("severity_level", "S0") or "S0",
        seed=seed,
    )


def stratified_tier_a_sample(tier_a_clips, n_target, seed=42, min_per_site=0,
                              max_per_site=None):
    """Select n_target Tier A clips stratified by (site, severity_level).

    Two-pass: first allocate per site (with floor/cap for site coverage), then
    within each site allocate by severity_level proportionally. Preserves both
    site diversity AND severity distribution. Used as a class-info-free
    alternative to curate_balanced_batch (e.g., for new clips without VLM
    predictions yet).
    """
    if not tier_a_clips:
        return []

    if min_per_site > 0:
        # Two-pass: stratify by site (with floor), then severity within each site
        sites = defaultdict(list)
        for c in tier_a_clips:
            sites[c.get("site", "?")].append(c)
        n_sites = len(sites)
        cap = max_per_site if max_per_site is not None else n_target
        # Per-site target — at least min_per_site, at most cap
        site_target = max(min_per_site, n_target // max(n_sites, 1))
        site_target = min(site_target, cap)

        selected = []
        for site, pool in sites.items():
            site_n = min(site_target, len(pool))
            picked = stratified_sample_with_floor(
                pool, site_n,
                key_fn=lambda c: c.get("severity_level", "S0") or "S0",
                seed=seed,
            )
            selected.extend(picked)

        # Trim or fill to exact n_target
        rng = random.Random(seed)
        rng.shuffle(selected)
        if len(selected) > n_target:
            selected = selected[:n_target]
        elif len(selected) < n_target:
            picked_ids = {c.get("clip_id") for c in selected}
            remaining = [c for c in tier_a_clips
                         if c.get("clip_id") not in picked_ids]
            extra = stratified_sample_with_floor(
                remaining, n_target - len(selected),
                key_fn=lambda c: c.get("severity_level", "S0") or "S0",
                seed=seed + 1,
            )
            selected.extend(extra)
        return selected[:n_target]

    # No site floor — fall back to severity-only stratification
    return stratified_sample_with_floor(
        tier_a_clips, n_target,
        key_fn=lambda c: c.get("severity_level", "S0") or "S0",
        seed=seed,
    )


def temporal_dedup(clips, window_sec=300, max_per_window=3, seed=42):
    """Remove near-duplicate clips from the same camera within a time window.

    Groups clips by (camera_id, date, floor(source_start_sec / window_sec)).
    Within each bucket with >max_per_window clips, selects the most diverse
    subset using greedy novelty selection.

    Tiebreaker (pre-VLM, no action labels available):
        1. severity_level diversity (different severity = more novel)
        2. avg_persons diversity (different worker count = more novel)
        3. lowest BRISQUE (best image quality)

    Singletons pass through unchanged. Never merges across cameras or dates.
    Deterministic with fixed seed.

    Returns:
        (deduped_clips, stats) where stats is a dict with removal counts.
    """
    rng = random.Random(seed)

    # Bucket clips
    buckets = defaultdict(list)
    for clip in clips:
        camera_id = clip.get("camera_id", "unknown")
        date = clip.get("date", "unknown")
        start_sec = float(clip.get("source_start_sec", 0))
        bucket_key = (camera_id, date, int(start_sec // window_sec))
        buckets[bucket_key].append(clip)

    deduped = []
    total_removed = 0
    bucket_sizes = []

    for key, bucket in buckets.items():
        bucket_sizes.append(len(bucket))
        if len(bucket) <= max_per_window:
            deduped.extend(bucket)
            continue

        # Greedy novelty selection
        kept = []
        remaining = list(bucket)

        while len(kept) < max_per_window and remaining:
            if not kept:
                # First pick: best quality (lowest BRISQUE)
                remaining.sort(key=lambda c: float(c.get("brisque_score", 999)))
                kept.append(remaining.pop(0))
                continue

            # Score each candidate on novelty vs kept set
            best_score = None
            best_idx = 0
            for i, candidate in enumerate(remaining):
                score = _novelty_score(candidate, kept)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
                elif score == best_score:
                    # Tiebreak: lower BRISQUE = better quality
                    if float(candidate.get("brisque_score", 999)) < \
                       float(remaining[best_idx].get("brisque_score", 999)):
                        best_idx = i

            kept.append(remaining.pop(best_idx))

        deduped.extend(kept)
        total_removed += len(bucket) - len(kept)

    stats = {
        "input_clips": len(clips),
        "output_clips": len(deduped),
        "removed": total_removed,
        "total_buckets": len(buckets),
        "singleton_buckets": sum(1 for s in bucket_sizes if s == 1),
        "multi_buckets": sum(1 for s in bucket_sizes if s > max_per_window),
    }
    return deduped, stats


def _novelty_score(candidate, kept):
    """Compute novelty score of candidate relative to the kept set.

    Higher = more novel. Priority:
        1. New severity_level (not seen in kept) → score 3
        2. New avg_persons bucket (not seen in kept) → score 2
        3. Base → score 0
    Scores are additive.
    """
    score = 0
    c_sev = candidate.get("severity_level", "S0") or "S0"
    c_persons = _persons_bucket(candidate)

    kept_sevs = {(k.get("severity_level", "S0") or "S0") for k in kept}
    kept_persons = {_persons_bucket(k) for k in kept}

    if c_sev not in kept_sevs:
        score += 3
    if c_persons not in kept_persons:
        score += 2
    return score


def _persons_bucket(clip):
    """Bucket avg_persons into low/med/high for diversity comparison."""
    avg = float(clip.get("avg_persons", 1))
    if avg <= 1.5:
        return "low"
    elif avg <= 4:
        return "med"
    return "high"


def load_exclude_clip_ids(csv_paths):
    """Load clip IDs to exclude from one or more CSV manifest files."""
    exclude_ids = set()
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                cid = row.get("clip_id", "")
                if cid:
                    exclude_ids.add(cid)
    return exclude_ids


def load_vlm_annotations(path=None):
    """Load VLM annotations for action class info.

    Args:
        path: Optional custom path to JSONL file. If None, uses default location.
    """
    results = {}
    results_path = path or os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    if not os.path.exists(results_path):
        return results
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                cid = r.get("clip_id", "")
                if r.get("vlm", {}).get("success") and r.get("vlm", {}).get("normalized"):
                    results[cid] = r.get("vlm", {}).get("normalized", {}).get("action_class", "")
            except json.JSONDecodeError:
                continue
    return results


def generate_x1_report(clips, vlm_results_path, output_path):
    """Generate report of X1 (unlisted_action) clips for taxonomy review.

    Writes CSV with clip_id, VLM free_text_description, site, camera_id
    for clips predicted as X1 by the VLM.
    """
    x1_clips = []
    if not os.path.exists(vlm_results_path):
        return x1_clips

    clip_meta = {c.get("clip_id"): c for c in clips}

    with open(vlm_results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                cid = r.get("clip_id", "")
                norm = r.get("vlm", {}).get("normalized", {})
                action = norm.get("action_class", "")
                if action and action.upper() == "X1":
                    meta = clip_meta.get(cid, {})
                    persons = norm.get("persons", [])
                    desc = persons[0].get("free_text_description", "") if persons else ""
                    x1_clips.append({
                        "clip_id": cid,
                        "site": meta.get("site", ""),
                        "camera_id": meta.get("camera_id", ""),
                        "free_text_description": desc,
                        "num_workers": norm.get("num_workers", ""),
                        "scene_type": norm.get("scene_type", ""),
                    })
            except json.JSONDecodeError:
                continue

    if x1_clips and output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(x1_clips[0].keys()))
            writer.writeheader()
            writer.writerows(x1_clips)
        print(f"\nX1 report written to {output_path} ({len(x1_clips)} clips)")

    return x1_clips


def load_existing_annotations():
    """Load already-annotated clip IDs and their action classes."""
    annotated = {}
    if not os.path.exists(ANNOTATIONS_DIR):
        return annotated
    for annotator_id in os.listdir(ANNOTATIONS_DIR):
        ann_dir = os.path.join(ANNOTATIONS_DIR, annotator_id)
        if not os.path.isdir(ann_dir):
            continue
        for fname in os.listdir(ann_dir):
            if fname.endswith(".json"):
                with open(os.path.join(ann_dir, fname)) as f:
                    ann = json.load(f)
                    cid = ann.get("clip_id", "")
                    persons = ann.get("persons", [])
                    action = persons[0].get("action_code", "") if persons else ""
                    annotated[cid] = action
    return annotated


def get_class_distribution(annotated):
    """Get current class distribution from completed annotations."""
    return Counter(v for v in annotated.values() if v)


def curate_balanced_batch(clips, vlm_actions, existing_dist, batch_size,
                          prioritize=None, exclude_annotated=True,
                          annotated_ids=None, targets=None,
                          min_per_class=0, unclassified_budget=0):
    """Select clips to maximize class diversity (hybrid mode).

    Hybrid strategy:
    1. Fill explicit per-class targets first (if provided)
    2. Fill remaining classes to min_per_class
    3. Add unclassified clips for discovery
    4. Fill remaining budget from most underrepresented classes
    5. If prioritize list given, boost those classes
    """
    if annotated_ids is None:
        annotated_ids = set()
    if targets is None:
        targets = {}

    # Group clips by predicted action
    by_class = defaultdict(list)
    unclassified = []
    for clip in clips:
        cid = clip.get("clip_id", "")
        if exclude_annotated and cid in annotated_ids:
            continue
        action = vlm_actions.get(cid, "")
        if action:
            by_class[action].append(clip)
        else:
            unclassified.append(clip)

    selected = []
    selected_ids = set()
    remaining_budget = batch_size

    def pick_from_class(cls, count):
        """Pick up to `count` clips from a class."""
        nonlocal remaining_budget
        available = [c for c in by_class.get(cls, [])
                     if c.get("clip_id") not in selected_ids]
        random.shuffle(available)
        picked = 0
        for clip in available[:count]:
            if remaining_budget <= 0:
                break
            selected.append(clip)
            selected_ids.add(clip.get("clip_id"))
            remaining_budget -= 1
            picked += 1
        return picked

    # Step 1: Fill explicit per-class targets
    if targets:
        for cls, target_count in targets.items():
            picked = pick_from_class(cls, target_count)
            if picked < target_count:
                pass  # Not enough clips of this class available

    # Step 2: Fill remaining classes to min_per_class
    if min_per_class > 0:
        all_classes = sorted(set(list(by_class.keys()) + list(existing_dist.keys())))
        for cls in all_classes:
            if cls in targets:
                continue  # Already handled
            current_in_batch = sum(1 for c in selected
                                   if vlm_actions.get(c.get("clip_id"), "") == cls)
            needed = max(0, min_per_class - current_in_batch)
            if needed > 0:
                pick_from_class(cls, needed)

    # Step 3: Add unclassified clips for discovery
    if unclassified_budget > 0 and remaining_budget > 0:
        random.shuffle(unclassified)
        uc_count = min(unclassified_budget, remaining_budget, len(unclassified))
        for clip in unclassified[:uc_count]:
            if remaining_budget <= 0:
                break
            selected.append(clip)
            selected_ids.add(clip.get("clip_id"))
            remaining_budget -= 1

    # Step 4: Fill remaining budget from most underrepresented classes
    if remaining_budget > 0:
        all_classes = sorted(set(list(by_class.keys()) + list(existing_dist.keys())))
        if not all_classes:
            all_classes = list(by_class.keys())

        # Compute deficit
        total_target = sum(existing_dist.values()) + batch_size
        per_class_target = total_target / max(len(all_classes), 1)
        deficits = {}
        for cls in all_classes:
            current = existing_dist.get(cls, 0)
            in_batch = sum(1 for c in selected
                           if vlm_actions.get(c.get("clip_id"), "") == cls)
            deficit = per_class_target - current - in_batch
            deficits[cls] = max(0, deficit)

        if prioritize:
            for cls in prioritize:
                deficits[cls] = deficits.get(cls, 0) + remaining_budget

        sorted_classes = sorted(deficits.keys(), key=lambda c: -deficits[c])

        rounds = 0
        while remaining_budget > 0 and rounds < batch_size:
            added_this_round = 0
            for cls in sorted_classes:
                if remaining_budget <= 0:
                    break
                available = by_class.get(cls, [])
                candidates = [c for c in available
                              if c.get("clip_id") not in selected_ids]
                if candidates:
                    pick = random.choice(candidates)
                    selected.append(pick)
                    selected_ids.add(pick.get("clip_id"))
                    remaining_budget -= 1
                    added_this_round += 1
            rounds += 1
            if added_this_round == 0:
                break

    # Fill remaining with unclassified or random
    if remaining_budget > 0:
        selected_ids = {c.get("clip_id") for c in selected}
        extras = [c for c in unclassified if c.get("clip_id") not in selected_ids]
        random.shuffle(extras)
        selected.extend(extras[:remaining_budget])

    return selected


def build_active_batch(selected_clips, batch_name=None):
    """Copy selected clips and frames into active_batch directory."""
    if batch_name is None:
        from datetime import datetime
        batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")

    batch_dir = ACTIVE_BATCH_DIR
    clips_dir = os.path.join(batch_dir, "clips")
    frames_dir = os.path.join(batch_dir, "frames")
    config_dir = os.path.join(batch_dir, "config")
    detections_dir = os.path.join(batch_dir, "metadata", "clip_detections")

    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)

    manifest_rows = []
    copied_clips = 0
    copied_frames = 0
    copied_detections = 0
    missing_clips = 0

    for clip in selected_clips:
        cid = clip.get("clip_id", "")
        clip_path = clip.get("clip_path", "")
        frames_rel = clip.get("frames_dir", f"frames/{cid}")

        # Copy video clip
        src_clip = os.path.join(OUTPUT_DIR, clip_path)
        if os.path.exists(src_clip):
            dst_clip = os.path.join(clips_dir, f"{cid}.mp4")
            if not os.path.exists(dst_clip):
                shutil.copy2(src_clip, dst_clip)
            copied_clips += 1
        else:
            missing_clips += 1

        # Copy frames
        src_frames = os.path.join(OUTPUT_DIR, frames_rel)
        dst_frames = os.path.join(frames_dir, cid)
        if os.path.exists(src_frames) and not os.path.exists(dst_frames):
            shutil.copytree(src_frames, dst_frames)
        if os.path.exists(dst_frames):
            copied_frames += len(os.listdir(dst_frames))

        # Copy detection JSON (for bounding box overlays)
        src_det = os.path.join(OUTPUT_DIR, "metadata", "clip_detections", f"{cid}.json")
        dst_det = os.path.join(detections_dir, f"{cid}.json")
        if os.path.exists(src_det) and not os.path.exists(dst_det):
            shutil.copy2(src_det, dst_det)
            copied_detections += 1

        manifest_rows.append(clip)

    # Write batch manifest
    if manifest_rows:
        manifest_path = os.path.join(config_dir, "batch_manifest.csv")
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    # Write batch info
    info = {
        "batch_name": batch_name,
        "total_clips": len(selected_clips),
        "copied_clips": copied_clips,
        "copied_frames": copied_frames,
        "missing_clips": missing_clips,
        "clip_ids": [c.get("clip_id", "") for c in selected_clips],
    }
    with open(os.path.join(config_dir, "batch_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


def print_distribution_report(selected_clips, vlm_actions, existing_dist):
    """Print class distribution of selected batch vs existing."""
    batch_dist = Counter()
    for clip in selected_clips:
        cid = clip.get("clip_id", "")
        action = vlm_actions.get(cid, "unknown")
        batch_dist[action] += 1

    all_classes = sorted(set(list(batch_dist.keys()) + list(existing_dist.keys())))

    print(f"\n{'Class':<8} {'Existing':>10} {'This Batch':>12} {'New Total':>10}")
    print("-" * 44)
    for cls in all_classes:
        existing = existing_dist.get(cls, 0)
        batch = batch_dist.get(cls, 0)
        total = existing + batch
        print(f"{cls:<8} {existing:>10} {batch:>12} {total:>10}")
    print("-" * 44)
    print(f"{'TOTAL':<8} {sum(existing_dist.values()):>10} "
          f"{len(selected_clips):>12} "
          f"{sum(existing_dist.values()) + len(selected_clips):>10}")


def main():
    parser = argparse.ArgumentParser(description="Curate class-balanced annotation batch")
    parser.add_argument("--size", type=int, default=50,
                        help="Batch size (default: 50)")
    parser.add_argument("--prioritize", type=str, default=None,
                        help="Comma-separated action codes to prioritize (e.g., F1,F2,C1)")
    parser.add_argument("--exclude-annotated", action="store_true", default=True,
                        help="Exclude already-annotated clips (default: True)")
    parser.add_argument("--include-annotated", action="store_true",
                        help="Include already-annotated clips")
    parser.add_argument("--batch-name", type=str, default=None,
                        help="Name for this batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show selection without copying files")
    parser.add_argument("--tier-b-ratio", type=float, default=0.0,
                        help="Fraction of batch from Tier B (0.0-1.0). "
                             "Stratified by severity_level (S0/S1/S2).")
    parser.add_argument("--tier-b-manifest", type=str, default=None,
                        help="Path to tier_b_manifest.csv (default: standard location)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--min-per-site", type=int, default=0,
                        help="Minimum clips per site (site-diverse stratification)")
    parser.add_argument("--max-per-site", type=int, default=None,
                        help="Maximum clips per site (site-diverse stratification)")
    parser.add_argument("--temporal-dedup-window", type=int, default=0,
                        help="Temporal dedup window in seconds (0 = disabled). "
                             "Groups clips by (camera_id, date, window) and keeps "
                             "max-per-window per group.")
    parser.add_argument("--max-per-window", type=int, default=3,
                        help="Max clips per temporal dedup bucket (default: 3)")
    parser.add_argument("--exclude-clip-ids", type=str, nargs="*", default=None,
                        help="CSV manifest file(s) with clip_ids to exclude from "
                             "dedup and curation (e.g., pilot and GT manifests)")
    parser.add_argument("--min-per-class", type=int, default=0,
                        help="Minimum clips per action class (class-balanced curation). "
                             "Requires VLM annotations.")
    parser.add_argument("--vlm-results", type=str, default=None,
                        help="Path to VLM annotations JSONL file (for class-balanced "
                             "curation). Defaults to standard location.")
    parser.add_argument("--temporal-dedup-only", action="store_true",
                        help="Only run temporal dedup and output deduped manifest, "
                             "skip curation and batch building.")
    parser.add_argument("--dedup-output", type=str, default=None,
                        help="Output path for deduped manifest CSV (used with "
                             "--temporal-dedup-only)")
    args = parser.parse_args()

    random.seed(args.seed)
    exclude = not args.include_annotated

    # Load data — always load both tiers for dedup; tier ratio only matters for curation
    clips_a = load_tier_a_manifest()
    load_tier_b = args.tier_b_ratio > 0 or args.temporal_dedup_window > 0
    tier_b_clips_raw = load_tier_b_manifest(args.tier_b_manifest) if load_tier_b else []
    all_clips = clips_a + tier_b_clips_raw

    print(f"Tier A clips available: {len(clips_a)}")
    if tier_b_clips_raw:
        print(f"Tier B clips available: {len(tier_b_clips_raw)}")
    print(f"Total clips in pool: {len(all_clips)}")

    # Exclude pilot/GT clip IDs if specified
    exclude_ids = set()
    if args.exclude_clip_ids:
        exclude_ids = load_exclude_clip_ids(args.exclude_clip_ids)
        all_clips = [c for c in all_clips if c.get("clip_id") not in exclude_ids]
        clips_a = [c for c in clips_a if c.get("clip_id") not in exclude_ids]
        tier_b_clips_raw = [c for c in tier_b_clips_raw if c.get("clip_id") not in exclude_ids]
        print(f"Excluded {len(exclude_ids)} clip IDs (pilot/GT)")
        print(f"Remaining pool: {len(all_clips)}")

    # --- Temporal deduplication (PRE-VLM) ---
    if args.temporal_dedup_window > 0:
        print(f"\nTemporal dedup (window={args.temporal_dedup_window}s, "
              f"max_per_window={args.max_per_window})...")

        # Dedup the combined pool. Each clip retains its tier membership
        # (from the 'tier' column or which manifest it came from).
        # We tag clips with their tier before dedup so we can split after.
        for c in clips_a:
            c.setdefault("tier", "A")
        for c in tier_b_clips_raw:
            c.setdefault("tier", "B")

        combined_pool = clips_a + tier_b_clips_raw
        deduped_all, stats_all = temporal_dedup(
            combined_pool, args.temporal_dedup_window, args.max_per_window, args.seed)

        print(f"  Combined: {stats_all['input_clips']} → {stats_all['output_clips']} "
              f"({stats_all['removed']} removed, {stats_all['total_buckets']} buckets, "
              f"{stats_all['multi_buckets']} multi-clip buckets, "
              f"{stats_all['singleton_buckets']} singletons)")

        # Split back into Tier A and B for downstream curation
        clips_a = [c for c in deduped_all if c.get("tier") == "A"]
        tier_b_clips_raw = [c for c in deduped_all if c.get("tier") == "B"]
        all_clips = deduped_all
        print(f"  After dedup: {len(clips_a)} Tier A + {len(tier_b_clips_raw)} Tier B "
              f"= {len(all_clips)} total")

        # Temporal-dedup-only mode: output manifest and exit
        if args.temporal_dedup_only:
            out_path = args.dedup_output or os.path.join(
                OUTPUT_DIR, "metadata", "deduped_manifest.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if all_clips:
                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_clips[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_clips)
            print(f"\nDeduped manifest written to: {out_path}")
            print(f"[DEDUP ONLY] No curation performed.")
            return

    # --- Load VLM annotations and existing annotations ---
    vlm_actions = load_vlm_annotations(args.vlm_results)
    annotated = load_existing_annotations()
    existing_dist = get_class_distribution(annotated)

    print(f"\nVLM-classified clips: {len(vlm_actions)}")
    print(f"Already annotated: {len(annotated)}")
    if existing_dist:
        print(f"Current distribution: {dict(existing_dist)}")

    prioritize = args.prioritize.split(",") if args.prioritize else None

    # Exclude already-annotated clips
    if exclude and annotated:
        clips_a = [c for c in clips_a if c.get("clip_id") not in annotated]
        tier_b_clips_raw = [c for c in tier_b_clips_raw if c.get("clip_id") not in annotated]

    # --- Compute Tier A vs Tier B split ---
    if args.tier_b_ratio > 0:
        n_tier_b_target = int(round(args.size * args.tier_b_ratio))
        # Handle Tier B shortfall: take all available if pool < target
        n_tier_b = min(n_tier_b_target, len(tier_b_clips_raw))
        n_tier_a = args.size - n_tier_b
        if n_tier_b < n_tier_b_target:
            print(f"\n⚠ Tier B shortfall: target {n_tier_b_target}, available {len(tier_b_clips_raw)}")
            print(f"  Taking all {n_tier_b} Tier B, filling deficit from Tier A")
            print(f"  Achieved ratio: {n_tier_a}/{n_tier_b} "
                  f"({100*n_tier_a/args.size:.0f}/{100*n_tier_b/args.size:.0f})")
        else:
            print(f"\nSplit: Tier A={n_tier_a}, Tier B={n_tier_b}")
    else:
        n_tier_b = 0
        n_tier_a = args.size
        tier_b_clips_raw = []

    # --- Curate Tier A slice ---
    # Use class-balanced curation when min_per_class is set and VLM labels available
    if args.min_per_class > 0 and vlm_actions:
        print(f"\nClass-balanced curation (min_per_class={args.min_per_class})...")
        selected_a = curate_balanced_batch(
            clips_a, vlm_actions, existing_dist, n_tier_a,
            prioritize=prioritize,
            exclude_annotated=False,  # already filtered above
            annotated_ids=set(),
            min_per_class=args.min_per_class,
            unclassified_budget=int(args.size * 0.07),  # ~7% for X1 discovery
        )
    elif args.min_per_site > 0:
        selected_a = stratified_tier_a_sample(
            clips_a, n_tier_a,
            seed=args.seed,
            min_per_site=args.min_per_site,
            max_per_site=args.max_per_site,
        )
    else:
        selected_a = curate_balanced_batch(
            clips_a, vlm_actions, existing_dist, n_tier_a,
            prioritize=prioritize,
            exclude_annotated=exclude,
            annotated_ids=set(annotated.keys()),
        )

    # --- Curate Tier B slice (severity-stratified, site-floor optional) ---
    selected_b = stratified_tier_b_sample(
        tier_b_clips_raw, n_tier_b, args.seed,
        min_per_site=args.min_per_site,
    )

    selected = selected_a + selected_b
    random.shuffle(selected)

    # --- Reports ---
    print(f"\nSelected {len(selected)} clips ({len(selected_a)} A + {len(selected_b)} B)")
    print_distribution_report(selected, vlm_actions, existing_dist)

    # Class coverage report with floor check
    if args.min_per_class > 0 and vlm_actions:
        print(f"\nClass coverage (floor={args.min_per_class}):")
        batch_by_class = defaultdict(list)
        for c in selected:
            action = vlm_actions.get(c.get("clip_id", ""), "unknown")
            batch_by_class[action].append(c)
        from annotation_tool.agreement import ACTION_TAXONOMY
        all_classes = sorted(set(list(ACTION_TAXONOMY.keys()) + list(batch_by_class.keys())))
        shortfalls = []
        for cls in all_classes:
            count = len(batch_by_class.get(cls, []))
            status = "✓" if count >= args.min_per_class else "⚠ short"
            if count < args.min_per_class:
                shortfalls.append(cls)
            print(f"  {cls:<25s} {count:>5d}  {status}")
        if shortfalls:
            print(f"\n⚠ {len(shortfalls)} classes below floor: {', '.join(shortfalls)}")

    # Severity cross-cutting report (full batch, not just Tier B)
    sev_dist_full = Counter(c.get("severity_level", "S0") or "S0" for c in selected)
    print(f"\nSeverity distribution (cross-cutting, full batch): {dict(sev_dist_full)}")

    if selected_b:
        sev_dist_b = Counter(c.get("severity_level", "?") for c in selected_b)
        print(f"Tier B severity distribution: {dict(sev_dist_b)}")

    site_dist = Counter(c.get("site", "?") for c in selected)
    print(f"\nSite distribution ({len(site_dist)} sites covered):")
    for site, n in site_dist.most_common():
        print(f"  {site:30s} {n:>5d}")

    # Scene type distribution (emergent, not enforced)
    if vlm_actions:
        scene_types = Counter()
        vlm_path = args.vlm_results or os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
        if os.path.exists(vlm_path):
            selected_ids = {c.get("clip_id") for c in selected}
            with open(vlm_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("clip_id") in selected_ids:
                            st = r.get("vlm", {}).get("normalized", {}).get("scene_type", "?")
                            scene_types[st] += 1
                    except json.JSONDecodeError:
                        continue
            if scene_types:
                print(f"\nScene type distribution (emergent): {dict(scene_types)}")

    # X1 report
    if args.vlm_results:
        x1_output = os.path.join(ACTIVE_BATCH_DIR, "config", "x1_review.csv")
        generate_x1_report(selected, args.vlm_results, x1_output)

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        return

    # Build active batch
    info = build_active_batch(selected, args.batch_name)
    print(f"\nBatch built in: {ACTIVE_BATCH_DIR}")
    print(f"  Clips copied: {info['copied_clips']}")
    print(f"  Frames copied: {info['copied_frames']}")
    if info['missing_clips'] > 0:
        print(f"  Missing clips: {info['missing_clips']}")

    # Estimate size
    total_size = 0
    for root, dirs, files in os.walk(ACTIVE_BATCH_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nReady to sync: ./sync_vps.sh push")


if __name__ == "__main__":
    main()
