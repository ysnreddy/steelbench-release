"""Streamlit metrics dashboard for SteelBench annotation QC.

Live monitoring of agreement scores, Kappa, annotation progress,
decoy catch rates, and annotator performance.

Usage:
    streamlit run annotation_tool/dashboard.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.agreement import compute_clip_agreement
from annotation_tool.qc_metrics import (
    cohens_kappa,
    weighted_cohens_kappa,
    anchoring_bias_analysis,
    decoy_catch_rate,
    compute_all_iaa,
)

ANNOTATIONS_DIR = os.environ.get(
    "STEELBENCH_ANNOTATIONS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"))
OUTPUT_DIR = os.environ.get(
    "STEELBENCH_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))

# Fallback for tier_a count — check active_batch if output/ doesn't exist
_tier_a_path = os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv")
_batch_path = os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest.csv")
MANIFEST_PATH = _tier_a_path if os.path.exists(_tier_a_path) else _batch_path


def load_all_annotations():
    """Load all annotations from all annotators."""
    all_annotations = defaultdict(dict)  # {annotator_id: {clip_id: annotation}}

    if not os.path.exists(ANNOTATIONS_DIR):
        return all_annotations

    for annotator_id in os.listdir(ANNOTATIONS_DIR):
        ann_dir = os.path.join(ANNOTATIONS_DIR, annotator_id)
        if not os.path.isdir(ann_dir):
            continue
        for fname in os.listdir(ann_dir):
            if fname.endswith(".json"):
                with open(os.path.join(ann_dir, fname)) as f:
                    ann = json.load(f)
                    all_annotations[annotator_id][ann["clip_id"]] = ann

    return all_annotations


def load_tier_a_count():
    """Get total Tier A clips count."""
    if not os.path.exists(MANIFEST_PATH):
        return 0
    with open(MANIFEST_PATH) as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    st.set_page_config(page_title="SteelBench QC Dashboard", layout="wide")
    st.title("SteelBench Annotation QC Dashboard")

    all_annotations = load_all_annotations()
    total_tier_a = load_tier_a_count()

    if not all_annotations:
        st.info("No annotations yet. Start the annotation tool first.")
        return

    # ---------- Overall Progress ----------
    st.header("Overall Progress")

    annotator_ids = sorted(all_annotations.keys())
    total_annotations = sum(len(anns) for anns in all_annotations.values())
    unique_clips = set()
    for anns in all_annotations.values():
        unique_clips.update(anns.keys())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Annotations", total_annotations)
    col2.metric("Unique Clips Annotated", len(unique_clips))
    col3.metric("Total Tier A Clips", total_tier_a)
    col4.metric("Annotators Active", len(annotator_ids))

    if total_tier_a > 0:
        st.progress(min(1.0, len(unique_clips) / total_tier_a))
        st.caption(f"{len(unique_clips)}/{total_tier_a} clips annotated "
                   f"({100 * len(unique_clips) / total_tier_a:.1f}%)")

    # ---------- Per-Annotator Stats ----------
    st.header("Per-Annotator Statistics")

    annotator_stats = []
    for ann_id in annotator_ids:
        anns = all_annotations[ann_id]
        submitted = sum(1 for a in anns.values() if a.get("status") == "submitted")
        skipped = sum(1 for a in anns.values() if a.get("status") == "skipped")
        flagged = sum(1 for a in anns.values() if a.get("status") == "flagged")
        avg_conf = np.mean([
            a.get("annotator_confidence", 0)
            for a in anns.values() if a.get("status") == "submitted"
        ]) if submitted > 0 else 0

        annotator_stats.append({
            "Annotator": ann_id,
            "Submitted": submitted,
            "Skipped": skipped,
            "Flagged": flagged,
            "Total": len(anns),
            "Avg Confidence": round(avg_conf, 3),
        })

    st.dataframe(annotator_stats, width="stretch")

    # ---------- Annotation Distribution ----------
    st.header("Annotation Distribution")

    col1, col2 = st.columns(2)

    # Scene type distribution
    scene_counts = Counter()
    action_counts = Counter()
    for anns in all_annotations.values():
        for ann in anns.values():
            if ann.get("status") == "submitted":
                # Count group flags as scene indicator
                persons = ann.get("persons", [])
                if persons:
                    flags = [p.get("group_flag", "solo") for p in persons]
                    if "coordinated" in flags:
                        scene_counts["coordinated"] += 1
                    elif len(persons) > 1:
                        scene_counts["concurrent"] += 1
                    else:
                        scene_counts["solo"] += 1
                    for p in persons:
                        code = p.get("action_code", "unknown")
                        action_counts[code] += 1
                else:
                    scene_counts["unknown"] += 1

    with col1:
        st.subheader("Scene Types")
        if scene_counts:
            st.bar_chart(dict(scene_counts))
        else:
            st.info("No submitted annotations yet")

    with col2:
        st.subheader("Top Action Classes")
        if action_counts:
            top_actions = dict(action_counts.most_common(15))
            st.bar_chart(top_actions)
        else:
            st.info("No submitted annotations yet")

    # ---------- Inter-Annotator Agreement ----------
    st.header("Inter-Annotator Agreement")

    # Find clips annotated by multiple annotators
    clip_annotators = defaultdict(list)
    for ann_id, anns in all_annotations.items():
        for clip_id, ann in anns.items():
            if ann.get("status") == "submitted":
                clip_annotators[clip_id].append((ann_id, ann))

    double_annotated = {
        cid: annotators for cid, annotators in clip_annotators.items()
        if len(annotators) >= 2
    }

    col1, col2 = st.columns(2)
    col1.metric("Double-Annotated Clips", len(double_annotated))
    col2.metric("Target (200-300)", "200-300",
                delta=f"{len(double_annotated) - 200}" if double_annotated else None)

    if len(double_annotated) >= 10:
        # Compute IAA on first two annotators for each clip
        annotations_a = []
        annotations_b = []

        for clip_id, annotators in double_annotated.items():
            ann_a = annotators[0][1]
            ann_b = annotators[1][1]
            annotations_a.append(ann_a)
            annotations_b.append(ann_b)

        iaa = compute_all_iaa(annotations_a, annotations_b)

        st.subheader("Kappa Scores vs Targets")

        kappa_data = [
            {
                "Axis": "A1: Action Class",
                "Kappa": iaa["a1_action_kappa"],
                "Target": 0.75,
                "Status": "PASS" if iaa["a1_meets_target"] else "BELOW TARGET",
            },
            {
                "Axis": "A2: Spatial Context",
                "Kappa": iaa["a2_spatial_kappa"],
                "Target": 0.70,
                "Status": "PASS" if iaa["a2_meets_target"] else "BELOW TARGET",
            },
            {
                "Axis": "A3: PPE Items",
                "Kappa": iaa["a3_ppe_kappa"],
                "Target": 0.65,
                "Status": "PASS" if iaa["a3_meets_target"] else "BELOW TARGET",
            },
        ]
        st.dataframe(kappa_data, width="stretch")

        # Detailed per-tag / per-item breakdown
        with st.expander("Detailed Kappa Breakdown"):
            if iaa.get("a2_per_tag"):
                st.markdown("**A2: Per-Tag Kappa**")
                st.json(iaa["a2_per_tag"])
            if iaa.get("a3_per_item"):
                st.markdown("**A3: Per-Item Kappa**")
                st.json(iaa["a3_per_item"])

    elif double_annotated:
        st.warning(f"Only {len(double_annotated)} double-annotated clips. "
                   f"Need at least 10 for meaningful IAA computation.")
    else:
        st.info("No double-annotated clips yet. Assign overlapping clips to "
                "multiple annotators to compute agreement.")

    # ---------- Flagged Clips ----------
    st.header("Flagged Clips")

    flagged_clips = []
    for ann_id, anns in all_annotations.items():
        for clip_id, ann in anns.items():
            if ann.get("status") == "flagged":
                flagged_clips.append({
                    "Clip ID": clip_id,
                    "Annotator": ann_id,
                    "Timestamp": ann.get("annotator_timestamp", ""),
                })

    if flagged_clips:
        st.dataframe(flagged_clips, width="stretch")
    else:
        st.info("No flagged clips.")

    # ---------- Quality Alerts ----------
    st.header("Quality Alerts")

    alerts = []

    # Check for annotators with very high skip rates
    for ann_id in annotator_ids:
        anns = all_annotations[ann_id]
        if len(anns) >= 10:
            skip_rate = sum(1 for a in anns.values() if a.get("status") == "skipped") / len(anns)
            if skip_rate > 0.20:
                alerts.append(f"Annotator {ann_id} has high skip rate: {skip_rate:.1%}")

    # Check for low-confidence annotations
    for ann_id in annotator_ids:
        anns = all_annotations[ann_id]
        low_conf = [
            a for a in anns.values()
            if a.get("status") == "submitted" and a.get("annotator_confidence", 1.0) < 0.5
        ]
        if len(low_conf) > 5:
            alerts.append(
                f"Annotator {ann_id} has {len(low_conf)} low-confidence annotations (<0.5)"
            )

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No quality alerts.")


if __name__ == "__main__":
    main()
