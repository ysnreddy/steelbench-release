#!/usr/bin/env python3
"""Unified QC Monitoring Dashboard for SteelBench.

Single dashboard combining:
- Extraction pipeline progress
- VLM annotation agreement metrics
- Human annotation progress & IAA
- Routing distribution & assignment status
- Safety rule compliance
- Quality alerts

Usage:
    streamlit run annotation_tool/qc_dashboard.py
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

from annotation_tool.agreement import (
    ACTION_TAXONOMY,
    ALL_ACTION_LABELS,
    PPE_ITEMS,
)
from annotation_tool.qc_metrics import (
    compute_all_iaa,
)

# Inactive annotators — excluded from all dashboard views.
# These annotators have zero submissions and their clips are
# being redistributed to active annotators in Phase 3.
INACTIVE_ANNOTATORS = {"annotator_5", "annotator_6", "annotator_7",
                       "annotator_8", "annotator_9"}

# ---------- Paths (overridable via env vars for VPS) ----------
OUTPUT_DIR = os.environ.get(
    "STEELBENCH_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
VLM_RESULTS_DIR = os.environ.get(
    "STEELBENCH_VLM_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results"))
ANNOTATIONS_DIR = os.environ.get(
    "STEELBENCH_ANNOTATIONS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"))
ASSIGNMENTS_DIR = os.environ.get(
    "STEELBENCH_ASSIGNMENTS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "assignments"))


# ---------- Data Loaders (cached to avoid redundant I/O) ----------

@st.cache_data(ttl=30)
def load_extraction_stats():
    """Load extraction checkpoint stats."""
    cp_path = os.path.join(OUTPUT_DIR, "checkpoints", "progress.json")
    if not os.path.exists(cp_path):
        return None
    with open(cp_path) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_filtering_stats():
    """Load filtering report."""
    report_path = os.path.join(OUTPUT_DIR, "metadata", "filtering_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        return json.load(f)


@st.cache_data(ttl=30)
def load_tier_a_manifest():
    """Load Tier A manifest (or batch manifest on VPS)."""
    candidates = [
        os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv"),
        os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest.csv"),
    ]
    for manifest_path in candidates:
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                return list(csv.DictReader(f))
    return []


@st.cache_data(ttl=30)
def load_phase2_clip_set():
    """Return the set of Phase 2 clip_ids.

    Prefers the phase2-specific manifest; falls back to subtracting the
    pilot+GT manifest from the active batch_manifest.
    """
    p2 = os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest_phase2_3000.csv")
    if os.path.exists(p2):
        with open(p2) as f:
            return {r["clip_id"] for r in csv.DictReader(f)}
    # fall back
    pgt = os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest_pilot_gt.csv")
    bm = os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest.csv")
    if os.path.exists(pgt) and os.path.exists(bm):
        with open(pgt) as f:
            pilot_gt = {r["clip_id"] for r in csv.DictReader(f)}
        with open(bm) as f:
            all_cids = {r["clip_id"] for r in csv.DictReader(f)}
        return all_cids - pilot_gt
    return set()


@st.cache_data(ttl=30)
def load_vlm_results():
    """Load VLM annotation results."""
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


@st.cache_data(ttl=30)
def load_assignments():
    """Load assignment manifest."""
    assign_path = os.path.join(ASSIGNMENTS_DIR, "assignments.json")
    if not os.path.exists(assign_path):
        return None
    with open(assign_path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_calibration_results():
    """Load vlm_calibration.py output JSON files (one per gt_tier)."""
    out = {}
    for tier in ("gold", "broader", "all"):
        path = os.path.join(VLM_RESULTS_DIR, f"calibration_results_{tier}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    out[tier] = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
    return out


@st.cache_data(ttl=60)
def load_calibration_set():
    """Load anchored/blind calibration split for anchoring bias analysis."""
    cal_path = os.path.join(PROJECT_ROOT, "active_batch", "data", "calibration_set.json")
    if not os.path.exists(cal_path):
        return None
    try:
        with open(cal_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=30)
def load_all_annotations():
    """Load all human annotations, excluding inactive annotators."""
    all_annotations = {}
    if not os.path.exists(ANNOTATIONS_DIR):
        return all_annotations
    for annotator_id in os.listdir(ANNOTATIONS_DIR):
        if annotator_id in INACTIVE_ANNOTATORS:
            continue
        ann_dir = os.path.join(ANNOTATIONS_DIR, annotator_id)
        if not os.path.isdir(ann_dir):
            continue
        all_annotations[annotator_id] = {}
        for fname in os.listdir(ann_dir):
            if fname.endswith(".json"):
                with open(os.path.join(ann_dir, fname)) as f:
                    ann = json.load(f)
                    all_annotations[annotator_id][ann.get("clip_id", "")] = ann
    return all_annotations


# ---------- Dashboard ----------

def main():
    st.set_page_config(
        page_title="SteelBench QC Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("SteelBench — Unified QC Dashboard")
    st.caption("Real-time monitoring of extraction, annotation, and quality control")

    # Load all data once (cached — repeated calls within 30s are free)
    extraction = load_extraction_stats()
    filtering = load_filtering_stats()
    tier_a = load_tier_a_manifest()
    vlm_results = load_vlm_results()
    assignments = load_assignments()
    all_annotations = load_all_annotations()

    # Create tabs for each section
    (tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,
     tab9, tab10, tab11, tab12) = st.tabs([
        "Extraction Pipeline",
        "VLM Pre-annotation",
        "Human Annotation",
        "Routing & Assignments",
        "Quality Alerts",
        "QA Mechanisms",
        "Safety Review",
        "Open Vocabulary (X1)",
        "Schema Health",
        "Flag Tracker",
        "Expert Edit Audit",
        "Phase 2 Progress",
    ])

    # ==================== TAB 1: EXTRACTION PIPELINE ====================
    with tab1:
        st.header("Extraction Pipeline Status")

        if extraction:
            stats = extraction.get("stats", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Videos Processed", stats.get("videos_processed", 0))
            col2.metric("Clips Extracted", stats.get("clips_extracted", 0))
            total_dur = stats.get("total_clip_duration_sec", 0)
            col3.metric("Total Duration", f"{total_dur/3600:.1f} hrs")
            col4.metric("Last Updated", extraction.get("last_updated", "")[:19])

            # Videos progress
            total_videos = len(extraction.get("completed_videos", []))
            st.caption(f"Processed {total_videos} videos from /<RAW_VIDEO_DIR>/")
        else:
            st.info("No extraction data. Run extract_clips.py first.")

        st.divider()

        if filtering:
            st.subheader("Quality Filtering Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Filtered", filtering.get("total_processed", 0))
            col2.metric("Tier A (Annotation-Ready)",
                       filtering.get("tier_a_count", len(tier_a)))
            col3.metric("Tier B (Degradation Pool)",
                       filtering.get("tier_b_count", 0))

            # Rejection reasons
            reasons = filtering.get("tier_b_reasons", {})
            if reasons:
                st.subheader("Tier B Rejection Reasons")
                st.bar_chart(reasons)
        elif tier_a:
            st.metric("Tier A Clips Available", len(tier_a))

        # Manifest stats
        if tier_a:
            st.subheader("Tier A Manifest Overview")
            cameras = Counter(r.get("camera_id", "") for r in tier_a)
            conditions = Counter(r.get("visual_condition_auto", "") for r in tier_a)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Camera**")
                st.bar_chart(dict(cameras))
            with col2:
                st.markdown("**By Visual Condition**")
                st.bar_chart(dict(conditions))

    # ==================== TAB 2: VLM PRE-ANNOTATION ====================
    with tab2:
        st.header("VLM Pre-annotation Quality")
        st.caption("Single VLM (Qwen3.5-35B-A3B) used for annotation prefill")

        if not vlm_results:
            st.info("No VLM results. Run vlm_annotator first.")
        else:
            # Extract VLM stats from single-VLM results
            vlm_success = 0
            vlm_fail = 0
            vlm_model = "unknown"
            layer1_clips = 0
            layer2_clips = 0
            action_dist = Counter()
            scene_dist = Counter()
            ppe_compliance_dist = Counter()  # NEW: overall_ppe_compliance histogram

            for r in vlm_results:
                # Use whichever VLM has data
                vlm_data = r.get("vlm") or {}
                if not vlm_model or vlm_model == "unknown":
                    vlm_model = vlm_data.get("model", "unknown")

                if vlm_data.get("success"):
                    vlm_success += 1
                    norm = vlm_data.get("normalized") or {}
                    n_workers = len(norm.get("persons", []))
                    if n_workers > 5:
                        layer1_clips += 1
                    else:
                        layer2_clips += 1
                    scene_dist[norm.get("scene_type", "?")] += 1
                    ppe_compliance_dist[norm.get("overall_ppe_compliance", "?")] += 1
                    for p in norm.get("persons", []):
                        ac = p.get("action_code", "")
                        if ac:
                            action_dist[ac] += 1
                else:
                    vlm_fail += 1

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Processed", len(vlm_results))
            col2.metric("VLM Success", vlm_success)
            col3.metric("Parse Failures", vlm_fail,
                       delta=f"{100*vlm_fail/max(len(vlm_results),1):.1f}%"
                       if vlm_fail > 0 else None)
            col4.metric("Model", vlm_model)

            # Layer distribution
            l1, l2 = st.columns(2)
            l1.metric("Layer 1 clips (6+ workers)", layer1_clips)
            l2.metric("Layer 2 clips (≤5 workers)", layer2_clips)

            st.divider()

            # Scene type and action distribution
            if scene_dist:
                st.subheader("VLM Scene Type Distribution")
                st.dataframe(
                    [{"Scene Type": k, "Count": v} for k, v in scene_dist.most_common()],
                    use_container_width=True)

            if action_dist:
                st.subheader("VLM Action Distribution (top 15)")
                st.dataframe(
                    [{"Action": k, "Count": v} for k, v in action_dist.most_common(15)],
                    use_container_width=True)

            if ppe_compliance_dist:
                st.subheader("VLM Overall PPE Compliance")
                total_compl = sum(ppe_compliance_dist.values())
                st.dataframe(
                    [{"Compliance": k, "Count": v,
                      "%": f"{100*v/max(total_compl,1):.1f}"}
                     for k, v in ppe_compliance_dist.most_common()],
                    use_container_width=True)

            # VLM worker count distribution
            st.subheader("VLM Worker Count Distribution")
            worker_counts = []
            for r in vlm_results:
                vlm = r.get("vlm") or {}
                norm = vlm.get("normalized") or {}
                wc = len(norm.get("persons", []))
                if wc > 0:
                    worker_counts.append(wc)
            if worker_counts:
                from collections import Counter as WC_Counter
                wc_dist = WC_Counter(worker_counts)
                st.bar_chart(dict(sorted(wc_dist.items())))

            st.divider()

            # Additional field distributions (per-worker)
            st.subheader("VLM Field Coverage & Distributions")
            group_flag_dist = Counter()
            unsafe_act_dist = Counter()
            spatial_dist = Counter()
            visibility_dist = Counter()
            ppe_status = {item: Counter() for item in
                          ["helmet", "high_vis_vest", "welding_protection", "harness", "safety_shoes"]}
            total_workers_t2 = 0
            phys_desc_count = 0
            free_desc_count = 0

            for r in vlm_results:
                vlm = r.get("vlm") or {}
                if not vlm.get("success"):
                    continue
                norm = vlm.get("normalized") or {}
                # Visibility (clip level)
                for vc in norm.get("visibility_conditions", []):
                    visibility_dist[vc] += 1
                for p in norm.get("persons", []):
                    total_workers_t2 += 1
                    gf = p.get("group_flag", "solo")
                    group_flag_dist[gf] += 1
                    ua = p.get("unsafe_act", "none")
                    if ua and ua != "none":
                        unsafe_act_dist[ua] += 1
                    for tag in p.get("spatial_context", []):
                        spatial_dist[tag] += 1
                    if p.get("physical_description"):
                        phys_desc_count += 1
                    if p.get("free_text_description"):
                        free_desc_count += 1
                    ppe = p.get("ppe", {})
                    for item in ppe_status:
                        ppe_status[item][ppe.get(item, "missing")] += 1

            # Field completeness metrics
            if total_workers_t2 > 0:
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("physical_description coverage",
                           f"{100*phys_desc_count/total_workers_t2:.0f}%",
                           help="Workers with physical description filled")
                fc2.metric("free_text_description coverage",
                           f"{100*free_desc_count/total_workers_t2:.0f}%",
                           help="Workers with free-text action description")
                fc3.metric("Workers with unsafe acts",
                           f"{sum(unsafe_act_dist.values())} ({100*sum(unsafe_act_dist.values())/total_workers_t2:.0f}%)")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Group Flag Distribution**")
                    st.bar_chart(dict(group_flag_dist))
                with col_b:
                    st.markdown("**Visibility Conditions (clip-level)**")
                    if visibility_dist:
                        st.bar_chart(dict(visibility_dist))
                    else:
                        st.caption("No visibility data")

                st.markdown("**Spatial Context Distribution**")
                if spatial_dist:
                    st.dataframe(
                        [{"Tag": k, "Count": v} for k, v in spatial_dist.most_common()],
                        use_container_width=True)

                st.markdown("**PPE Distribution (per item)**")
                ppe_table = []
                for item, statuses in ppe_status.items():
                    total = sum(statuses.values())
                    ppe_table.append({
                        "Item": item,
                        "present": statuses.get("present", 0),
                        "absent": statuses.get("absent", 0),
                        "cannot_determine": statuses.get("cannot_determine", 0),
                        "not_applicable": statuses.get("not_applicable", 0),
                    })
                st.dataframe(ppe_table, use_container_width=True)

                if unsafe_act_dist:
                    st.markdown("**Unsafe Acts (top 10)**")
                    st.dataframe(
                        [{"Unsafe Act": k, "Count": v} for k, v in unsafe_act_dist.most_common(10)],
                        use_container_width=True)

    # ==================== TAB 3: HUMAN ANNOTATION ====================
    with tab3:
        st.header("Human Annotation Progress")

        total_tier_a = len(tier_a)

        if not all_annotations:
            st.info("No human annotations yet. Launch the annotation UI with: "
                   "streamlit run annotation_tool/app.py")
        else:
            annotator_ids = sorted(all_annotations.keys())
            total_ann = sum(len(anns) for anns in all_annotations.values())
            unique_clips = set()
            for anns in all_annotations.values():
                unique_clips.update(anns.keys())

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Annotations", total_ann)
            col2.metric("Unique Clips", len(unique_clips))
            col3.metric("Tier A Total", total_tier_a)
            col4.metric("Active Annotators", len(annotator_ids))

            if total_tier_a > 0:
                st.progress(min(1.0, len(unique_clips) / total_tier_a))

            # Per-annotator table
            st.subheader("Per-Annotator Statistics")
            stats_rows = []
            for aid in annotator_ids:
                anns = all_annotations[aid]
                submitted = sum(1 for a in anns.values()
                               if a.get("status") == "submitted")
                skipped = sum(1 for a in anns.values()
                             if a.get("status") == "skipped")
                flagged = sum(1 for a in anns.values()
                             if a.get("status") == "flagged")
                discarded = sum(1 for a in anns.values()
                               if a.get("status") == "discarded")
                confs = [a.get("annotator_confidence", 0)
                         for a in anns.values() if a.get("status") == "submitted"]
                avg_conf = np.mean(confs) if confs else 0

                stats_rows.append({
                    "Annotator": aid,
                    "Submitted": submitted,
                    "Skipped": skipped,
                    "Flagged": flagged,
                    "Discarded": discarded,
                    "Total": len(anns),
                    "Avg Confidence": round(avg_conf, 3),
                })
            st.dataframe(stats_rows, width="stretch")

            # Inter-Annotator Agreement (split by layer)
            st.subheader("Inter-Annotator Agreement (IAA)")

            clip_annotators = defaultdict(list)
            for aid, anns in all_annotations.items():
                for cid, ann in anns.items():
                    if ann.get("status") == "submitted":
                        clip_annotators[cid].append((aid, ann))

            double_annotated = {
                cid: anns for cid, anns in clip_annotators.items()
                if len(anns) >= 2
            }

            # Split by annotation_layer
            layer1_pairs = {}
            layer2_pairs = {}
            for cid, anns in double_annotated.items():
                layer = anns[0][1].get("annotation_layer",
                                       2 if anns[0][1].get("persons") else 1)
                if layer == 1:
                    layer1_pairs[cid] = anns
                else:
                    layer2_pairs[cid] = anns

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Double-Annotated", len(double_annotated))
            col2.metric("Layer 1 (scene-level)", len(layer1_pairs))
            col3.metric("Layer 2 (per-person)", len(layer2_pairs))

            # Layer 2: per-person IAA via existing compute_all_iaa
            st.markdown("**Layer 2 (per-person) IAA**")
            if len(layer2_pairs) >= 10:
                annotations_a = [anns[0][1] for anns in layer2_pairs.values()]
                annotations_b = [anns[1][1] for anns in layer2_pairs.values()]
                iaa = compute_all_iaa(annotations_a, annotations_b)

                kappa_data = [
                    {"Axis": "A1: Action", "Kappa": round(iaa["a1_action_kappa"], 3),
                     "Target": 0.75,
                     "Status": "PASS" if iaa["a1_meets_target"] else "BELOW"},
                    {"Axis": "A2: Spatial", "Kappa": round(iaa["a2_spatial_kappa"], 3),
                     "Target": 0.70,
                     "Status": "PASS" if iaa["a2_meets_target"] else "BELOW"},
                    {"Axis": "A3: PPE", "Kappa": round(iaa["a3_ppe_kappa"], 3),
                     "Target": 0.65,
                     "Status": "PASS" if iaa["a3_meets_target"] else "BELOW"},
                ]
                st.dataframe(kappa_data, width="stretch")
            elif layer2_pairs:
                st.warning(f"Only {len(layer2_pairs)} Layer 2 double-annotated. "
                          "Need >= 10 for IAA.")
            else:
                st.info("No Layer 2 double-annotated clips yet.")

            # Layer 1: scene-level agreement (Jaccard on dominant_actions, exact on overall_ppe)
            st.markdown("**Layer 1 (scene-level) IAA**")
            if len(layer1_pairs) >= 10:
                action_jaccard_scores = []
                ppe_match_scores = []
                scene_match_scores = []
                for cid, anns in layer1_pairs.items():
                    a, b = anns[0][1], anns[1][1]
                    # Dominant actions: Jaccard similarity
                    set_a = set(a.get("dominant_actions", []))
                    set_b = set(b.get("dominant_actions", []))
                    if set_a or set_b:
                        jacc = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 1.0
                        action_jaccard_scores.append(jacc)
                    # Overall PPE: exact match
                    ppe_a = a.get("overall_ppe_compliance", "")
                    ppe_b = b.get("overall_ppe_compliance", "")
                    if ppe_a or ppe_b:
                        ppe_match_scores.append(1.0 if ppe_a == ppe_b else 0.0)
                    # Scene type: exact match
                    if a.get("scene_type") and b.get("scene_type"):
                        scene_match_scores.append(
                            1.0 if a["scene_type"] == b["scene_type"] else 0.0)

                layer1_data = []
                if action_jaccard_scores:
                    mean_jacc = sum(action_jaccard_scores) / len(action_jaccard_scores)
                    layer1_data.append({
                        "Metric": "Dominant Actions (Jaccard)",
                        "Score": round(mean_jacc, 3),
                        "n": len(action_jaccard_scores),
                        "Target": 0.70,
                        "Status": "PASS" if mean_jacc >= 0.70 else "BELOW",
                    })
                if ppe_match_scores:
                    ppe_acc = sum(ppe_match_scores) / len(ppe_match_scores)
                    layer1_data.append({
                        "Metric": "Overall PPE (exact match)",
                        "Score": round(ppe_acc, 3),
                        "n": len(ppe_match_scores),
                        "Target": 0.80,
                        "Status": "PASS" if ppe_acc >= 0.80 else "BELOW",
                    })
                if scene_match_scores:
                    scene_acc = sum(scene_match_scores) / len(scene_match_scores)
                    layer1_data.append({
                        "Metric": "Scene Type (exact match)",
                        "Score": round(scene_acc, 3),
                        "n": len(scene_match_scores),
                        "Target": 0.85,
                        "Status": "PASS" if scene_acc >= 0.85 else "BELOW",
                    })
                if layer1_data:
                    st.dataframe(layer1_data, width="stretch")
                else:
                    st.info("No scene-level data to compare.")
            elif layer1_pairs:
                st.warning(f"Only {len(layer1_pairs)} Layer 1 double-annotated. "
                          "Need >= 10 for IAA.")
            else:
                st.info("No Layer 1 double-annotated clips yet.")

            # Action distribution (combine Layer 1 dominant_actions + Layer 2 per-person)
            st.subheader("Annotation Distribution")
            action_counts = Counter()
            scene_counts = Counter()
            l1_count = 0
            l2_count = 0
            for anns in all_annotations.values():
                for ann in anns.values():
                    if ann.get("status") != "submitted":
                        continue
                    layer = ann.get("annotation_layer",
                                    2 if ann.get("persons") else 1)
                    # Use scene_type field directly (from app.py)
                    st_type = ann.get("scene_type", "?")
                    if st_type:
                        scene_counts[st_type] += 1
                    if layer == 1:
                        l1_count += 1
                        # Layer 1: count dominant_actions (one per action)
                        for ac in ann.get("dominant_actions", []):
                            action_counts[ac] += 1
                    else:
                        l2_count += 1
                        # Layer 2: count per-person actions
                        for p in ann.get("persons", []):
                            action_counts[p.get("action_code", "?")] += 1

            l1c, l2c = st.columns(2)
            l1c.metric("Layer 1 annotations", l1_count)
            l2c.metric("Layer 2 annotations", l2_count)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Action Classes**")
                if action_counts:
                    st.bar_chart(dict(action_counts.most_common(15)))
                else:
                    st.info("No data")
            with col2:
                st.markdown("**Scene Types**")
                if scene_counts:
                    st.bar_chart(dict(scene_counts))
                else:
                    st.info("No data")

    # ==================== TAB 4: ROUTING & ASSIGNMENTS ====================
    with tab4:
        st.header("Routing & Assignment Status")

        if vlm_results:
            vlm_ok = [r for r in vlm_results if r.get("vlm", {}).get("success")]

            # VLM confidence distribution (single VLM policy: all clips need human review)
            if vlm_ok:
                # Compute mean confidence per clip
                confidences = []
                for r in vlm_ok:
                    persons = r.get("vlm", {}).get("normalized", {}).get("persons", [])
                    confs = [p.get("confidence", 0.0) for p in persons
                             if p.get("confidence") is not None]
                    if confs:
                        confidences.append(sum(confs) / len(confs))

                st.subheader("VLM Confidence Distribution")
                st.caption("Single VLM policy: all clips routed to full human review. "
                           "Confidence used for prioritization (low → tier_2 expert).")
                if confidences:
                    high = sum(1 for c in confidences if c >= 0.80)
                    medium = sum(1 for c in confidences if 0.50 <= c < 0.80)
                    low = sum(1 for c in confidences if c < 0.50)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("High VLM Confidence",
                                f"{high} ({100*high/len(confidences):.0f}%)",
                                help="VLM mean confidence ≥0.80 — annotator likely agrees")
                    col2.metric("Medium VLM Confidence",
                                f"{medium} ({100*medium/len(confidences):.0f}%)",
                                help="VLM mean confidence 0.50-0.80")
                    col3.metric("Low VLM Confidence",
                                f"{low} ({100*low/len(confidences):.0f}%)",
                                help="VLM mean confidence <0.50 — route to tier_2 expert")
                    st.metric("Mean VLM Confidence",
                              f"{sum(confidences)/len(confidences):.2f}")

                # Human effort: 100% of clips need review under single-VLM policy
                st.metric("Human Effort Required",
                          f"{len(vlm_results)}/{len(vlm_results)} clips (100%)",
                          help="All clips routed to human verification")

        if assignments:
            st.divider()
            st.subheader("Annotator Assignments")

            assign_rows = []
            for aid, data in assignments.get("assignments", {}).items():
                if aid in INACTIVE_ANNOTATORS:
                    continue
                clips_list = data.get("clips", [])
                assign_rows.append({
                    "Annotator": data.get("annotator_id", aid),
                    "Role": data.get("role", "tier_1"),
                    "Total Clips": data.get("total_clips", len(clips_list)),
                    "Fresh Annotation": sum(1 for c in clips_list if c.get("task") == "fresh_annotation"),
                    "Review Pre-fill": sum(1 for c in clips_list if c.get("task") == "review_prefill"),
                    "Verify Consensus": sum(1 for c in clips_list if c.get("task") == "verify_consensus"),
                    "Safety Review": sum(1 for c in clips_list if c.get("task") == "safety_verification"),
                })
            st.dataframe(assign_rows, width="stretch")

            # Assignment completion tracking
            if all_annotations:
                st.subheader("Assignment Completion")
                for aid, data in assignments.get("assignments", {}).items():
                    if aid in INACTIVE_ANNOTATORS:
                        continue
                    assigned_clips = {c["clip_id"] for c in data.get("clips", [])}
                    completed = assigned_clips & set(all_annotations.get(aid, {}).keys())
                    remaining = len(assigned_clips) - len(completed)
                    pct = len(completed) / len(assigned_clips) if assigned_clips else 0
                    st.progress(pct, text=f"{aid}: {len(completed)}/{len(assigned_clips)} "
                               f"({remaining} remaining)")

        if not vlm_results and not assignments:
            st.info("No routing or assignment data available.")

    # ==================== TAB 5: QUALITY ALERTS ====================
    with tab5:
        st.header("Quality Alerts & Health Checks")

        alerts = []
        warnings = []
        successes = []

        # Check extraction
        if extraction:
            stats = extraction.get("stats", {})
            if stats.get("videos_processed", 0) > 0:
                successes.append(
                    f"Extraction: {stats['videos_processed']} videos processed, "
                    f"{stats['clips_extracted']} clips extracted")
            else:
                alerts.append("Extraction: No videos processed yet")

        # Check VLM results
        if vlm_results:
            vlm_success = sum(1 for r in vlm_results
                              if r.get("vlm", {}).get("success"))
            success_rate = vlm_success / len(vlm_results)
            if success_rate >= 0.80:
                successes.append(
                    f"VLM success rate: {vlm_success}/{len(vlm_results)} "
                    f"({100*success_rate:.0f}%)")
            else:
                warnings.append(
                    f"VLM success rate low: {vlm_success}/{len(vlm_results)} "
                    f"({100*success_rate:.0f}%). "
                    "Check API rate limits or model availability.")

            # Parse failure tracking
            vlm_ok = [r for r in vlm_results if r.get("vlm", {}).get("success")]
            if vlm_ok and False:  # Agreement N/A for single-VLM
                composites = [0.0 for r in vlm_ok]
                mean_comp = np.mean(composites)
                if mean_comp < 0.50:
                    alerts.append(
                        f"Low VLM composite agreement: {mean_comp:.3f}. "
                        "Models may need calibration or prompt tuning.")
                elif mean_comp < 0.70:
                    warnings.append(
                        f"Moderate VLM agreement: {mean_comp:.3f}. "
                        "Consider prompt improvements for higher auto-accept rate.")
                else:
                    successes.append(f"VLM composite agreement: {mean_comp:.3f}")

        # Check human annotations
        if all_annotations:
            for aid in sorted(all_annotations.keys()):
                anns = all_annotations[aid]
                if len(anns) >= 10:
                    skip_rate = sum(1 for a in anns.values()
                                   if a.get("status") == "skipped") / len(anns)
                    if skip_rate > 0.20:
                        alerts.append(
                            f"Annotator {aid}: high skip rate {skip_rate:.0%} "
                            "(threshold: 20%)")

                    low_conf = sum(1 for a in anns.values()
                                  if a.get("status") == "submitted"
                                  and a.get("annotator_confidence", 1.0) < 0.5)
                    if low_conf > 5:
                        warnings.append(
                            f"Annotator {aid}: {low_conf} low-confidence submissions")

        # Display alerts
        if alerts:
            st.subheader("Alerts")
            for a in alerts:
                st.error(a)

        if warnings:
            st.subheader("Warnings")
            for w in warnings:
                st.warning(w)

        if successes:
            st.subheader("Health Checks Passed")
            for s in successes:
                st.success(s)

        if not alerts and not warnings and not successes:
            st.info("No data available for health checks. "
                   "Run the pipeline stages first.")

        # Flagged clips
        st.divider()
        st.subheader("Flagged Clips")
        flagged = []
        for aid, anns in all_annotations.items():
            for cid, ann in anns.items():
                if ann.get("status") == "flagged":
                    flagged.append({
                        "Clip": cid,
                        "Annotator": aid,
                        "Time": ann.get("annotator_timestamp", ""),
                    })
        if flagged:
            st.dataframe(flagged, width="stretch")
        else:
            st.info("No flagged clips.")

    # ==================== TAB 6: QA MECHANISMS ====================
    with tab6:
        st.header("QA Mechanisms (QA-1 through QA-6)")

        st.subheader("QA-1: VLM Confidence Calibration (ECE)")
        cal_results = load_calibration_results()
        if not cal_results:
            st.info(
                "No calibration_results_*.json found. Run:\n"
                "`python -m annotation_tool.vlm_calibration --gt-tier gold "
                "--config experiments/gt_batch_125.yaml --output annotation_tool/data/vlm_results/calibration_results_gold.json`"
            )
        else:
            def _show_layer_calibration(layer_data, layer_label):
                """Render one layer's calibration sub-report."""
                n = layer_data.get("n_total", 0)
                if n == 0:
                    st.caption(f"{layer_label}: {layer_data.get('warning', 'no data')}")
                    return
                ece_data = layer_data.get("ece", {})
                label_kind = layer_data.get("label_kind", "?")
                st.markdown(f"**{layer_label}** ({label_kind}, n={n})")
                cols = st.columns(4)
                cols[0].metric("ECE", f"{ece_data.get('ece', 0):.4f}")
                cols[1].metric("Brier", f"{layer_data.get('brier', 0):.4f}")
                cols[2].metric("Accuracy", f"{layer_data.get('accuracy', 0):.1%}")
                cols[3].metric("Avg Conf", f"{layer_data.get('avg_confidence', 0):.1%}")
                if layer_data.get("sample_size_warning"):
                    st.warning(f"{layer_label} sample size < 30 — interpret with caution.")

                # Reliability diagram
                non_empty = [b for b in ece_data.get("bins", []) if b.get("count")]
                if non_empty:
                    chart = {}
                    for b in non_empty:
                        label = f"{b['lo']:.1f}-{b['hi']:.1f}"
                        chart[label] = {
                            "accuracy": b["accuracy"],
                            "avg_confidence": b["avg_confidence"],
                        }
                    st.markdown(f"_{layer_label} Reliability Diagram (perfect calibration: lines overlap)_")
                    st.line_chart(chart)

                # Per-label ECE table
                pa = layer_data.get("per_label") or {}
                if pa:
                    rows = sorted(
                        [
                            {
                                "label": k,
                                "n": v["n"],
                                "ECE": round(v["ece"], 4),
                                "accuracy": v.get("accuracy"),
                                "avg_conf": v.get("avg_confidence"),
                            }
                            for k, v in pa.items()
                        ],
                        key=lambda r: -r["ECE"],
                    )
                    st.markdown(f"_{layer_label} Per-label ECE (worst-first)_")
                    st.dataframe(rows, width="stretch")

            for tier, data in cal_results.items():
                st.markdown(
                    f"### Ground Truth Tier: `{tier}` "
                    f"(total n={data.get('n_total', 0)}, "
                    f"Layer 1: {data.get('n_layer_1', 0)}, "
                    f"Layer 2: {data.get('n_layer_2', 0)})"
                )
                if data.get("n_total", 0) == 0:
                    st.caption(f"No data: {data.get('warning', 'unknown')}")
                    continue
                if "combined_ece_weighted" in data:
                    st.metric(
                        "Combined ECE (weighted across layers)",
                        f"{data['combined_ece_weighted']:.4f}",
                    )
                if data.get("sample_size_warning"):
                    st.warning("Total sample size < 100 — interpret with caution.")

                if "layer_2" in data:
                    _show_layer_calibration(data["layer_2"], "Layer 2")
                if "layer_1" in data:
                    _show_layer_calibration(data["layer_1"], "Layer 1")
                st.divider()

        st.divider()
        st.subheader("QA-2: Human-Human Agreement (IAA)")
        if all_annotations:
            # Filter to tier_1 annotators only for IAA
            tier1_ids = {aid for aid, anns in all_annotations.items()
                         if "annotator" in aid}

            # Find double-annotated clips (tier_1 vs tier_1 only)
            clip_anns = defaultdict(list)
            for aid in tier1_ids:
                for cid, ann in all_annotations.get(aid, {}).items():
                    if ann.get("status") == "submitted":
                        clip_anns[cid].append(ann)
            double = {c: a for c, a in clip_anns.items() if len(a) >= 2}

            # Split by calibration condition
            cal_set = load_calibration_set()
            blind_ids = set((cal_set or {}).get("blind_clips", {}).keys())
            double_blind = {c: a for c, a in double.items()
                           if c in blind_ids
                           and all(ann.get("calibration_condition") == "blind"
                                  for ann in a)}
            double_anchored = {c: a for c, a in double.items()
                               if c not in double_blind}

            st.metric("Double-Annotated Clips (tier_1 only)", len(double))
            cols = st.columns(2)
            cols[0].metric("Blind pairs (independent)", len(double_blind))
            cols[1].metric("Anchored pairs (shared VLM)", len(double_anchored))

            def _show_iaa(label, pairs):
                if len(pairs) < 5:
                    st.caption(f"{label}: need >= 5 pairs (have {len(pairs)})")
                    return
                ann_a = [list(a)[0] for a in pairs.values()]
                ann_b = [list(a)[1] for a in pairs.values()]
                iaa = compute_all_iaa(ann_a, ann_b)
                st.dataframe([
                    {"Axis": "Action", "Kappa": iaa["a1_action_kappa"],
                     "Target": 0.75, "Pass": iaa["a1_meets_target"]},
                    {"Axis": "Spatial", "Kappa": iaa["a2_spatial_kappa"],
                     "Target": 0.70, "Pass": iaa["a2_meets_target"]},
                    {"Axis": "PPE", "Kappa": iaa["a3_ppe_kappa"],
                     "Target": 0.65, "Pass": iaa["a3_meets_target"]},
                ], width="stretch")

            st.markdown("**All pairs (anchored + blind)**")
            _show_iaa("All", double)

            if double_blind:
                st.markdown("**Blind pairs only (independent IAA)**")
                _show_iaa("Blind", double_blind)

            if double_anchored:
                st.markdown("**Anchored pairs (shared VLM pre-fill)**")
                _show_iaa("Anchored", double_anchored)

            st.caption(
                "NOTE: Blind IAA measures true independent agreement. "
                "Anchored IAA is inflated by shared VLM pre-fill. "
                "Report both in the paper."
            )
        else:
            st.info("No annotations yet.")

        st.divider()
        st.subheader("QA-3: VLM Override Rate")
        if all_annotations and vlm_results:
            from annotation_tool.qc_metrics import vlm_override_rate as vor_fn
            # Tier_1 annotators only — experts/SO have different baselines
            all_human = []
            for aid, anns in all_annotations.items():
                if "annotator" not in aid:
                    continue  # skip expert_*, safety_officer
                for ann in anns.values():
                    if ann.get("status") == "submitted":
                        all_human.append(ann)
            vlm_map = {}
            for r in vlm_results:
                if r.get("vlm", {}).get("success"):
                    vlm_map[r["clip_id"]] = r.get("vlm", {})["normalized"]
            if all_human and vlm_map:
                vor = vor_fn(all_human, vlm_map)
                # Top-level summary across both layers
                top_cols = st.columns(3)
                top_cols[0].metric(
                    "Combined Override Rate",
                    f"{vor['combined_override_rate']:.1%}",
                )
                top_cols[1].metric("Total Clips", vor["total_clips"])
                ci = vor.get("combined_ci_95", (0, 0))
                top_cols[2].metric(
                    "95% CI",
                    f"{ci[0]:.1%}–{ci[1]:.1%}",
                )

                # Per-layer breakdown
                l1 = vor.get("layer_1", {})
                l2 = vor.get("layer_2", {})

                st.markdown("**Layer 2 (per-person)**")
                if l2.get("n_clips", 0) > 0:
                    cols = st.columns(2)
                    cols[0].metric(
                        "Layer 2 Override Rate",
                        f"{l2.get('overall_override_rate', 0):.1%}",
                    )
                    cols[1].metric("Layer 2 Clips", l2.get("n_clips", 0))
                    st.json(l2.get("per_axis_rates", {}))
                else:
                    st.caption("No Layer 2 records yet.")

                st.markdown("**Layer 1 (scene-level)**")
                if l1.get("n_clips", 0) > 0:
                    cols = st.columns(2)
                    cols[0].metric(
                        "Layer 1 Override Rate",
                        f"{l1.get('overall_override_rate', 0):.1%}",
                    )
                    cols[1].metric("Layer 1 Clips", l1.get("n_clips", 0))
                    st.json(l1.get("per_axis_rates", {}))
                else:
                    st.caption("No Layer 1 records yet.")
            else:
                st.info("Need both human annotations and VLM results.")
        else:
            st.info("Need both human annotations and VLM results.")

        st.divider()
        st.subheader("QA-4: Anchoring Bias")
        cal_set = load_calibration_set()
        if not cal_set:
            st.info(
                "No calibration_set.json found. Run:\n"
                "`python scripts/create_phase2_calibration.py --size 250`"
            )
        elif not all_annotations:
            st.info("No annotations submitted yet.")
        else:
            anchored_ids = set(cal_set.get("anchored_clips", {}).keys())
            blind_ids = set(cal_set.get("blind_clips", {}).keys())
            st.caption(
                f"Calibration subset: {len(anchored_ids)} anchored + "
                f"{len(blind_ids)} blind = {cal_set.get('total', 0)} total"
            )

            # Build vlm_map for proxy comparison
            vlm_map = {}
            for r in vlm_results:
                if r.get("vlm", {}).get("success"):
                    vlm_map[r["clip_id"]] = (r.get("vlm", {}).get("normalized") or {})

            # Tier_1 annotators only for anchoring bias
            # Exclude contaminated blind clips (calibration_condition="contaminated")
            l2_anchored = []
            l2_blind = []
            l1_anchored = []
            l1_blind = []
            n_contaminated = 0

            for aid, anns in all_annotations.items():
                if "annotator" not in aid:
                    continue  # tier_1 only
                for cid, ann in anns.items():
                    if ann.get("status") != "submitted":
                        continue
                    # Skip contaminated blind clips
                    if ann.get("calibration_condition") == "contaminated":
                        n_contaminated += 1
                        continue
                    ann_layer = ann.get("annotation_layer")
                    vlm_norm = vlm_map.get(cid) or {}

                    if ann_layer == 2:
                        persons = ann.get("persons") or []
                        v_persons = vlm_norm.get("persons") or []
                        if not persons or not v_persons:
                            continue
                        h_label = persons[0].get("action_code", "")
                        v_label = v_persons[0].get("action_code", "")
                        if not h_label or not v_label:
                            continue
                        if cid in anchored_ids:
                            l2_anchored.append((h_label, v_label))
                        elif cid in blind_ids:
                            l2_blind.append((h_label, v_label))
                    elif ann_layer == 1:
                        h_label = (ann.get("scene_type") or "").strip()
                        v_label = (vlm_norm.get("scene_type") or "").strip()
                        if not h_label or not v_label:
                            continue
                        if cid in anchored_ids:
                            l1_anchored.append((h_label, v_label))
                        elif cid in blind_ids:
                            l1_blind.append((h_label, v_label))

            if n_contaminated:
                st.warning(
                    f"{n_contaminated} contaminated blind clips excluded "
                    "(annotated before calibration deployment, VLM was shown)."
                )

            from annotation_tool.qc_metrics import anchoring_bias_analysis

            def _show_layer(name, anchored_pairs, blind_pairs, label_kind):
                st.markdown(f"**{name}** ({label_kind})")
                if not anchored_pairs or not blind_pairs:
                    st.info(
                        f"Need pairs in both conditions for {name}. "
                        f"Anchored: {len(anchored_pairs)}, Blind: {len(blind_pairs)}"
                    )
                    return None
                result = anchoring_bias_analysis(anchored_pairs, blind_pairs)
                cols = st.columns(4)
                cols[0].metric(
                    "Anchored agreement (vs VLM)",
                    f"{result['anchored_accuracy']:.1%}",
                )
                cols[1].metric(
                    "Blind agreement (vs VLM)",
                    f"{result['blind_accuracy']:.1%}",
                )
                cols[2].metric(
                    "Bias magnitude", f"{result['bias_magnitude']:+.1%}"
                )
                cols[3].metric(
                    "Significant", "Yes" if result.get("significant") else "No"
                )
                st.caption(result.get("recommendation", ""))
                return result

            l2_result = _show_layer(
                "Layer 2", l2_anchored, l2_blind, "persons[0].action_code"
            )
            st.markdown("")
            l1_result = _show_layer(
                "Layer 1", l1_anchored, l1_blind, "scene_type"
            )

            st.caption(
                "NOTE: Both layers compare annotator→VLM agreement (proxy). "
                "True anchoring bias compares anchored vs blind accuracy "
                "against expert ground truth. Layer 2 measures per-person "
                "action; Layer 1 measures scene type — they are NOT comparable "
                "as a single number."
            )

        st.divider()
        st.subheader("QA-5: Decoy Catch Rate")
        st.info("Requires decoy clips with deliberately wrong labels. "
               "Configure decoy_ratio in experiment config.")

        st.divider()
        st.subheader("QA-6: Confusion Matrix")
        if all_annotations:
            from annotation_tool.qc_metrics import confusion_matrix as cm_fn
            all_actions = []
            for anns in all_annotations.values():
                for ann in anns.values():
                    if ann.get("status") == "submitted" and ann.get("persons"):
                        all_actions.append(
                            ann["persons"][0].get("action_code", "?"))
            if len(all_actions) >= 5:
                st.caption(f"Action distribution from {len(all_actions)} annotations")
                from collections import Counter as C
                dist = dict(C(all_actions).most_common(15))
                st.bar_chart(dist)
            else:
                st.info("Need at least 5 submitted annotations for analysis.")
        else:
            st.info("No annotations yet.")

    # ==================== TAB 7: SAFETY REVIEW ====================
    with tab7:
        st.header("Safety Review Status")

        # Check for safety-critical clips in assignments
        safety_clips = []
        if assignments:
            safety_clips = assignments.get("safety_critical_clips", [])

        col1, col2, col3 = st.columns(3)
        col1.metric("Safety-Critical Clips", len(safety_clips))

        # Count tier_3 annotations
        tier3_done = 0
        if all_annotations:
            for aid, anns in all_annotations.items():
                for ann in anns.values():
                    if ann.get("annotator_role") == "tier_3" and ann.get("status") == "submitted":
                        tier3_done += 1
        col2.metric("Tier 3 Reviews Done", tier3_done)
        col3.metric("Remaining", max(0, len(safety_clips) - tier3_done))

        if safety_clips:
            st.progress(min(1.0, tier3_done / len(safety_clips))
                       if safety_clips else 0)

        # Per-rule compliance summary
        st.subheader("Per-Rule Compliance")
        rule_stats = defaultdict(lambda: {"compliant": 0, "violation": 0, "not_evaluable": 0})
        if all_annotations:
            for aid, anns in all_annotations.items():
                for ann in anns.values():
                    sr = ann.get("safety_review")
                    if sr and sr.get("per_rule"):
                        for rule in sr["per_rule"]:
                            rid = rule.get("rule_id", "")
                            status = rule.get("status", "")
                            if rid and status:
                                rule_stats[rid][status] += 1

        if rule_stats:
            rows = []
            for rid in sorted(rule_stats.keys()):
                s = rule_stats[rid]
                total = s["compliant"] + s["violation"] + s["not_evaluable"]
                rows.append({
                    "Rule": rid,
                    "Compliant": s["compliant"],
                    "Violation": s["violation"],
                    "Not Evaluable": s["not_evaluable"],
                    "Violation Rate": f"{s['violation']/total:.0%}" if total > 0 else "n/a",
                })
            st.dataframe(rows, width="stretch")
        else:
            st.info("No safety reviews completed yet. "
                   "Tier 3 annotators will review safety-critical clips.")

        # Overall safety compliance (from safety officer's independent judgment)
        st.subheader("Overall Safety Compliance")
        overall_stats = {"compliant": 0, "not_compliant": 0, "cannot_determine": 0}
        safety_descriptions = []

        # NEW: explicit VLM agreement tracking from safety_review.agree_with_vlm
        vlm_agree_count = 0
        vlm_disagree_count = 0
        disagree_details = []
        vlm_flagged_total = 0
        vlm_safe_total = 0

        if all_annotations:
            for aid, anns in all_annotations.items():
                for ann in anns.values():
                    sr = ann.get("safety_review")
                    if not sr:
                        continue
                    if sr.get("overall_compliance"):
                        oc = sr["overall_compliance"]
                        overall_stats[oc] = overall_stats.get(oc, 0) + 1
                        if oc == "not_compliant" and sr.get("safety_description"):
                            safety_descriptions.append({
                                "clip_id": ann.get("clip_id", ""),
                                "description": sr["safety_description"],
                            })

                    # NEW: Use explicit agree_with_vlm field
                    if sr.get("vlm_flagged_unsafe"):
                        vlm_flagged_total += 1
                    else:
                        vlm_safe_total += 1

                    if sr.get("agree_with_vlm") == "Agree":
                        vlm_agree_count += 1
                    elif sr.get("agree_with_vlm") == "Disagree":
                        vlm_disagree_count += 1
                        disagree_details.append({
                            "clip_id": ann.get("clip_id", ""),
                            "vlm_flagged": "unsafe" if sr.get("vlm_flagged_unsafe") else "safe",
                            "vlm_workers_flagged": len(sr.get("vlm_unsafe_workers", [])),
                            "officer_compliance": sr.get("overall_compliance", ""),
                            "reason": sr.get("disagree_reason", ""),
                        })

            total_reviewed = sum(overall_stats.values())
            if total_reviewed > 0:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Compliant", overall_stats["compliant"])
                s2.metric("Not Compliant", overall_stats["not_compliant"])
                s3.metric("Cannot Determine", overall_stats["cannot_determine"])
                s4.metric("Total Reviewed", total_reviewed)

                # Safety officer vs VLM agreement (from explicit agree button)
                st.markdown("**Safety Officer vs VLM Agreement (from explicit agree/disagree):**")
                ag1, ag2, ag3, ag4 = st.columns(4)
                ag1.metric("VLM-flagged (any worker unsafe)", vlm_flagged_total)
                ag2.metric("VLM-safe", vlm_safe_total)
                ag3.metric("Officer agrees with VLM", vlm_agree_count)
                ag4.metric("Officer disagrees with VLM", vlm_disagree_count)
                if vlm_agree_count + vlm_disagree_count > 0:
                    agree_rate = vlm_agree_count / (vlm_agree_count + vlm_disagree_count)
                    st.metric("VLM Safety Accuracy (per officer)",
                              f"{100*agree_rate:.0f}%")

                if disagree_details:
                    st.subheader(f"VLM Safety Disagreements ({len(disagree_details)})")
                    st.dataframe(disagree_details, use_container_width=True)

                if safety_descriptions:
                    st.subheader(f"Safety Concerns Flagged ({len(safety_descriptions)})")
                    st.dataframe(safety_descriptions, use_container_width=True)
            else:
                st.info("No safety reviews completed yet.")

        # Discard tracking
        st.subheader("Discarded Clips")
        discard_stats = Counter()
        discarded_clips = []
        if all_annotations:
            for aid, anns in all_annotations.items():
                for ann in anns.values():
                    if ann.get("status") == "discarded":
                        reason = ann.get("discard_reason", "unknown")
                        discard_stats[reason] += 1
                        discarded_clips.append({
                            "clip_id": ann.get("clip_id", ""),
                            "annotator": aid,
                            "reason": reason,
                        })

        if discard_stats:
            d1, d2 = st.columns(2)
            d1.metric("Total Discarded", sum(discard_stats.values()))
            d2.metric("Discard Rate",
                      f"{100*sum(discard_stats.values())/max(len(tier_a),1):.1f}%")
            st.dataframe(
                [{"Reason": k, "Count": v} for k, v in discard_stats.most_common()],
                use_container_width=True,
            )
        else:
            st.info("No clips discarded yet.")

        # Annotator role breakdown
        st.subheader("Annotator Roles")
        if assignments:
            role_rows = []
            for aid, data in assignments.get("assignments", {}).items():
                if aid in INACTIVE_ANNOTATORS:
                    continue
                role_rows.append({
                    "Annotator": aid,
                    "Role": data.get("role", "?"),
                    "Assigned Clips": data.get("total_clips", len(data.get("clips", []))),
                })
            if role_rows:
                st.dataframe(role_rows, use_container_width=True)


    # ==================== TAB 8: OPEN VOCABULARY (X1) ====================
    with tab8:
        st.header("Open Vocabulary (X1) Tracking")

        # --- VLM X1 usage ---
        st.subheader("VLM X1 Usage")
        if vlm_results:
            vlm_x1_clips = 0
            vlm_x1_workers = 0
            x1_descriptions = []
            x1_closest_classes = Counter()
            total_vlm_clips = len(vlm_results)
            vlm_failures = 0

            for record in vlm_results:
                clip_id = record.get("clip_id", "")
                vlm_data = record.get("vlm", {})
                if not vlm_data.get("success", False):
                    vlm_failures += 1
                    continue
                normalized = vlm_data.get("normalized", {})
                persons = normalized.get("persons", [])
                clip_has_x1 = False
                for person in persons:
                    if person.get("action_code") == "X1":
                        clip_has_x1 = True
                        vlm_x1_workers += 1
                        unlisted = person.get("unlisted_action", {})
                        desc = unlisted.get("free_text_description", "")
                        closest = unlisted.get("closest_existing_class", "")
                        if desc:
                            x1_descriptions.append({
                                "clip_id": clip_id,
                                "description": desc,
                                "closest_class": closest,
                            })
                        if closest:
                            x1_closest_classes[closest] += 1
                if clip_has_x1:
                    vlm_x1_clips += 1

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total VLM Clips", total_vlm_clips)
            m2.metric("X1 Clips", vlm_x1_clips)
            m3.metric("X1 Workers", vlm_x1_workers)
            coverage = round((1 - vlm_x1_clips / max(total_vlm_clips, 1)) * 100, 1)
            m4.metric("Taxonomy Coverage", f"{coverage}%")

            # Parse failure rates
            st.subheader("Parse Failure Rates")
            st.metric("VLM Failures",
                      f"{vlm_failures} ({100*vlm_failures/max(total_vlm_clips,1):.1f}%)")

            # X1 descriptions table
            if x1_descriptions:
                st.subheader(f"X1 Descriptions ({len(x1_descriptions)} entries)")
                st.dataframe(x1_descriptions, width="stretch")

            # Closest class distribution
            if x1_closest_classes:
                st.subheader("X1 Closest Existing Classes")
                closest_rows = [{"Class": k, "Count": v}
                                for k, v in x1_closest_classes.most_common(20)]
                st.dataframe(closest_rows, width="stretch")

        else:
            st.info("No VLM results available yet. Run vlm_annotator.py first.")

        # --- Human X1 usage ---
        st.subheader("Human Annotation X1 Usage")
        if all_annotations:
            human_x1_clips = 0
            human_x1_workers = 0
            human_total_clips = 0
            vlm_x1_human_agreed = 0
            vlm_x1_human_overrode = 0
            human_initiated_x1 = 0

            for annotator_id, clips in all_annotations.items():
                for clip_id, ann in clips.items():
                    human_total_clips += 1
                    persons = ann.get("persons", [])
                    clip_has_x1 = False
                    for person in persons:
                        action = person.get("action_code", "")
                        if action == "X1":
                            clip_has_x1 = True
                            human_x1_workers += 1
                            # Check if VLM also suggested X1
                            prefilled = person.get("prefilled_from", "")
                            fields_modified = person.get("fields_modified", [])
                            if prefilled and "action_code" not in fields_modified:
                                vlm_x1_human_agreed += 1
                            elif prefilled and "action_code" in fields_modified:
                                human_initiated_x1 += 1
                        else:
                            # Check if VLM suggested X1 but human overrode
                            prefilled = person.get("prefilled_from", "")
                            if prefilled:
                                fields_modified = person.get("fields_modified", [])
                                if "action_code" in fields_modified:
                                    # Could be VLM X1 -> human standard
                                    pass  # Would need VLM data to confirm
                    if clip_has_x1:
                        human_x1_clips += 1

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Human-Annotated Clips", human_total_clips)
            h2.metric("Clips with X1", human_x1_clips)
            h3.metric("X1 Workers", human_x1_workers)
            if human_total_clips > 0:
                cov = round((1 - human_x1_clips / human_total_clips) * 100, 1)
                h4.metric("Taxonomy Coverage", f"{cov}%")
        else:
            st.info("No human annotations available yet.")

    # ==================== TAB 9: SCHEMA HEALTH ====================
    with tab9:
        st.header("Schema Health")
        st.caption(
            "Real-time validator results across all annotation files on disk. "
            "Re-runs the schema validator on every load. Any record showing as "
            "INVALID here would be blocked from save by the new app.py form."
        )
        from annotation_tool.schema_validator import (
            categorize_record, validate_annotation,
        )

        # Walk the annotations directory directly (not via load_all_annotations,
        # which caches and may filter)
        ann_root = Path(os.environ.get(
            "STEELBENCH_ANNOTATIONS_DIR",
            os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"),
        ))
        per_annotator_counts = {}
        invalid_records = []
        category_counts = {
            "valid": 0,
            "invalid_recoverable": 0,
            "invalid_requeue": 0,
            "non_submitted": 0,
        }
        if ann_root.exists():
            for sub in sorted(ann_root.iterdir()):
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                ann_id = sub.name
                per_annotator_counts.setdefault(
                    ann_id,
                    {"valid": 0, "invalid_recoverable": 0,
                     "invalid_requeue": 0, "non_submitted": 0},
                )
                for fp in sub.iterdir():
                    if not fp.name.endswith(".json"):
                        continue
                    try:
                        with open(fp) as f:
                            rec = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue
                    cat = categorize_record(rec)
                    category_counts[cat] += 1
                    per_annotator_counts[ann_id][cat] += 1
                    if cat == "invalid_requeue":
                        invalid_records.append({
                            "annotator": ann_id,
                            "clip_id": rec.get("clip_id", fp.name),
                            "errors": "; ".join(validate_annotation(rec)[:3]),
                        })

            total_submitted = (
                category_counts["valid"]
                + category_counts["invalid_recoverable"]
                + category_counts["invalid_requeue"]
            )
            cols = st.columns(4)
            cols[0].metric("Total submitted", total_submitted)
            cols[1].metric("Valid", category_counts["valid"])
            cols[2].metric(
                "Bonus per-person (legacy)",
                category_counts["invalid_recoverable"],
            )
            cols[3].metric(
                "Invalid (re-queue)",
                category_counts["invalid_requeue"],
                delta=("0 — clean!" if category_counts["invalid_requeue"] == 0
                       else f"-{category_counts['invalid_requeue']}"),
                delta_color="inverse",
            )

            st.subheader("Per-annotator breakdown")
            ann_rows = []
            for ann, counts in sorted(per_annotator_counts.items()):
                total = sum(counts.values())
                ann_rows.append({
                    "annotator": ann,
                    "total": total,
                    "valid": counts["valid"],
                    "bonus": counts["invalid_recoverable"],
                    "requeue": counts["invalid_requeue"],
                    "non_submitted": counts["non_submitted"],
                })
            if ann_rows:
                st.dataframe(ann_rows, width="stretch")

            if invalid_records:
                st.subheader(f"Invalid records ({len(invalid_records)})")
                st.dataframe(invalid_records, width="stretch")
            else:
                st.success(
                    "All submitted records pass strict validation. "
                    "Schema invariants are intact."
                )
        else:
            st.info(f"Annotations directory not found: {ann_root}")

    # ==================== TAB 10: FLAG TRACKER ====================
    with tab10:
        st.header("Flag Tracker")
        st.caption(
            "Breakdown of clips flagged for tier_2 expert review. Each flag "
            "carries a category and a free-text reason. Resolution status is "
            "tracked by checking whether the clip has a corresponding tier_2 "
            "expert annotation."
        )

        # Load tier2_queue.json
        queue_path = Path(os.environ.get(
            "STEELBENCH_ASSIGNMENTS_DIR",
            os.path.join(PROJECT_ROOT, "annotation_tool", "data", "assignments"),
        )) / "tier2_queue.json"
        queue = []
        if queue_path.exists():
            try:
                with open(queue_path) as f:
                    queue = json.load(f)
            except (OSError, json.JSONDecodeError):
                queue = []

        # Walk all annotation files for flagged tier_1 records and tier_2
        # resolutions
        ann_root = Path(os.environ.get(
            "STEELBENCH_ANNOTATIONS_DIR",
            os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"),
        ))
        flagged_records = []
        tier2_resolutions = set()
        if ann_root.exists():
            for sub in sorted(ann_root.iterdir()):
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                ann_id = sub.name
                for fp in sub.iterdir():
                    if not fp.name.endswith(".json"):
                        continue
                    try:
                        with open(fp) as f:
                            rec = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue
                    role = rec.get("annotator_role", "")
                    status = rec.get("status", "")
                    cid = rec.get("clip_id", "")
                    if role == "tier_1" and status == "flagged":
                        flagged_records.append({
                            "annotator": ann_id,
                            "clip_id": cid,
                            "category": rec.get("flag_category", ""),
                            "reason": (rec.get("flag_reason", "") or "")[:100],
                            "timestamp": rec.get("annotator_timestamp", "")[:19],
                        })
                    if role == "tier_2" and status == "submitted":
                        tier2_resolutions.add(cid)

        # Per-category breakdown
        from collections import Counter as _C
        cat_counts = _C(r["category"] for r in flagged_records if r["category"])
        cols = st.columns(3)
        cols[0].metric("Total flagged", len(flagged_records))
        cols[1].metric("Tier 2 queue (live)", len(queue))
        n_resolved = sum(
            1 for r in flagged_records if r["clip_id"] in tier2_resolutions
        )
        cols[2].metric("Resolved by expert", n_resolved)

        if cat_counts:
            st.subheader("Flag categories")
            cat_rows = [
                {"category": c, "count": n}
                for c, n in cat_counts.most_common()
            ]
            st.dataframe(cat_rows, width="stretch")

        if flagged_records:
            st.subheader("Flagged clips")
            for r in flagged_records:
                r["resolved"] = (
                    "yes" if r["clip_id"] in tier2_resolutions else "pending"
                )
            st.dataframe(flagged_records, width="stretch")
        else:
            st.info(
                "No clips with status='flagged' yet. Annotators flag clips "
                "they cannot annotate normally (e.g. wrong VLM count)."
            )

    # ==================== TAB 11: EXPERT EDIT AUDIT ====================
    with tab11:
        st.header("Expert Edit Audit")
        st.caption(
            "For each tier_2 expert review, shows which fields the expert "
            "modified relative to the tier_1 annotation baseline. Uses the "
            "new edit_tracking.source_type field to distinguish 'expert "
            "edited tier_1' from 'tier_1 edited VLM'."
        )

        ann_root = Path(os.environ.get(
            "STEELBENCH_ANNOTATIONS_DIR",
            os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"),
        ))
        expert_reviews = []
        from collections import Counter as _C2
        field_counter = _C2()
        per_tier1_counter = _C2()  # tier_1 annotator -> times their work was modified
        if ann_root.exists():
            for sub in sorted(ann_root.iterdir()):
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                expert_id = sub.name
                for fp in sub.iterdir():
                    if not fp.name.endswith(".json"):
                        continue
                    try:
                        with open(fp) as f:
                            rec = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        continue
                    if rec.get("annotator_role") != "tier_2":
                        continue
                    if rec.get("status") != "submitted":
                        continue
                    et = rec.get("edit_tracking") or {}
                    src_type = et.get("source_type") or et.get("prefilled_from")
                    if src_type != "tier_1":
                        continue  # only count expert edits made vs tier_1 baseline
                    fields = et.get("fields_modified") or []
                    expert_reviews.append({
                        "expert": expert_id,
                        "clip_id": rec.get("clip_id", fp.name),
                        "n_fields_modified": len(fields),
                        "modified": ", ".join(fields[:6])
                        + ("..." if len(fields) > 6 else ""),
                    })
                    for f_name in fields:
                        field_counter[f_name] += 1

                    # Find the tier_1 annotator who originally submitted this clip
                    cid = rec.get("clip_id", "")
                    if cid:
                        for sub2 in ann_root.iterdir():
                            if not sub2.is_dir() or sub2.name.startswith("."):
                                continue
                            if sub2.name == expert_id:
                                continue
                            t1_path = sub2 / f"{cid}.json"
                            if not t1_path.exists():
                                continue
                            try:
                                with open(t1_path) as f:
                                    t1 = json.load(f)
                            except (OSError, json.JSONDecodeError):
                                continue
                            if (t1.get("annotator_role") == "tier_1"
                                    and t1.get("status") == "submitted"
                                    and len(fields) > 0):
                                per_tier1_counter[sub2.name] += 1
                                break

        cols = st.columns(3)
        cols[0].metric("Expert reviews (vs tier_1)", len(expert_reviews))
        if expert_reviews:
            avg_fields = sum(r["n_fields_modified"] for r in expert_reviews) / len(expert_reviews)
            cols[1].metric("Avg fields modified", f"{avg_fields:.1f}")
        cols[2].metric("Total field-edits", sum(field_counter.values()))

        if field_counter:
            st.subheader("Most-modified fields (top 15)")
            field_rows = [
                {"field": f, "count": n}
                for f, n in field_counter.most_common(15)
            ]
            st.dataframe(field_rows, width="stretch")

        if per_tier1_counter:
            st.subheader("Tier 1 annotators with most expert overrides")
            t1_rows = [
                {"tier_1_annotator": a, "clips_modified_by_expert": n}
                for a, n in per_tier1_counter.most_common()
            ]
            st.dataframe(t1_rows, width="stretch")

        if expert_reviews:
            st.subheader(f"All expert reviews ({len(expert_reviews)})")
            st.dataframe(expert_reviews, width="stretch")
        else:
            st.info(
                "No expert reviews with edit_tracking.source_type='tier_1' yet. "
                "This tab populates as experts review and modify tier_1 submissions."
            )

    # ==================== TAB 12: PHASE 2 PROGRESS ====================
    with tab12:
        st.header("Phase 2 Progress — fresh-batch annotations & per-annotator IAA")

        p2_clips = load_phase2_clip_set()
        if not p2_clips:
            st.warning(
                "Could not identify Phase 2 clip set. Expected one of: "
                "`active_batch/config/batch_manifest_phase2_3000.csv` or "
                "`batch_manifest_pilot_gt.csv` (fallback)."
            )
        else:
            st.caption(f"Phase 2 universe: **{len(p2_clips)} clips** "
                       "(loaded from batch_manifest_phase2_3000.csv)")

            # ---- Per-annotator Phase 2 progress ----
            # compute_all_iaa already imported at module level (line 36, from qc_metrics)
            _iaa = compute_all_iaa

            assignments_path = os.path.join(ASSIGNMENTS_DIR, "assignments.json")
            assigned_by_aid = {}
            if os.path.exists(assignments_path):
                with open(assignments_path) as f:
                    d = json.load(f)
                    _a = d.get("assignments", d)
                    for aid, info in _a.items():
                        if aid in INACTIVE_ANNOTATORS:
                            continue
                        p2_assigned = sum(
                            1 for c in info.get("clips", []) or []
                            if c.get("clip_id") in p2_clips
                        )
                        assigned_by_aid[aid] = p2_assigned

            # Collect submitted Phase 2 annotations per annotator
            p2_records_by_aid = defaultdict(dict)  # aid -> cid -> annotation
            for aid, anns in all_annotations.items():
                for cid, ann in anns.items():
                    if cid in p2_clips and ann.get("status") == "submitted":
                        p2_records_by_aid[aid][cid] = ann

            # Per-annotator totals
            summary_rows = []
            for aid in sorted(set(list(assigned_by_aid.keys()) + list(all_annotations.keys()))):
                all_p2 = [
                    ann for cid, ann in all_annotations.get(aid, {}).items()
                    if cid in p2_clips
                ]
                sub = sum(1 for a in all_p2 if a.get("status") == "submitted")
                flag = sum(1 for a in all_p2 if a.get("status") == "flagged")
                disc = sum(1 for a in all_p2 if a.get("status") == "discarded")
                assigned = assigned_by_aid.get(aid, 0)
                pct = (100 * sub / assigned) if assigned else 0.0
                summary_rows.append({
                    "annotator": aid,
                    "P2 assigned": assigned,
                    "P2 submitted": sub,
                    "P2 flagged": flag,
                    "P2 discarded": disc,
                    "% complete": round(pct, 1),
                })

            st.subheader("Per-annotator Phase 2 submissions")
            if summary_rows:
                st.dataframe(summary_rows, width="stretch")
                t_sub = sum(r["P2 submitted"] for r in summary_rows)
                t_flag = sum(r["P2 flagged"] for r in summary_rows)
                t_assigned = sum(r["P2 assigned"] for r in summary_rows)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total P2 submitted", t_sub)
                c2.metric("Total P2 flagged", t_flag)
                c3.metric("Overall P2 completion",
                          f"{100*t_sub/max(t_assigned,1):.1f}%")
            else:
                st.info("No Phase 2 records yet.")

            st.divider()

            # ---- Pairwise IAA (Cohen's κ) across Phase 2 double-annotated clips ----
            st.subheader("Pairwise IAA — Phase 2 double-annotated clips")
            # Build {clip_id: {aid: ann}} for P2 submissions only
            p2_cid_to_aid_ann = defaultdict(dict)
            for aid, cid_map in p2_records_by_aid.items():
                for cid, ann in cid_map.items():
                    p2_cid_to_aid_ann[cid][aid] = ann

            # Collect pairs
            pair_clips = defaultdict(lambda: {"a": [], "b": []})  # (aid_a,aid_b) sorted
            for cid, aid_map in p2_cid_to_aid_ann.items():
                if len(aid_map) < 2:
                    continue
                aids = sorted(aid_map.keys())
                for i in range(len(aids)):
                    for j in range(i + 1, len(aids)):
                        a_id, b_id = aids[i], aids[j]
                        pair_clips[(a_id, b_id)]["a"].append(aid_map[a_id])
                        pair_clips[(a_id, b_id)]["b"].append(aid_map[b_id])

            if not pair_clips:
                st.info(
                    "No Phase 2 clips with 2+ submitted annotations yet. "
                    "IAA populates once the same clip is annotated by at least "
                    "two annotators (15% double-annotation ratio is configured "
                    "in experiments/phase2_3000.yaml)."
                )
            else:
                MIN_PAIR_N = 5  # lower threshold so early signal shows; statistically noisy
                pair_rows = []
                for (a_id, b_id), lists in sorted(pair_clips.items()):
                    n = len(lists["a"])
                    row = {
                        "annotator_A": a_id,
                        "annotator_B": b_id,
                        "shared_clips": n,
                    }
                    if n >= MIN_PAIR_N:
                        try:
                            iaa = _iaa(lists["a"], lists["b"])
                            row.update({
                                "action_κ": round(iaa["a1_action_kappa"], 3),
                                "spatial_κ": round(iaa["a2_spatial_kappa"], 3),
                                "ppe_κ": round(iaa["a3_ppe_kappa"], 3),
                                "action pass (≥0.75)": iaa["a1_meets_target"],
                                "spatial pass (≥0.70)": iaa["a2_meets_target"],
                                "ppe pass (≥0.65)": iaa["a3_meets_target"],
                            })
                        except Exception as e:
                            row["error"] = str(e)[:80]
                    else:
                        row["note"] = f"need ≥{MIN_PAIR_N} shared (have {n})"
                    pair_rows.append(row)
                st.dataframe(pair_rows, width="stretch")

            st.divider()

            # ---- Per-annotator IAA vs rest (each annotator's average κ against everyone else on shared P2 clips) ----
            st.subheader("Per-annotator average κ (across all their Phase 2 shared clips)")
            per_aid_rollup = defaultdict(lambda: {"a": [], "b": []})
            for cid, aid_map in p2_cid_to_aid_ann.items():
                if len(aid_map) < 2:
                    continue
                aids = sorted(aid_map.keys())
                for focal in aids:
                    for other in aids:
                        if focal == other:
                            continue
                        per_aid_rollup[focal]["a"].append(aid_map[focal])
                        per_aid_rollup[focal]["b"].append(aid_map[other])

            if per_aid_rollup:
                rollup_rows = []
                for aid in sorted(per_aid_rollup.keys()):
                    lists = per_aid_rollup[aid]
                    n = len(lists["a"])
                    if n < MIN_PAIR_N:
                        rollup_rows.append({
                            "annotator": aid,
                            "shared comparisons": n,
                            "note": f"need ≥{MIN_PAIR_N}",
                        })
                        continue
                    try:
                        iaa = _iaa(lists["a"], lists["b"])
                        rollup_rows.append({
                            "annotator": aid,
                            "shared comparisons": n,
                            "action_κ": round(iaa["a1_action_kappa"], 3),
                            "spatial_κ": round(iaa["a2_spatial_kappa"], 3),
                            "ppe_κ": round(iaa["a3_ppe_kappa"], 3),
                        })
                    except Exception as e:
                        rollup_rows.append({
                            "annotator": aid,
                            "shared comparisons": n,
                            "error": str(e)[:80],
                        })
                st.dataframe(rollup_rows, width="stretch")
            else:
                st.info("No overlapping Phase 2 submissions yet to compute per-annotator κ.")

            st.caption(
                "Targets: Action κ ≥ 0.75 • Spatial κ ≥ 0.70 • PPE κ ≥ 0.65. "
                "Phase 2 uses `double_annotate_ratio=0.15` (≈406 clips double-annotated)."
            )


if __name__ == "__main__":
    main()
