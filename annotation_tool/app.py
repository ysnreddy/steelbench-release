"""Streamlit annotation tool for SteelBench human verification.

Role-aware UI with tier_1 (annotation), tier_2 (adjudication),
and tier_3 (safety review) workflows. Supports scene type classification,
dual-layer taxonomy (X1 unlisted), coordination, and action transitions.

Usage:
    streamlit run annotation_tool/app.py
"""

import csv
import glob
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.agreement import (
    ACTION_TAXONOMY, ALL_ACTION_LABELS, SPATIAL_TAGS, PPE_ITEMS,
    PPE_VALUES, PPE_VALUES_EXTENDED, PPE_ITEMS_WITH_NA,
    INTERACTION_TYPES, VISIBILITY_LEVELS, VISIBILITY_CONDITIONS,
    OCCLUSION_LEVELS, OCCLUSION_SOURCES, GROUP_FLAGS,
    SCENE_TYPES, SCENE_TYPE_LABELS, ANNOTATOR_ROLES,
)
from annotation_tool.schema_validator import (
    derive_annotation_layer,
    num_workers_bounds_for_layer,
    validate_annotation,
    FLAG_CATEGORIES,
)

# ---------- Configuration ----------

OUTPUT_DIR = os.environ.get(
    "STEELBENCH_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
ANNOTATIONS_DIR = os.environ.get(
    "STEELBENCH_ANNOTATIONS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"))
# Manifest: try tier_a_manifest first, fallback to active_batch manifest (for VPS)
_tier_a_path = os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv")
_batch_path = os.path.join(PROJECT_ROOT, "active_batch", "config", "batch_manifest.csv")
TIER_A_MANIFEST = os.environ.get(
    "STEELBENCH_MANIFEST",
    _tier_a_path if os.path.exists(_tier_a_path) else _batch_path)
ASSIGNMENTS_DIR = os.environ.get(
    "STEELBENCH_ASSIGNMENTS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "assignments"))
VLM_RESULTS_DIR = os.environ.get(
    "STEELBENCH_VLM_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results"))

ACTION_DROPDOWN = [""] + [f"[{code}] {name}" for code, name in ACTION_TAXONOMY.items()]


# ---------- Data Loading ----------

@st.cache_data
def load_clips():
    clips = []
    if not os.path.exists(TIER_A_MANIFEST):
        return clips
    with open(TIER_A_MANIFEST) as f:
        for row in csv.DictReader(f):
            clips.append(row)
    return clips


@st.cache_data(ttl=30)
def load_assignments():
    assign_path = os.path.join(ASSIGNMENTS_DIR, "assignments.json")
    if not os.path.exists(assign_path):
        return None
    with open(assign_path) as f:
        return json.load(f)


@st.cache_data(ttl=5)
def load_existing_annotations(annotator_id):
    ann_dir = os.path.join(ANNOTATIONS_DIR, annotator_id)
    annotations = {}
    if os.path.exists(ann_dir):
        for fname in os.listdir(ann_dir):
            if fname.endswith(".json"):
                with open(os.path.join(ann_dir, fname)) as f:
                    ann = json.load(f)
                    annotations[ann["clip_id"]] = ann
    return annotations


@st.cache_data(ttl=30)
def load_vlm_suggestions():
    """Load VLM results for pre-filling annotation forms."""
    results_path = os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    suggestions = {}
    if not os.path.exists(results_path):
        return suggestions
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                cid = r.get("clip_id", "")
                vlm = r.get("vlm") or {}
                norm = vlm.get("normalized") if vlm.get("success") else None

                if norm:
                    suggestions[cid] = norm
                    suggestions[cid]["_agreement"] = r.get("agreement", {})
            except json.JSONDecodeError:
                continue
    return suggestions


@st.cache_data(ttl=60)
def load_calibration_set():
    """Load anchored/blind calibration set for Phase 2 anchoring bias experiment."""
    cal_path = os.path.join(PROJECT_ROOT, "active_batch", "data", "calibration_set.json")
    if not os.path.exists(cal_path):
        return {"anchored_clips": {}, "blind_clips": {}}
    try:
        with open(cal_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"anchored_clips": {}, "blind_clips": {}}


def get_calibration_condition(clip_id, cal_set):
    """Returns 'anchored', 'blind', or None for the given clip."""
    if clip_id in cal_set.get("anchored_clips", {}):
        return "anchored"
    if clip_id in cal_set.get("blind_clips", {}):
        return "blind"
    return None


def compute_edit_tracking(annotation, prefill_source, source_type="vlm"):
    """Compare a human annotation against its prefill source. Tracks which
    fields the human modified relative to the prefill.

    Args:
        annotation: the dict that the human just submitted
        prefill_source: the dict the form was pre-filled FROM (VLM normalized
            for tier_1 fresh, latest tier_1 annotation for tier_2 review)
        source_type: "vlm" or "tier_1" — used to label the audit trail so
            downstream consumers can distinguish "human edited the VLM" from
            "expert edited the annotator's submission"

    Returns dict with `prefilled_from`, `source_type`, `fields_modified`, and `modification_count`.
    """
    if not prefill_source:
        return {
            "prefilled_from": None,
            "source_type": None,
            "fields_modified": [],
            "modification_count": 0,
        }
    # Backward-compat alias used throughout the function body
    vlm_suggestion = prefill_source

    modified = []

    # Scene-level fields
    vlm_scene = vlm_suggestion.get("scene_type", "")
    if vlm_scene and annotation.get("scene_type") != vlm_scene:
        modified.append("scene_type")

    vlm_vis = vlm_suggestion.get("visibility", "")
    if vlm_vis and annotation.get("visibility") != vlm_vis:
        modified.append("visibility")

    vlm_nw = vlm_suggestion.get("num_workers")
    if vlm_nw is not None and annotation.get("num_workers") != vlm_nw:
        modified.append("num_workers")

    vlm_vc = set(vlm_suggestion.get("visibility_conditions", []))
    ann_vc = set(annotation.get("visibility_conditions", []))
    if vlm_vc and ann_vc != vlm_vc:
        modified.append("visibility_conditions")

    # Per-person fields
    ann_persons = annotation.get("persons", [])
    vlm_persons = vlm_suggestion.get("persons", [])

    for p_idx in range(min(len(ann_persons), len(vlm_persons))):
        ap = ann_persons[p_idx]
        vp = vlm_persons[p_idx]
        prefix = f"persons[{p_idx}]"

        if ap.get("action_code") != vp.get("action_code", ""):
            modified.append(f"{prefix}.action_code")

        # Spatial context (compare as sets)
        ann_spatial = set(ap.get("spatial_context", []))
        vlm_spatial = set(vp.get("spatial_context", []))
        if ann_spatial != vlm_spatial:
            modified.append(f"{prefix}.spatial_context")

        # PPE items
        ann_ppe = ap.get("ppe", {})
        vlm_ppe = vp.get("ppe", {})
        for item in set(list(ann_ppe.keys()) + list(vlm_ppe.keys())):
            if ann_ppe.get(item) != vlm_ppe.get(item, ""):
                modified.append(f"{prefix}.ppe.{item}")

        if ap.get("occlusion_level") != vp.get("occlusion_level", "fully_visible"):
            modified.append(f"{prefix}.occlusion_level")

        ann_occ_src = set(ap.get("occlusion_source", []))
        vlm_occ_src = set(vp.get("occlusion_source", []))
        if ann_occ_src != vlm_occ_src:
            modified.append(f"{prefix}.occlusion_source")

        if ap.get("unsafe_act", "") != vp.get("unsafe_act", ""):
            modified.append(f"{prefix}.unsafe_act")

        if ap.get("interaction") != vp.get("interaction", ""):
            modified.append(f"{prefix}.interaction")

        if ap.get("group_flag") != vp.get("group_flag", ""):
            modified.append(f"{prefix}.group_flag")

    # Extra persons added or removed
    if len(ann_persons) != len(vlm_persons):
        modified.append("num_persons_changed")

    return {
        "prefilled_from": source_type,  # "vlm" or "tier_1"
        "source_type": source_type,
        "fields_modified": modified,
        "modification_count": len(modified),
    }


def save_annotation(annotator_id, annotation):
    ann_dir = os.path.join(ANNOTATIONS_DIR, annotator_id)
    os.makedirs(ann_dir, exist_ok=True)
    path = os.path.join(ann_dir, f"{annotation['clip_id']}.json")
    with open(path, "w") as f:
        json.dump(annotation, f, indent=2)


def add_to_tier2_queue(clip_id, reason):
    """Add a clip to the tier_2 review queue (stored as JSON file)."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier2_queue.json")
    queue = []
    if os.path.exists(queue_path):
        with open(queue_path) as f:
            queue = _normalize_queue(json.load(f))
    if not any(q.get("clip_id") == clip_id for q in queue):
        queue.append({
            "clip_id": clip_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)


def load_tier2_queue():
    """Load the tier_2 review queue."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier2_queue.json")
    if not os.path.exists(queue_path):
        return []
    with open(queue_path) as f:
        return _normalize_queue(json.load(f))


def remove_from_tier2_queue(clip_id):
    """Remove a clip from the tier_2 review queue (e.g., after expert submission)."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier2_queue.json")
    if not os.path.exists(queue_path):
        return
    with open(queue_path) as f:
        queue = _normalize_queue(json.load(f))
    new_queue = [q for q in queue if q.get("clip_id") != clip_id]
    if len(new_queue) != len(queue):
        with open(queue_path, "w") as f:
            json.dump(new_queue, f, indent=2)


def get_tier2_queue_reason(clip_id):
    """Return reason string if clip is in tier_2 queue, else None."""
    queue = load_tier2_queue()
    for q in queue:
        if q.get("clip_id") == clip_id:
            return q.get("reason", "")
    return None


# ---------- Tier 3 (safety officer) dynamic queue ----------
# The tier_3 queue grows at runtime when tier_2 experts confirm violations.
# It merges with the static safety_officer assignment (zone-flagged + VLM-sampled)
# for the UI to show a unified audit queue.


def _normalize_queue(raw):
    """Normalize a queue that may be a list of strings, dicts, or a {"queue": [...]} wrapper."""
    if isinstance(raw, dict):
        raw = raw.get("queue", [])
    if not isinstance(raw, list):
        return []
    normalized = []
    for item in raw:
        if isinstance(item, str):
            normalized.append({"clip_id": item, "reason": "", "timestamp": ""})
        elif isinstance(item, dict):
            normalized.append(item)
    return normalized


def add_to_tier3_queue(clip_id, reason):
    """Add a clip to the tier_3 safety-officer review queue (dynamic additions)."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier3_queue.json")
    queue = []
    if os.path.exists(queue_path):
        with open(queue_path) as f:
            queue = _normalize_queue(json.load(f))
    if not any(q.get("clip_id") == clip_id for q in queue):
        queue.append({
            "clip_id": clip_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)


def load_tier3_queue():
    """Load the tier_3 safety-officer review queue."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier3_queue.json")
    if not os.path.exists(queue_path):
        return []
    with open(queue_path) as f:
        return _normalize_queue(json.load(f))


def remove_from_tier3_queue(clip_id):
    """Remove a clip from the tier_3 queue (after safety officer submission)."""
    queue_path = os.path.join(ASSIGNMENTS_DIR, "tier3_queue.json")
    if not os.path.exists(queue_path):
        return
    with open(queue_path) as f:
        queue = _normalize_queue(json.load(f))
    new_queue = [q for q in queue if q.get("clip_id") != clip_id]
    if len(new_queue) != len(queue):
        with open(queue_path, "w") as f:
            json.dump(new_queue, f, indent=2)


def get_tier3_queue_reason(clip_id):
    """Return reason string if clip is in tier_3 queue, else None."""
    for q in load_tier3_queue():
        if q.get("clip_id") == clip_id:
            return q.get("reason", "")
    return None


# ---------- Prefill dispatcher (D5: single source of truth) ----------


def find_latest_tier1(clip_id, annotations_dir=None):
    """Find the most recent tier_1 annotation file for a given clip across
    all annotator dirs. Returns the dict or None.

    Used by tier_2 expert review to load the annotator's submission verbatim.
    """
    annotations_dir = annotations_dir or ANNOTATIONS_DIR
    if not os.path.exists(annotations_dir):
        return None
    latest = None
    latest_ts = ""
    for ann_dir_name in os.listdir(annotations_dir):
        if ann_dir_name.startswith("."):
            continue
        ann_path = os.path.join(annotations_dir, ann_dir_name, f"{clip_id}.json")
        if not os.path.exists(ann_path):
            continue
        try:
            with open(ann_path) as f:
                candidate = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if candidate.get("annotator_role") != "tier_1":
            continue
        if candidate.get("status") != "submitted":
            continue
        ts = candidate.get("annotator_timestamp", "")
        if ts >= latest_ts:
            latest_ts = ts
            latest = candidate
    return latest


def get_prefill(annotator_role, clip_id, vlm_normalized, annotations_dir=None):
    """Single source of truth for form prefill.

    Returns (prefill_dict, source_label, vlm_reference) where:
    - prefill_dict: the dict the form widgets should read from. For tier_2
      reviewing a clip with an existing tier_1 annotation, this is the
      tier_1 record VERBATIM (no merging with VLM). For tier_1 fresh
      annotation (or tier_2 with no prior tier_1), this is the VLM record.
    - source_label: human-readable label like "tier_1: annotator_2" or "VLM"
    - vlm_reference: the VLM dict (set only when prefill source is tier_1,
      so the expert can compare against the VLM in a side panel)

    The form widgets read EXCLUSIVELY from prefill_dict — no fallbacks, no
    merging. This eliminates the schema-vs-source contradictions that
    plagued the previous merge logic.
    """
    if annotator_role == "tier_2":
        latest_t1 = find_latest_tier1(clip_id, annotations_dir)
        if latest_t1:
            return (
                latest_t1,
                f"tier_1: {latest_t1.get('annotator_id', '?')}",
                vlm_normalized,
            )
    return vlm_normalized, "VLM", None


def _prefill_signature(prefill):
    """Stable hash of a prefill dict, used to scope the session_state primer.

    When the prefill changes (e.g., a new tier_1 annotation arrives), the
    signature changes and the primer re-runs once for the new version.
    """
    if not prefill:
        return "none"
    try:
        s = json.dumps(prefill, sort_keys=True, default=str)
    except (TypeError, ValueError):
        s = str(prefill)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def prime_session_state(clip_id, prefill):
    """One-shot primer: write prefill values directly into st.session_state
    BEFORE any per-clip widget is created.

    Streamlit's `value=` widget parameter is ignored when the widget's key
    already exists in session_state (which can happen across reruns or
    deploys). This primer defeats that by writing the defaults directly,
    gated by `f"primed::{clip_id}::{prefill_sig}"` so it runs once per clip
    per prefill version.
    """
    if not prefill:
        return
    sig = _prefill_signature(prefill)
    primer_key = f"primed::{clip_id}::{sig}"
    if st.session_state.get(primer_key):
        return  # already primed for this prefill version

    # Scene-level
    scene_type = prefill.get("scene_type", "") or ""
    if scene_type in SCENE_TYPES:
        st.session_state[f"scene_type_{clip_id}"] = scene_type
    else:
        st.session_state[f"scene_type_{clip_id}"] = ""

    nw = prefill.get("num_workers")
    if isinstance(nw, int) and nw >= 1:
        st.session_state[f"num_workers_{clip_id}"] = nw
    st.session_state[f"scene_desc_{clip_id}"] = prefill.get("scene_description", "") or ""
    st.session_state[f"visible_equipment_{clip_id}"] = prefill.get("visible_equipment", "") or ""
    vc = prefill.get("visibility_conditions", []) or []
    vc = [v for v in vc if v in VISIBILITY_CONDITIONS]
    st.session_state[f"visibility_conditions_{clip_id}"] = vc if vc else ["clear"]
    overall_ppe = prefill.get("overall_ppe_compliance", "")
    if overall_ppe in ("compliant", "partial", "non_compliant", "cannot_determine"):
        st.session_state[f"overall_ppe_{clip_id}"] = overall_ppe
    # dominant_actions widget uses display labels with [code] prefix
    da = prefill.get("dominant_actions", []) or []
    da_labels = [f"[{a}] {ACTION_TAXONOMY[a]}" for a in da if a in ACTION_TAXONOMY]
    if da_labels:
        st.session_state[f"dominant_actions_{clip_id}"] = da_labels
    # scene_unsafe_act (Layer 1 only, optional)
    st.session_state[f"scene_unsafe_act_{clip_id}"] = prefill.get("scene_unsafe_act", "") or ""

    # annotator confidence
    conf = prefill.get("annotator_confidence")
    if isinstance(conf, (int, float)):
        st.session_state[f"confidence_{clip_id}"] = float(conf)

    # Per-person fields (Layer 2)
    persons = prefill.get("persons") or []
    for p_idx, p in enumerate(persons):
        if not isinstance(p, dict):
            continue
        # Action class — needs the display label, not just the code
        ac = p.get("action_code", "")
        if ac in ACTION_TAXONOMY:
            st.session_state[f"action_{clip_id}_{p_idx}"] = f"[{ac}] {ACTION_TAXONOMY[ac]}"
        # Position
        pos = p.get("position", "")
        if pos in ("left", "center", "right", "foreground", "background"):
            st.session_state[f"pos_{clip_id}_{p_idx}"] = pos
        # Free-text fields (NEW: per-person physical_description, free_text_description)
        st.session_state[f"physical_desc_{clip_id}_{p_idx}"] = p.get("physical_description", "") or ""
        st.session_state[f"free_text_{clip_id}_{p_idx}"] = p.get("free_text_description", "") or ""
        # Spatial context
        sc = p.get("spatial_context", []) or []
        sc = [t for t in sc if t in SPATIAL_TAGS]
        st.session_state[f"spatial_{clip_id}_{p_idx}"] = sc if sc else ["ground_level"]
        # PPE
        ppe = p.get("ppe", {}) or {}
        for item in PPE_ITEMS:
            v = ppe.get(item, "")
            allowed = PPE_VALUES_EXTENDED if item in PPE_ITEMS_WITH_NA else PPE_VALUES
            if v in allowed:
                st.session_state[f"ppe_{clip_id}_{p_idx}_{item}"] = v
        # Occlusion
        ol = p.get("occlusion_level", "")
        if ol in OCCLUSION_LEVELS:
            st.session_state[f"occ_level_{clip_id}_{p_idx}"] = ol
        os_list = p.get("occlusion_source", []) or []
        os_list = [s for s in os_list if s in OCCLUSION_SOURCES]
        if os_list:
            st.session_state[f"occ_source_{clip_id}_{p_idx}"] = os_list
        # Unsafe act
        st.session_state[f"unsafe_{clip_id}_{p_idx}"] = p.get("unsafe_act", "") or ""
        # Interaction
        inter = p.get("interaction", "")
        if inter in INTERACTION_TYPES:
            st.session_state[f"interaction_{clip_id}_{p_idx}"] = inter
        # Group flag
        gf = p.get("group_flag", "")
        if gf in GROUP_FLAGS:
            st.session_state[f"group_{clip_id}_{p_idx}"] = gf
        # Per-person confidence (NEW)
        p_conf = p.get("confidence")
        if isinstance(p_conf, (int, float)):
            st.session_state[f"p_confidence_{clip_id}_{p_idx}"] = float(p_conf)

    st.session_state[primer_key] = True


# ---------- UI Components ----------

@st.cache_data(ttl=300)
def load_clip_detections(clip_id):
    """Load per-frame YOLO detection data for a clip."""
    candidates = [
        os.path.join(OUTPUT_DIR, "metadata", "clip_detections", f"{clip_id}.json"),
        os.path.join(PROJECT_ROOT, "active_batch", "detections", f"{clip_id}.json"),
    ]
    for det_path in candidates:
        if os.path.exists(det_path):
            with open(det_path) as f:
                return json.load(f)
    return None


@st.cache_data(ttl=300)
def get_thumbnail_with_boxes(frame_path, persons, width=480, quality=65):
    """Return JPEG bytes of a resized thumbnail with bounding boxes and P1/P2 labels."""
    try:
        from PIL import ImageDraw, ImageFont
        img = Image.open(frame_path)
        orig_w, orig_h = img.size
        ratio = width / orig_w
        new_height = int(orig_h * ratio)
        img = img.resize((width, new_height), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Colors for different persons
        colors = ["#FF4444", "#44FF44", "#4488FF", "#FFAA00", "#FF44FF",
                  "#44FFFF", "#FF8844", "#88FF44", "#4444FF", "#FFFF44"]

        for i, person in enumerate(persons):
            bbox = person.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Scale bbox to thumbnail size
            x1 = int(x1 * ratio)
            y1 = int(y1 * ratio)
            x2 = int(x2 * ratio)
            y2 = int(y2 * ratio)

            color = colors[i % len(colors)]
            pid = f"P{i + 1}"

            # Draw bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label background
            label_w = len(pid) * 8 + 6
            label_h = 16
            label_y = max(0, y1 - label_h)
            draw.rectangle([x1, label_y, x1 + label_w, label_y + label_h], fill=color)
            draw.text((x1 + 3, label_y + 1), pid, fill="white")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_thumbnail(frame_path, width=480, quality=65):
    """Return JPEG bytes of a resized thumbnail (no boxes)."""
    try:
        img = Image.open(frame_path)
        ratio = width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((width, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


def _find_closest_detections(detections_data, clip_start, clip_duration, frame_pct):
    """Find the sampled frame closest to a representative frame's timestamp."""
    if not detections_data or not detections_data.get("sampled_frames"):
        return []
    target_ts = clip_start + frame_pct * clip_duration
    sampled = detections_data["sampled_frames"]
    closest = min(sampled, key=lambda f: abs(f["timestamp_sec"] - target_ts))
    return closest.get("persons", [])


def render_frames(clip):
    frames_dir = clip.get("frames_dir", "")
    clip_id = clip.get("clip_id", "")
    if not frames_dir and not clip_id:
        st.warning("No frames available.")
        return

    # Try multiple locations for frames
    candidates = [
        os.path.join(OUTPUT_DIR, frames_dir) if frames_dir else "",
        os.path.join(PROJECT_ROOT, "active_batch", "frames", clip_id),
        os.path.join(PROJECT_ROOT, frames_dir) if frames_dir else "",
    ]
    abs_frames_dir = next((p for p in candidates if p and os.path.exists(p)), "")
    if not abs_frames_dir:
        st.warning("No frames available.")
        return

    # Load detection data for bounding boxes
    detections = load_clip_detections(clip_id)
    clip_start = float(clip.get("source_start_sec", 0) or 0)
    clip_dur = float(clip.get("clip_duration_sec", 15) or 15)

    # Detect available frames
    frame_files = sorted(glob.glob(os.path.join(abs_frames_dir, "frame_*.jpg")))
    n_frames = len(frame_files)
    if n_frames == 0:
        st.warning("No frames found")
        return
    labels = [f"{int(100*i/(n_frames-1))}%" if n_frames > 1 else "0%" for i in range(n_frames)]
    pcts = [i/(n_frames-1) if n_frames > 1 else 0.0 for i in range(n_frames)]

    # Display in rows of 4
    for row_start in range(0, n_frames, 4):
        cols = st.columns(min(4, n_frames - row_start))
        for j, col in enumerate(cols):
            i = row_start + j
            path = frame_files[i]

            # Try to get detection data for this frame's timestamp
            persons = _find_closest_detections(detections, clip_start, clip_dur, pcts[i])

            if persons:
                thumb = get_thumbnail_with_boxes(path, persons)
                n_persons = len(persons)
                caption = f"Frame {i+1} ({labels[i]}) - {n_persons} person{'s' if n_persons != 1 else ''}"
            else:
                thumb = get_thumbnail(path)
                caption = f"Frame {i+1} ({labels[i]})"

        if thumb:
            col.image(thumb, caption=caption)
        else:
            col.image(path, caption=caption)


def render_frames_clean(clip):
    """Render frames without YOLO bounding boxes.

    YOLO person IDs are per-frame and don't match VLM worker IDs,
    so showing YOLO P labels is misleading. Show clean thumbnails instead.
    """
    frames_dir = clip.get("frames_dir", "")
    clip_id = clip.get("clip_id", "")
    if not frames_dir and not clip_id:
        st.warning("No frames available.")
        return

    candidates = [
        os.path.join(OUTPUT_DIR, frames_dir) if frames_dir else "",
        os.path.join(PROJECT_ROOT, "active_batch", "frames", clip_id),
        os.path.join(PROJECT_ROOT, frames_dir) if frames_dir else "",
    ]
    abs_frames_dir = next((p for p in candidates if p and os.path.exists(p)), "")
    if not abs_frames_dir:
        st.warning("No frames available.")
        return

    frame_files = sorted(glob.glob(os.path.join(abs_frames_dir, "frame_*.jpg")))
    n_frames = len(frame_files)
    if n_frames == 0:
        st.warning("No frames found")
        return
    labels = [f"{int(100*i/(n_frames-1))}%" if n_frames > 1 else "0%" for i in range(n_frames)]

    for row_start in range(0, n_frames, 4):
        cols = st.columns(min(4, n_frames - row_start))
        for j, col in enumerate(cols):
            i = row_start + j
            path = frame_files[i]
            thumb = get_thumbnail(path)
            caption = f"Frame {i+1} ({labels[i]})"
            if thumb:
                col.image(thumb, caption=caption)
            else:
                col.image(path, caption=caption)


def render_clip_metadata(clip):
    st.sidebar.subheader("Clip Info")
    st.sidebar.text(f"ID: {clip.get('clip_id', '')}")
    st.sidebar.text(f"Camera: {clip.get('camera_id', '')}")
    st.sidebar.text(f"Date: {clip.get('date', '')}")
    st.sidebar.text(f"Duration: {clip.get('clip_duration_sec', '')}s")
    st.sidebar.text(f"Max persons: {clip.get('max_persons', '')}")
    st.sidebar.text(f"BRISQUE: {clip.get('brisque_score', '')}")
    st.sidebar.text(f"Condition: {clip.get('visual_condition_auto', '')}")


def render_person_form(person_idx, vlm_suggestion=None, clip_id=""):
    """Render annotation form for a single person with all addendum v2 fields."""
    pid = f"P{person_idx + 1}"

    # Show VLM's position + physical description + action description
    vlm_position = ""
    vlm_physical = ""
    vlm_desc = ""
    vlm_tool = ""
    if vlm_suggestion:
        vlm_position = vlm_suggestion.get("position", "")
        vlm_physical = vlm_suggestion.get("physical_description", "")
        vlm_desc = vlm_suggestion.get("free_text_description", "")
        vlm_tool = vlm_suggestion.get("tool_or_equipment", "")

    header = f"### {pid}"
    if vlm_position:
        header += f" — *{vlm_position}*"
    st.markdown(header)
    if vlm_physical:
        st.markdown(f"**Appearance:** {vlm_physical}")
    if vlm_desc:
        st.caption(f"{vlm_desc}")
    elif vlm_tool:
        st.caption(f"Using: {vlm_tool}")

    col1, col2 = st.columns([3, 1])

    # Action class
    default_action = 0
    if vlm_suggestion:
        suggested = vlm_suggestion.get("action_code", "")
        for i, opt in enumerate(ACTION_DROPDOWN):
            if suggested and suggested in opt:
                default_action = i
                break

    action_label = col1.selectbox(
        f"Action Class ({pid})", ACTION_DROPDOWN,
        index=default_action, key=f"action_{clip_id}_{person_idx}"
    )
    action_code = ""
    action_name = ""
    if action_label:
        action_code = action_label.split("]")[0].replace("[", "").strip()
        action_name = ACTION_TAXONOMY.get(action_code, "")

    position = col2.selectbox(
        f"Position ({pid})", ["left", "center", "right", "foreground", "background"],
        key=f"pos_{clip_id}_{person_idx}"
    )

    # Per-person free-text fields (NEW: physical_description + free_text_description).
    # OPTIONAL — pre-filled from VLM, annotator can edit. Validator does not
    # enforce these as required to avoid forcing 74% re-annotation of legacy
    # records that pre-date these widgets.
    physical_description = st.text_input(
        f"Appearance / physical description ({pid}) — optional",
        value=vlm_physical or "",
        key=f"physical_desc_{clip_id}_{person_idx}",
        help="e.g., 'man in blue shirt and white helmet'. Pre-filled from VLM, edit if needed."
    )
    free_text_description = st.text_area(
        f"Activity description ({pid}) — optional",
        value=vlm_desc or "",
        key=f"free_text_{clip_id}_{person_idx}",
        height=60,
        help="Free-text description of what this worker is doing. Pre-filled from VLM."
    )

    # X1 unlisted action fields (conditional)
    unlisted_action = None
    taxonomy_layer = 1
    if action_code == "X1":
        taxonomy_layer = 2
        st.info(f"Unlisted Action Details ({pid}) - Required for X1")

        # Prefill X1 details from VLM suggestion if available
        vlm_x1 = {}
        if vlm_suggestion and vlm_suggestion.get("action_code") == "X1":
            vlm_x1 = vlm_suggestion.get("unlisted_action", {})

        ua_c1, ua_c2 = st.columns(2)
        closest_class_opts = [""] + [c for c in ACTION_TAXONOMY.keys() if c != "X1"]
        default_closest = 0
        vlm_closest = vlm_x1.get("closest_existing_class", "")
        if vlm_closest in closest_class_opts:
            default_closest = closest_class_opts.index(vlm_closest)
        closest_existing = ua_c1.selectbox(
            f"Closest existing class ({pid})", closest_class_opts,
            index=default_closest, key=f"closest_{clip_id}_{person_idx}")
        why_not = ua_c2.text_input(
            f"Why doesn't it fit? ({pid})",
            value=vlm_x1.get("why_not_existing", ""),
            key=f"whynot_{clip_id}_{person_idx}")
        free_text = st.text_input(
            f"Free-text description ({pid})",
            value=vlm_x1.get("free_text_description", ""),
            key=f"freetext_{clip_id}_{person_idx}")
        ua_c3, ua_c4 = st.columns(2)
        tool_equip = ua_c3.text_input(
            f"Tool/equipment ({pid})",
            value=vlm_x1.get("tool_or_equipment", ""),
            key=f"tool_{clip_id}_{person_idx}")
        industry_specific = ua_c4.checkbox(
            f"Industry-specific ({pid})",
            value=vlm_x1.get("industry_specific", True),
            key=f"industry_{clip_id}_{person_idx}")
        unlisted_action = {
            "closest_existing_class": closest_existing,
            "why_not_existing": why_not,
            "free_text_description": free_text,
            "tool_or_equipment": tool_equip,
            "industry_specific": industry_specific,
        }

    # Spatial context
    default_spatial = ["ground_level"]
    if vlm_suggestion:
        s = vlm_suggestion.get("spatial_context", [])
        if s:
            default_spatial = [t for t in s if t in SPATIAL_TAGS]

    spatial_context = st.multiselect(
        f"Spatial Context ({pid})", SPATIAL_TAGS,
        default=default_spatial, key=f"spatial_{clip_id}_{person_idx}"
    )

    # PPE assessment
    st.markdown(f"*PPE Assessment ({pid})*")
    ppe_cols = st.columns(5)
    ppe = {}
    for p_idx, item in enumerate(PPE_ITEMS):
        values = PPE_VALUES_EXTENDED if item in PPE_ITEMS_WITH_NA else PPE_VALUES
        default_val = 0
        if vlm_suggestion:
            suggested_val = vlm_suggestion.get("ppe", {}).get(item, "")
            if suggested_val in values:
                default_val = values.index(suggested_val)
        ppe[item] = ppe_cols[p_idx].selectbox(
            item.replace("_", " ").title(), values,
            index=default_val, key=f"ppe_{clip_id}_{person_idx}_{item}"
        )

    # Occlusion assessment
    occ_cols = st.columns(2)
    default_occ_level = 0
    if vlm_suggestion:
        suggested_occ = vlm_suggestion.get("occlusion_level", "")
        if suggested_occ in OCCLUSION_LEVELS:
            default_occ_level = OCCLUSION_LEVELS.index(suggested_occ)
    occlusion_level = occ_cols[0].selectbox(
        f"Occlusion Level ({pid})", OCCLUSION_LEVELS,
        index=default_occ_level, key=f"occ_level_{clip_id}_{person_idx}"
    )

    default_occ_sources = []
    if vlm_suggestion:
        os_list = vlm_suggestion.get("occlusion_source", [])
        if os_list:
            default_occ_sources = [s for s in os_list if s in OCCLUSION_SOURCES]
    occlusion_source = []
    if occlusion_level != "fully_visible":
        occlusion_source = occ_cols[1].multiselect(
            f"Occlusion Source ({pid})", OCCLUSION_SOURCES,
            default=default_occ_sources, key=f"occ_source_{clip_id}_{person_idx}"
        )

    # Unsafe act
    unsafe_act = st.text_input(
        f"Unsafe Act ({pid}) - leave empty if none",
        value=vlm_suggestion.get("unsafe_act", "") if vlm_suggestion else "",
        key=f"unsafe_{clip_id}_{person_idx}"
    )

    # Interaction
    default_interaction = 0
    if vlm_suggestion:
        suggested_int = vlm_suggestion.get("interaction", "")
        if suggested_int in INTERACTION_TYPES:
            default_interaction = INTERACTION_TYPES.index(suggested_int)

    interaction = st.selectbox(
        f"Interaction ({pid})", INTERACTION_TYPES,
        index=default_interaction, key=f"interaction_{clip_id}_{person_idx}"
    )

    # Group flag
    default_flag = 0
    if vlm_suggestion:
        suggested_flag = vlm_suggestion.get("group_flag", "")
        if suggested_flag in GROUP_FLAGS:
            default_flag = GROUP_FLAGS.index(suggested_flag)

    group_flag = st.selectbox(
        f"Group Flag ({pid})", GROUP_FLAGS,
        index=default_flag, key=f"group_{clip_id}_{person_idx}"
    )

    # Coordination fields (shown when coordinated)
    coordinated_with = []
    role_in_coordination = ""
    if group_flag == "coordinated":
        other_pids = [f"P{j+1}" for j in range(10) if j != person_idx]
        coordinated_with = st.multiselect(
            f"Coordinated with ({pid})", other_pids,
            key=f"coord_with_{clip_id}_{person_idx}")
        role_in_coordination = st.text_input(
            f"Role in coordination ({pid})",
            placeholder="e.g., signaller, lifter, guide",
            key=f"coord_role_{clip_id}_{person_idx}")

    # Per-person confidence (NEW). OPTIONAL — pre-filled from VLM if available.
    default_p_conf = 0.8
    if vlm_suggestion and isinstance(vlm_suggestion.get("confidence"), (int, float)):
        default_p_conf = float(vlm_suggestion["confidence"])
    p_confidence = st.slider(
        f"Your confidence ({pid}) — optional",
        0.0, 1.0, default_p_conf, 0.05,
        key=f"p_confidence_{clip_id}_{person_idx}",
        help="How confident are you in your annotation of this specific worker? Pre-filled from VLM."
    )

    st.divider()

    # Normalize empty unsafe_act → "none" so the saved record always has a
    # non-empty value (per the new convention)
    unsafe_act_normalized = (unsafe_act or "").strip()
    if not unsafe_act_normalized:
        unsafe_act_normalized = "none"

    person_data = {
        "person_id": pid,
        "position": position,
        "action_code": action_code,
        "action_name": action_name,
        "taxonomy_layer": taxonomy_layer,
        "physical_description": (physical_description or "").strip(),
        "free_text_description": (free_text_description or "").strip(),
        "spatial_context": spatial_context,
        "ppe": ppe,
        "occlusion_level": occlusion_level,
        "occlusion_source": occlusion_source,
        "unsafe_act": unsafe_act_normalized,
        "interaction": interaction,
        "group_flag": group_flag,
        "coordinated_with": coordinated_with,
        "role_in_coordination": role_in_coordination,
        "confidence": p_confidence,
    }
    if unlisted_action:
        person_data["unlisted_action"] = unlisted_action
    return person_data


def render_safety_review_form(clip, clip_id, vlm_suggestion=None):
    """Render safety officer review form (tier_3)."""
    st.subheader("Safety Rule Compliance Review")

    # Load applicable safety rules from camera zone / site metadata.
    # Uses the new site-based matching (safety_rules.py post Phase 2 refactor).
    from annotation_tool.safety_rules import (
        get_applicable_rules, load_safety_rules, load_camera_zones,
    )
    rules_config = load_safety_rules()
    zones_config = load_camera_zones()
    camera_id = clip.get("camera_id", "")
    site = clip.get("site", "")
    work_area = clip.get("work_area", "")
    zone_info = get_applicable_rules(
        camera_id, rules_config, zones_config,
        site=site, work_area=work_area,
    )

    # Flatten new {general_rules, zone_rules, match_basis} shape into the
    # list the downstream safety-review form consumes.
    applicable_rules = []
    for rid, desc in (zone_info.get("general_rules") or {}).items():
        applicable_rules.append({"rule_id": rid, "description": desc, "scope": "general"})
    for dept, info in (zone_info.get("zone_rules") or {}).items():
        for rid, desc in (info.get("observations") or {}).items():
            applicable_rules.append({"rule_id": rid, "description": desc, "scope": dept})
    zones = list((zone_info.get("zone_rules") or {}).keys()) or [zone_info.get("match_basis", "general")]

    st.info(f"Camera: {camera_id} | Site: {site} | Applicable zones/depts: {', '.join(zones)} | "
            f"Rules: {len(applicable_rules)}")

    # ---- VLM Safety Reasoning (NEW) ----
    vlm_unsafe_workers = []
    if vlm_suggestion:
        for i, p in enumerate(vlm_suggestion.get("persons", [])):
            ua = (p.get("unsafe_act") or "").strip()
            if ua and ua.lower() != "none":
                vlm_unsafe_workers.append({
                    "person_id": f"P{i+1}",
                    "physical_description": p.get("physical_description", ""),
                    "position": p.get("position", ""),
                    "action_code": p.get("action_code", ""),
                    "unsafe_act": ua,
                    "ppe": p.get("ppe", {}),
                })

    vlm_flagged_unsafe = len(vlm_unsafe_workers) > 0

    if vlm_flagged_unsafe:
        st.error(f"VLM flagged this clip as UNSAFE — {len(vlm_unsafe_workers)} worker(s) with violations")
        with st.expander("VLM reasoning (click to expand)", expanded=True):
            for w in vlm_unsafe_workers:
                st.markdown(
                    f"**{w['person_id']}** ({w['action_code']}) — *{w['position']}*  \n"
                    f"Appearance: {w['physical_description']}  \n"
                    f"**Unsafe act:** `{w['unsafe_act']}`  \n"
                    f"PPE: helmet={w['ppe'].get('helmet','?')}, "
                    f"vest={w['ppe'].get('high_vis_vest','?')}, "
                    f"harness={w['ppe'].get('harness','?')}, "
                    f"shoes={w['ppe'].get('safety_shoes','?')}"
                )
                st.markdown("---")
    else:
        st.success("VLM flagged this clip as SAFE — no unsafe acts detected")

    # ---- Safety Officer Agree/Disagree with VLM (NEW) ----
    st.markdown("**Do you agree with the VLM's safety assessment?**")
    agree_cols = st.columns([1, 1, 2])
    agree_with_vlm = agree_cols[0].radio(
        "VLM assessment",
        ["Agree", "Disagree"],
        key=f"vlm_agree_{clip_id}",
        horizontal=True,
    )
    disagree_reason = ""
    if agree_with_vlm == "Disagree":
        disagree_reason = agree_cols[2].text_area(
            "Reason for disagreement (REQUIRED)",
            placeholder=("Why is the VLM wrong? E.g., 'VLM said no_helmet but worker P3 "
                         "is wearing one — partially occluded by equipment' or 'VLM missed "
                         "fall hazard for worker on elevated platform'."),
            key=f"vlm_disagree_reason_{clip_id}",
            height=80,
        )

    # Load existing tier_1 annotation for this clip
    tier1_ann = None
    if os.path.exists(ANNOTATIONS_DIR):
        for ann_dir_name in os.listdir(ANNOTATIONS_DIR):
            ann_path = os.path.join(ANNOTATIONS_DIR, ann_dir_name, f"{clip_id}.json")
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    tier1_ann = json.load(f)
                break

    if tier1_ann:
        with st.expander("Tier 1 Annotation (reference)"):
            st.json(tier1_ann)

    # Overall safety compliance (officer's independent judgment)
    st.markdown("---")
    st.markdown("**Overall Safety Compliance (your independent assessment)**")
    overall_cols = st.columns([1, 2])
    overall_compliance = overall_cols[0].selectbox(
        "Is this clip compliant?",
        ["compliant", "not_compliant", "cannot_determine"],
        key=f"overall_safety_{clip_id}",
    )
    safety_description = ""
    if overall_compliance == "not_compliant":
        safety_description = overall_cols[1].text_area(
            "Describe the safety concern",
            placeholder="What violation do you observe? Which workers are involved? What is the risk?",
            key=f"safety_desc_{clip_id}",
        )
    elif overall_compliance == "cannot_determine":
        safety_description = overall_cols[1].text_input(
            "Why can't you determine?",
            placeholder="e.g., view blocked, too far, unclear activity",
            key=f"safety_desc_{clip_id}",
        )
    st.markdown("---")

    # Per-rule review (if rules are applicable to this zone)
    per_rule = []
    if applicable_rules:
        st.markdown("**Per-Rule Compliance**")
        for rule in applicable_rules:
            rule_id = rule.get("rule_id", "")
            rule_desc = rule.get("description", "")
            st.markdown(f"**{rule_id}**: {rule_desc}")
            r_cols = st.columns([2, 3])
            status = r_cols[0].selectbox(
                f"Status ({rule_id})",
                ["compliant", "violation", "not_evaluable"],
                key=f"safety_{clip_id}_{rule_id}",
            )
            evidence = r_cols[1].text_input(
                f"Evidence ({rule_id})", key=f"evidence_{clip_id}_{rule_id}")
            per_rule.append({
                "rule_id": rule_id,
                "status": status,
                "evidence": evidence,
            })

    return {
        "overall_compliance": overall_compliance,
        "safety_description": safety_description,
        "applicable_rules": [r.get("rule_id", "") for r in applicable_rules],
        "zones": zones,
        "per_rule": per_rule,
        # VLM agreement tracking (NEW)
        "vlm_flagged_unsafe": vlm_flagged_unsafe,
        "vlm_unsafe_workers": vlm_unsafe_workers,
        "agree_with_vlm": agree_with_vlm,
        "disagree_reason": disagree_reason if agree_with_vlm == "Disagree" else "",
    }


# ---------- Main App ----------

def main():
    st.set_page_config(page_title="SteelBench Annotation Tool", layout="wide")
    st.title("SteelBench Annotation Tool")

    # Annotator login
    if "annotator_id" not in st.session_state:
        st.session_state.annotator_id = ""

    raw_annotator_id = st.sidebar.text_input(
        "Annotator ID (lowercase)", value=st.session_state.annotator_id,
        help="Enter your assigned ID exactly as given (e.g., annotator_1, expert_1, safety_officer)"
    )
    annotator_id = raw_annotator_id.strip().lower()
    st.session_state.annotator_id = annotator_id

    if not annotator_id:
        st.info("Enter your Annotator ID in the sidebar to begin.")
        return

    # Block inactive annotators
    INACTIVE_ANNOTATORS = {"annotator_5", "annotator_6", "annotator_7",
                           "annotator_8", "annotator_9"}
    if annotator_id in INACTIVE_ANNOTATORS:
        st.error(f"Annotator '{annotator_id}' is inactive. "
                 "Your clips have been redistributed to active annotators. "
                 "Contact the experiment lead for reassignment.")
        return

    # Load assignments and determine role
    assignments = load_assignments()
    annotator_role = "tier_1"  # default
    annotator_data = None

    if assignments and annotator_id in assignments.get("assignments", {}):
        annotator_data = assignments["assignments"][annotator_id]
        annotator_role = annotator_data.get("role", "tier_1")
        st.sidebar.success(f"Role: {annotator_role.replace('_', ' ').title()}")
        st.sidebar.caption(ANNOTATOR_ROLES.get(annotator_role, ""))
    elif assignments and assignments.get("experiment_name"):
        # Experiment mode — reject unknown annotators
        st.error(f"Annotator '{annotator_id}' is not registered for experiment "
                f"'{assignments.get('experiment_name', '')}'.")
        st.info("Contact the experiment lead to be added to the config.")
        return
    else:
        st.sidebar.info("No experiment config — open mode")

    # Load clips based on role
    all_clips = load_clips()
    if not all_clips:
        st.error(f"No clips found. Expected manifest at: {TIER_A_MANIFEST}")
        return

    # Filter clips based on assignment
    if annotator_data and annotator_data.get("clips"):
        # Build lookup of assignment metadata (priority, calibration_condition)
        assign_meta = {}
        for ac in annotator_data["clips"]:
            if isinstance(ac, dict):
                assign_meta[ac["clip_id"]] = ac

        assigned_ids = set(assign_meta.keys())
        # Tier_3 (safety officer) also pulls dynamic escalations from tier3_queue.json
        # (confirmed violations pushed up from tier_2 at save time).
        if annotator_role == "tier_3":
            dynamic_q = load_tier3_queue()
            assigned_ids |= {q["clip_id"] for q in dynamic_q}
            if dynamic_q:
                st.sidebar.metric("Dynamic SO escalations", len(dynamic_q))
        clips = [c for c in all_clips if c.get("clip_id", "") in assigned_ids]

        # Merge assignment metadata (priority, calibration_condition) into clip objects
        for c in clips:
            cid = c.get("clip_id", "")
            if cid in assign_meta:
                meta = assign_meta[cid]
                if meta.get("priority"):
                    c["priority"] = meta["priority"]
                if meta.get("calibration_condition"):
                    c["calibration_condition"] = meta["calibration_condition"]
    elif annotator_role == "tier_2":
        # Tier 2 gets dynamic queue
        queue = load_tier2_queue()
        queue_ids = {q["clip_id"] for q in queue}
        clips = [c for c in all_clips if c.get("clip_id", "") in queue_ids]
        st.sidebar.metric("Review Queue", len(clips))
    else:
        clips = all_clips

    if not clips:
        if annotator_role == "tier_2":
            st.info("No clips in review queue yet. Clips appear here when "
                   "Tier 1 annotators flag clips or produce disagreements.")
        else:
            st.info("No clips assigned.")
        return

    st.sidebar.markdown(f"**Clips: {len(clips)}**")

    existing = load_existing_annotations(annotator_id)
    progress = len(existing)

    # Sidebar reference docs (tier_1 only)
    if annotator_role == "tier_1":
        st.sidebar.divider()
        with st.sidebar.expander("Safety Reference (Annotator Only)"):
            st.markdown("""
            **Common Unsafe Acts:**
            - Working without required PPE
            - Working under suspended load
            - Lone worker in confined space
            - No fire watch during hot work
            - Missing spotter for vehicle operations

            *VLMs are evaluated zero-shot without these references.*
            """)

    # Progress
    st.sidebar.divider()
    st.sidebar.subheader("Progress")
    st.sidebar.metric("Annotated", f"{progress} / {len(clips)}")
    st.sidebar.progress(min(1.0, progress / len(clips)) if clips else 0)

    unannotated = [c for c in clips if c.get("clip_id", "") not in existing]
    if not unannotated:
        st.success("All clips annotated!")
        return

    # Sort by priority: urgent > high > normal
    # Urgent clips BLOCK all others until completed
    urgent_pri = [c for c in unannotated if c.get("priority") == "urgent"]
    high_pri = [c for c in unannotated if c.get("priority") == "high"]
    normal_pri = [c for c in unannotated if c.get("priority") not in ("urgent", "high")]

    if urgent_pri:
        # BLOCK all other clips until urgent are done
        unannotated = urgent_pri
        if annotator_role == "tier_1":
            st.sidebar.warning(
                f"🔴 **{len(urgent_pri)} urgent blind clips** must be completed first. "
                f"Please annotate from scratch — no AI suggestions are shown. "
                f"Other clips will appear after these are done."
            )
    elif high_pri:
        unannotated = high_pri + normal_pri
        if annotator_role == "tier_1":
            st.sidebar.info(
                f"**{len(high_pri)} high-priority clips** at the top of your queue. "
                f"Please complete these first."
            )

    # Expert gate: for tier_2, sort clips by priority:
    #   0. HIGH-PRIORITY first (blind audit clips + proper-chain candidates)
    #   1. GT batch (expert_gold) — expert is first human, annotate from VLM
    #   2. Adjudication — disagreement clips needing expert resolution
    #   3. Audit with tier_1 — review human work (proper chain)
    #   4. Audit without tier_1 — tier_1 hasn't done these yet, skip for now
    if annotator_role == "tier_2":
        # Split high-priority clips FIRST (blind + proper-chain audit)
        high_priority = [c for c in unannotated if c.get("priority") == "high"]
        normal_clips = [c for c in unannotated if c.get("priority") != "high"]

        # Sort high-priority: blind first, then proper-chain
        high_blind = [c for c in high_priority if c.get("calibration_condition") == "blind"]
        high_other = [c for c in high_priority if c.get("calibration_condition") != "blind"]

        # Sort normal clips by task type
        gt_batch = [c for c in normal_clips if c.get("task") == "expert_gold"]
        adjudication = [c for c in normal_clips if c.get("task") == "adjudication"]
        audit_with_t1 = []
        audit_no_t1 = []
        for c in normal_clips:
            if c.get("task") == "audit":
                if find_latest_tier1(c.get("clip_id", ""), ANNOTATIONS_DIR):
                    audit_with_t1.append(c)
                else:
                    audit_no_t1.append(c)
        known_cids = set(c.get("clip_id") for c in gt_batch + adjudication + audit_with_t1 + audit_no_t1)
        other = [c for c in normal_clips if c.get("clip_id") not in known_cids]

        unannotated = high_blind + high_other + gt_batch + adjudication + audit_with_t1 + audit_no_t1 + other

        st.sidebar.markdown("**Expert Queue Priority:**")
        st.sidebar.markdown(
            f"0. **HIGH PRIORITY: {len(high_priority)}** clips "
            f"(blind: {len(high_blind)}, proper-chain: {len(high_other)})\n"
            f"1. GT batch: {len(gt_batch)} clips\n"
            f"2. Adjudication: {len(adjudication)} clips\n"
            f"3. Audit (tier_1 ready): {len(audit_with_t1)} clips\n"
            f"4. Audit (awaiting tier_1): {len(audit_no_t1)} clips"
        )

    # Navigation
    if "clip_index" not in st.session_state:
        st.session_state.clip_index = 0

    nav = st.columns([1, 3, 1])
    if nav[0].button("Previous"):
        st.session_state.clip_index = max(0, st.session_state.clip_index - 1)
    if nav[2].button("Next"):
        st.session_state.clip_index = min(len(unannotated) - 1,
                                           st.session_state.clip_index + 1)

    idx = min(st.session_state.clip_index, len(unannotated) - 1)
    clip = unannotated[idx]
    clip_id = clip.get("clip_id", "")
    nav[1].markdown(f"**Clip {idx+1} of {len(unannotated)}** | `{clip_id}`")

    # Start timer for this clip
    clip_start_key = f"clip_start_time_{clip_id}"
    if clip_start_key not in st.session_state:
        st.session_state[clip_start_key] = datetime.now().isoformat()

    render_clip_metadata(clip)

    # ---------- Prefill: single source of truth (D5) ----------
    vlm_suggestions = load_vlm_suggestions()
    vlm_normalized = vlm_suggestions.get(clip_id)

    # Calibration condition lookup (Phase 2 anchoring bias experiment)
    cal_set = load_calibration_set()
    cal_condition = get_calibration_condition(clip_id, cal_set)

    if cal_condition == "blind":
        # Hide VLM prefill for blind calibration clips
        vlm_normalized = None
        st.warning(
            "Calibration clip (BLIND) — please annotate from scratch using "
            "only the video and frames. No AI suggestions are shown for this clip."
        )
    elif cal_condition == "anchored":
        st.info("Calibration clip (anchored) — VLM prefill shown as normal.")

    # Get the prefill via the dispatcher. For tier_2 reviewing a clip with an
    # existing tier_1 annotation, prefill is the tier_1 record VERBATIM (no
    # merging). VLM is shown separately in a reference panel.
    prefill, prefill_source_label, vlm_reference = get_prefill(
        annotator_role, clip_id, vlm_normalized, ANNOTATIONS_DIR
    )

    # Expert gate: tier_2 should only review clips that have a tier_1
    # annotation. If no tier_1 exists, the expert would be editing VLM
    # directly (source_type=vlm), which breaks GT independence.
    # Show a warning and let the expert skip manually or move to next clip.
    if annotator_role == "tier_2" and vlm_reference is None and prefill is not None and cal_condition != "blind":
        st.warning(
            "**GT Batch Clip — VLM pre-fill only.** "
            "No tier_1 annotator has reviewed this clip yet. "
            "You are the first human reviewer. The form shows VLM suggestions, "
            "NOT a human annotation. Please annotate carefully from scratch."
        )

    # Source type for edit_tracking
    # For blind clips, vlm_reference is None (hidden), but if tier_1 exists
    # the source is still tier_1, not vlm.
    if vlm_reference is not None:
        edit_source_type = "tier_1"
    elif cal_condition == "blind" and prefill is not None and prefill_source_label.startswith("tier_1"):
        edit_source_type = "tier_1"
    else:
        edit_source_type = "vlm"
    flag_reason = (
        get_tier2_queue_reason(clip_id) if annotator_role == "tier_2" else None
    )

    # Prime session_state from prefill BEFORE any widget is created. This
    # defeats stale Streamlit widget state from prior renders/deploys.
    prime_session_state(clip_id, prefill)

    # ---------- Banners ----------
    if annotator_role == "tier_2" and flag_reason:
        # Show only the essential info — what type of disagreement
        # Parse reason: "Safety disagreement: annotator_4 changed unsafe_act from VLM"
        # or "Disagreement: annotator_1=B1 vs annotator_2=F1"
        # or "Flagged by annotator_1 (vlm_count_wrong): reason text"
        if "Safety disagreement" in flag_reason:
            st.error("Safety disagreement — annotator changed unsafe_act from VLM prediction")
        elif "Disagreement:" in flag_reason:
            st.error(f"Action disagreement — {flag_reason}")
        elif "Flagged by" in flag_reason:
            st.error(f"{flag_reason[:150]}")
        else:
            st.error(f"Review needed — {flag_reason[:150]}")

    if prefill:
        nw_for_banner = (prefill or {}).get("num_workers") or 0
        scene_for_banner = (prefill or {}).get("scene_type", "?") or "?"
        if annotator_role == "tier_2" and edit_source_type == "tier_1":
            t1_aid = (prefill or {}).get("annotator_id", "?")
            st.success(
                f"Reviewing **{t1_aid}**'s annotation (human-reviewed) | "
                f"{nw_for_banner} workers | {scene_for_banner}"
            )
        elif annotator_role == "tier_2" and edit_source_type == "vlm":
            st.warning(
                f"Pre-fill: **VLM only (no human review yet)** | "
                f"{nw_for_banner} workers | {scene_for_banner}"
            )
        else:
            st.info(
                f"Pre-fill: **{prefill_source_label}** | "
                f"{nw_for_banner} workers | {scene_for_banner}"
            )

    # VLM reference panel (only shown to experts so they can compare against
    # the VLM's original suggestion). Read-only.
    if vlm_reference is not None:
        with st.expander("VLM suggestion (reference only — for comparison)"):
            st.json(vlm_reference)

    # Backward-compat alias used by the rest of the form
    vlm_suggestion = prefill

    # Safety officer: show additional safety context
    if annotator_role == "tier_3" and annotator_data:
        clip_entry = next(
            (c for c in annotator_data.get("clips", [])
             if c.get("clip_id") == clip_id), None
        )
        if clip_entry:
            is_vlm_unsafe = clip_entry.get("vlm_flagged_unsafe", False)
            priority = clip_entry.get("priority", "normal")
            if is_vlm_unsafe:
                st.error("VLM flagged unsafe act in this clip — mandatory review")
            elif priority == "high":
                st.warning("Safety-critical zone — review safety compliance")

    # Frames — show without YOLO bounding boxes (YOLO IDs don't match VLM worker IDs)
    st.subheader("Representative Frames")
    render_frames_clean(clip)

    # Video (lazy-loaded) — check multiple locations
    clip_rel = clip.get("clip_path", "")
    clip_filename = os.path.basename(clip_rel) if clip_rel else f"{clip_id}.mp4"
    candidate_paths = [
        os.path.join(OUTPUT_DIR, clip_rel),                          # output/clips/...
        os.path.join(PROJECT_ROOT, "active_batch", "clips", clip_filename),  # active_batch/clips/
        os.path.join(PROJECT_ROOT, clip_rel),                        # relative from project root
    ]
    clip_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if clip_path:
        with st.expander("Watch Video Clip"):
            vid_key = f"vid_{clip_id}"
            if st.button("Load Video", key=f"btn_{vid_key}"):
                st.session_state[vid_key] = True
            if st.session_state.get(vid_key, False):
                st.video(clip_path)

    st.divider()

    # VLM suggestion already loaded above for confidence banner
    # vlm_suggestion is set from vlm_suggestions.get(clip_id)

    # ---------- TIER 3: Safety Review ----------
    if annotator_role == "tier_3":
        safety_review = render_safety_review_form(clip, clip_id,
                                                   vlm_suggestion=vlm_normalized)

        col1, col2 = st.columns(2)
        submitted = col1.button("Submit Safety Review", type="primary",
                               width="stretch")
        skipped = col2.button("Skip", width="stretch")
        flagged = False

        if submitted:
            # Strict validation for tier_3 (D15)
            errors = []
            if not safety_review.get("agree_with_vlm"):
                errors.append("agree_with_vlm is required (Agree or Disagree)")
            if (safety_review.get("agree_with_vlm") == "Disagree"
                    and not (safety_review.get("disagree_reason") or "").strip()):
                errors.append("disagree_reason is required when you DISAGREE with VLM")
            if not safety_review.get("overall_compliance"):
                errors.append("overall_compliance is required")
            if (safety_review.get("overall_compliance") in ("not_compliant", "cannot_determine")
                    and not (safety_review.get("safety_description") or "").strip()):
                errors.append("safety_description is required when overall_compliance is not 'compliant'")
            # All applicable rules must have a status set (no None/empty)
            for r in safety_review.get("per_rule", []):
                if not r.get("status"):
                    errors.append(f"per-rule {r.get('rule_id', '?')}: status is required")
            if errors:
                error_lines = "\n".join(f"- {e}" for e in errors)
                st.error(f"Cannot save safety review — please fix:\n\n{error_lines}")
                return

        if submitted or skipped:
            annotation = {
                "clip_id": clip_id,
                "annotator_id": annotator_id,
                "annotator_role": "tier_3",
                "annotator_timestamp": datetime.now().isoformat(),
                "status": "submitted" if submitted else "skipped",
                "source": "human",
                "safety_review": safety_review if submitted else None,
                "calibration_condition": cal_condition,
            }
            save_annotation(annotator_id, annotation)
            load_existing_annotations.clear()
            if submitted:
                st.toast("Safety review submitted ✓", icon="🛡️")
                st.success(f"Safety review saved: {clip_id}")
            else:
                st.toast("Skipped", icon="⏭️")
                st.warning(f"Skipped: {clip_id}")
            st.session_state.clip_index += 1
            st.rerun()
        return

    # ---------- TIER 2: Adjudication reference panels ----------
    if annotator_role == "tier_2":
        st.subheader("Adjudication Review")

        # Show ALL existing tier_1 annotations for this clip (raw JSON for
        # audit). The form below is pre-filled with the latest one.
        if os.path.exists(ANNOTATIONS_DIR):
            tier1_anns = []
            for ann_dir_name in os.listdir(ANNOTATIONS_DIR):
                ann_path = os.path.join(ANNOTATIONS_DIR, ann_dir_name,
                                        f"{clip_id}.json")
                if os.path.exists(ann_path):
                    try:
                        with open(ann_path) as f:
                            tier1_anns.append(json.load(f))
                    except Exception:
                        pass

            if tier1_anns:
                st.caption(
                    f"{len(tier1_anns)} prior annotation(s) for this clip "
                    f"(latest is pre-filled into the form below)"
                )
                for ann in tier1_anns:
                    role = ann.get("annotator_role", "?")
                    aid = ann.get("annotator_id", "?")
                    status = ann.get("status", "?")
                    ts = ann.get("annotator_timestamp", "")[:19]
                    with st.expander(
                        f"{role} • {aid} • {status} • {ts}"
                    ):
                        st.json(ann)
            else:
                st.caption(
                    "No prior tier_1 annotations found for this clip — "
                    "the form below is pre-filled with VLM suggestions only."
                )

    # ---------- TIER 1 / TIER 2: Annotation Form ----------
    st.subheader("Clip-Level Metadata")

    # Determine the annotation_layer from num_workers.
    # All annotators can now adjust num_workers freely (1-30).
    # The annotation_layer is re-derived dynamically from the input.
    # Previously tier_1 was locked within VLM-suggested layer bounds,
    # but this caused ~12% flag rate (vlm_count_wrong) which overloaded
    # expert queues unnecessarily.
    vlm_count = 0
    if vlm_normalized and isinstance(vlm_normalized.get("num_workers"), int):
        vlm_count = int(vlm_normalized["num_workers"])
    elif prefill and isinstance(prefill.get("num_workers"), int):
        vlm_count = int(prefill["num_workers"])
    if vlm_count < 1:
        vlm_count = 1
    vlm_suggested_layer = derive_annotation_layer(vlm_count)
    nw_min, nw_max = 1, 30

    meta_cols = st.columns(4)

    scene_type_options = [""] + SCENE_TYPES
    scene_type = meta_cols[0].selectbox(
        "Scene Type", scene_type_options,
        format_func=lambda x: f"{x} - {SCENE_TYPE_LABELS.get(x, '')}" if x else "Select...",
        key=f"scene_type_{clip_id}",
    )
    visibility = meta_cols[1].selectbox("Visibility", VISIBILITY_LEVELS,
                                        key=f"visibility_{clip_id}")
    # num_workers widget — bounded by layer. Annotator can adjust within
    nw_key = f"num_workers_{clip_id}"
    # Clamp session_state value to bounds
    if nw_key in st.session_state:
        cur = st.session_state[nw_key]
        if not isinstance(cur, int) or cur < nw_min or cur > nw_max:
            st.session_state[nw_key] = max(nw_min, min(nw_max, vlm_count))
    else:
        st.session_state[nw_key] = max(nw_min, min(nw_max, vlm_count))
    _widget_label = f"Number of workers (1-30)"
    _widget_help = (
        f"VLM detected {vlm_count} worker(s) → suggested Layer {vlm_suggested_layer}. "
        f"You can change this to the correct count. The form will automatically "
        f"switch between Layer 1 (scene-level, >5) and Layer 2 (per-person, ≤5)."
        )
    num_workers = meta_cols[2].number_input(
        _widget_label,
        min_value=nw_min, max_value=nw_max,
        key=nw_key,
        help=_widget_help,
    )
    visible_equipment = meta_cols[3].text_input("Visible equipment",
                                                 key=f"visible_equipment_{clip_id}")

    # Visibility conditions (multi-select)
    visibility_conditions = st.multiselect(
        "Visibility Conditions", VISIBILITY_CONDITIONS,
        key=f"visibility_conditions_{clip_id}")

    st.divider()

    # annotation_layer is derived from the CURRENT num_workers value.
    # - Tier 1: num_workers is clamped within the VLM-suggested layer, so this
    #   always equals vlm_suggested_layer (boundary is locked by the widget).
    # - Tier 2 / Tier 3: expert can have crossed the boundary, so this may
    #   differ from vlm_suggested_layer. The per-person section below will
    #   render or hide accordingly. The schema validator will accept the
    #   record because (num_workers, annotation_layer) remain consistent.
    annotation_layer = derive_annotation_layer(int(num_workers or 1))
    vlm_worker_count = int(num_workers or 0)

    # Layer 1 fields (ALL clips) — dominant actions + overall PPE
    st.subheader("Scene-Level Annotation (Layer 1)")

    # Dominant actions — multi-label (select all visible actions). Default
    # comes from session_state primer (set above), so we don't pass `default=`
    dominant_actions_labels = st.multiselect(
        "Dominant Actions (select ALL visible actions in this clip)",
        [f"[{code}] {name}" for code, name in ACTION_TAXONOMY.items()],
        key=f"dominant_actions_{clip_id}",
    )
    dominant_actions = [
        label.split("]")[0].replace("[", "").strip()
        for label in dominant_actions_labels
    ]

    # Overall PPE compliance
    overall_ppe = st.selectbox(
        "Overall PPE Compliance",
        ["compliant", "partial", "non_compliant", "cannot_determine"],
        key=f"overall_ppe_{clip_id}",
    )

    st.divider()

    # Per-person annotations (Layer 2 only) + Layer 1 scene-level unsafe act
    persons = []
    scene_unsafe_act = ""
    if annotation_layer == 2:
        st.subheader(f"Per-Person Annotations (Layer 2) — {int(num_workers)} workers")

        for p_idx in range(int(num_workers)):
            person_vlm = None
            if vlm_suggestion and vlm_suggestion.get("persons"):
                if p_idx < len(vlm_suggestion["persons"]):
                    person_vlm = vlm_suggestion["persons"][p_idx]

            person_data = render_person_form(p_idx, vlm_suggestion=person_vlm, clip_id=clip_id)
            persons.append(person_data)
    else:
        st.info(
            f"Layer 1 — scene-level annotation only ({int(num_workers)} workers detected). "
            f"Per-person fields are disabled for clips with 6+ workers; use the scene-level "
            f"fields above and below."
        )
        # NEW: Layer 1 scene-level unsafe act field. Optional. If non-empty
        # AND clip is safety-critical, the auto-escalation hook escalates.
        scene_unsafe_act = st.text_input(
            "Scene-level unsafe act (optional) — describe any unsafe behavior you observe",
            placeholder="e.g., 'workers near suspended load without spotter'. Leave empty if none.",
            key=f"scene_unsafe_act_{clip_id}",
            help=(
                "Layer 1 clips have no per-person form, so use this field to flag "
                "any scene-level unsafe acts. Used for safety auto-escalation."
            ),
        )

    # Action transitions
    action_transition = {"detected": False, "transitions": []}
    vlm_transition = vlm_suggestion.get("action_transition", {}) if vlm_suggestion else {}
    vlm_detected = vlm_transition.get("detected", False)
    vlm_trans_list = vlm_transition.get("transitions", [])

    if vlm_detected and vlm_trans_list:
        st.subheader("Action Transitions")
        st.caption("VLM detected the following transition(s):")
        for vt in vlm_trans_list:
            if isinstance(vt, dict):
                st.info(f"Worker {vt.get('worker_id','?')}: "
                        f"**{vt.get('from_action','?')}** → **{vt.get('to_action','?')}** "
                        f"({vt.get('frame_range', vt.get('description', ''))})")
            elif isinstance(vt, str):
                st.info(vt)

        agree_transition = st.radio(
            "Do you agree with the VLM's action transition assessment?",
            ["Agree", "Disagree — no transition", "Disagree — different transition"],
            key=f"agree_transition_{clip_id}",
        )

        if agree_transition == "Agree":
            action_transition = vlm_transition
        elif agree_transition == "Disagree — different transition":
            st.caption("Specify the correct transition(s):")
            n_trans = st.number_input("Number of transitions", 1, 5, 1,
                                       key=f"n_transitions_{clip_id}")
            transitions = []
            for t_idx in range(int(n_trans)):
                t_cols = st.columns(3)
                from_act = t_cols[0].selectbox(
                    f"From (T{t_idx+1})", ACTION_DROPDOWN, key=f"tfrom_{clip_id}_{t_idx}")
                to_act = t_cols[1].selectbox(
                    f"To (T{t_idx+1})", ACTION_DROPDOWN, key=f"tto_{clip_id}_{t_idx}")
                at_frame = t_cols[2].selectbox(
                    f"At frame (T{t_idx+1})", list(range(1, 9)), key=f"tframe_{clip_id}_{t_idx}")
                transitions.append({
                    "from_action": from_act.split("]")[0].replace("[", "").strip() if from_act else "",
                    "to_action": to_act.split("]")[0].replace("[", "").strip() if to_act else "",
                    "at_frame": at_frame,
                })
            action_transition = {"detected": True, "transitions": transitions}
        # else: "Disagree — no transition" → keeps default {"detected": False}

    # Group activity (shown for MAC/MAC-X)
    group_activity = ""
    if scene_type in ("MAC", "MAC-X"):
        group_activity = st.text_input(
            "Group activity description",
            placeholder="e.g., Team lifting steel beam",
            key=f"group_activity_{clip_id}")

    # Scene description + confidence (primer handles defaults from prefill)
    scene_desc = st.text_area(
        "Scene description (≥10 chars, required)",
        key=f"scene_desc_{clip_id}",
        help="A 1-2 sentence description of what's happening in the clip.",
    )
    confidence = st.slider(
        "Your confidence", 0.0, 1.0, 0.8, 0.05,
        key=f"confidence_{clip_id}",
    )

    # Submit buttons
    col1, col2, col3, col4 = st.columns(4)
    submitted = col1.button("Submit", type="primary", width="stretch")
    skipped = col2.button("Skip", width="stretch")
    flagged_btn = col3.button("Flag for review", width="stretch")
    discarded = col4.button("Discard", width="stretch")

    # ---------- Inline FLAG form (D13) ----------
    # When the user clicks "Flag for review", show an inline section requiring
    # flag_category + flag_reason. Same UX pattern as the discard reason.
    if flagged_btn:
        st.session_state[f"show_flag_form_{clip_id}"] = True

    flag_category = ""
    flag_reason_text = ""
    confirmed_flag = False
    if st.session_state.get(f"show_flag_form_{clip_id}", False):
        st.warning(
            "FLAG this clip — choose a category and provide a brief reason. "
            "All other fields are optional when flagging."
        )
        flag_cols = st.columns([1, 3])
        flag_category = flag_cols[0].selectbox(
            "Flag category",
            [""] + sorted(FLAG_CATEGORIES),
            key=f"flag_category_{clip_id}",
        )
        flag_reason_text = flag_cols[1].text_area(
            "Flag reason (≥5 chars)",
            placeholder="e.g., 'VLM said 8 workers but only 3 visible'",
            key=f"flag_reason_{clip_id}",
            height=68,
        )
        confirmed_flag = st.button("Confirm Flag", type="primary", key=f"confirm_flag_{clip_id}")
        if confirmed_flag:
            if not flag_category:
                st.error("Please select a flag category before confirming.")
                confirmed_flag = False
            elif not flag_reason_text or len(flag_reason_text.strip()) < 5:
                st.error("Flag reason must be at least 5 characters.")
                confirmed_flag = False

    flagged = confirmed_flag

    # Discard reason (shown above buttons, saved when Discard is clicked)
    discard_reason = ""
    if discarded or st.session_state.get("show_discard_reason", False):
        st.session_state["show_discard_reason"] = True
        DISCARD_REASONS = [
            "",
            "no_persons_visible",
            "too_dark_to_annotate",
            "heavy_occlusion_unusable",
            "steam_dust_no_visibility",
            "corrupt_video",
            "wrong_content",
            "duplicate_clip",
            "other",
        ]
        discard_reason = st.selectbox(
            "Reason for discarding", DISCARD_REASONS,
            key=f"discard_reason_{clip_id}")
        if discard_reason == "other":
            discard_reason = st.text_input(
                "Specify reason", key=f"discard_other_{clip_id}")
        if discarded and not discard_reason:
            st.error("Please select a reason before discarding.")
            discarded = False  # Block until reason provided

    if submitted or skipped or flagged or discarded:
        if discarded:
            status = "discarded"
        elif submitted:
            status = "submitted"
        elif skipped:
            status = "skipped"
        else:
            status = "flagged"

        # Compute time spent on this clip
        clip_start_key = f"clip_start_time_{clip_id}"
        if clip_start_key not in st.session_state:
            st.session_state[clip_start_key] = datetime.now().isoformat()
        clip_start_time = st.session_state.get(clip_start_key, datetime.now().isoformat())
        time_spent_sec = (datetime.now() - datetime.fromisoformat(clip_start_time)).total_seconds()

        annotation = {
            "clip_id": clip_id,
            "annotator_id": annotator_id,
            "annotator_role": annotator_role,
            "annotator_timestamp": datetime.now().isoformat(),
            "status": status,
            "source": "human",
            "annotation_layer": annotation_layer,
            "scene_type": scene_type,
            "num_workers": int(num_workers),
            "dominant_actions": dominant_actions,
            "overall_ppe_compliance": overall_ppe,
            "visibility": visibility,
            "visibility_conditions": visibility_conditions,
            "visible_equipment": visible_equipment,
            "persons": persons,  # Empty [] for Layer 1 clips (enforced by validator)
            "scene_unsafe_act": scene_unsafe_act,  # Layer 1 only, optional
            "action_transition": action_transition,
            "group_activity": group_activity,
            "scene_description": scene_desc,
            "annotator_confidence": confidence,
            "time_spent_sec": round(time_spent_sec, 1),
            "tier": clip.get("tier", "A"),
            "severity_level": clip.get("severity_level", ""),
            "calibration_condition": cal_condition,
            "vlm_suggestion_snapshot": {
                "scene_type": (vlm_normalized or {}).get("scene_type", ""),
                "num_workers": (vlm_normalized or {}).get("num_workers", 0),
                "first_worker_action": (
                    (vlm_normalized or {}).get("persons", [{}])[0].get("action_code", "")
                    if vlm_normalized and vlm_normalized.get("persons") else ""
                ),
            } if vlm_normalized else None,
        }

        # Add discard metadata
        if status == "discarded":
            annotation["discard_reason"] = discard_reason
            annotation["persons"] = []  # No annotation data for discarded clips
            st.session_state["show_discard_reason"] = False

        # Add flag metadata
        if status == "flagged":
            annotation["flag_category"] = flag_category
            annotation["flag_reason"] = flag_reason_text.strip()
            st.session_state[f"show_flag_form_{clip_id}"] = False

        # ---------- VALIDATION GUARD (D3) ----------
        # Save is BLOCKED if validate_annotation returns errors. Annotator
        # sees the missing fields and must fix before re-submitting. Skip,
        # flag, and discard run their own (relaxed) validation in the
        # validator's status branches.
        validation_errors = validate_annotation(annotation)
        if validation_errors:
            error_lines = "\n".join(f"- {e}" for e in validation_errors)
            st.error(
                f"Cannot save — please fix these issues and re-submit:\n\n{error_lines}"
            )
            return  # do NOT save, do NOT advance

        # Track what the human changed vs the prefill source
        annotation["edit_tracking"] = compute_edit_tracking(
            annotation, vlm_suggestion, source_type=edit_source_type
        )

        # Store VLM prefill unsafe_act values for auto-escalation comparison.
        # This lets downstream scripts compare tier_1's final unsafe_act against
        # the VLM's original prediction to detect safety disagreements.
        if vlm_suggestion and edit_source_type == "vlm":
            vlm_unsafe_acts = []
            for p in vlm_suggestion.get("persons", []):
                vlm_unsafe_acts.append(p.get("unsafe_act", "none"))
            annotation["edit_tracking"]["vlm_prefill_unsafe_acts"] = vlm_unsafe_acts
            # Also store scene-level unsafe_act for Layer 1
            annotation["edit_tracking"]["vlm_prefill_scene_unsafe_act"] = \
                vlm_suggestion.get("scene_unsafe_act", "")

        save_annotation(annotator_id, annotation)
        load_existing_annotations.clear()

        # Auto-escalate safety disagreements: if tier_1 edited unsafe_act
        # differently from VLM prefill, add to expert queue for review.
        if (status == "submitted" and annotator_role == "tier_1"
                and vlm_suggestion and edit_source_type == "vlm"):
            vlm_unsafe = [p.get("unsafe_act", "none")
                          for p in vlm_suggestion.get("persons", [])]
            ann_unsafe = [p.get("unsafe_act", "none")
                          for p in annotation.get("persons", [])]
            if vlm_unsafe != ann_unsafe:
                add_to_tier2_queue(
                    clip_id,
                    f"Safety disagreement: {annotator_id} changed unsafe_act from VLM"
                )

        # If a tier_2 expert is submitting (not flagging) a clip that was in the
        # tier_2 review queue, clear it from the queue so it doesn't reappear.
        if annotator_role == "tier_2" and status == "submitted":
            remove_from_tier2_queue(clip_id)

            # Tier_2 → Tier_3 escalation: when an expert confirms an unsafe_act
            # (non-"none"), append to safety-officer audit queue.
            expert_has_unsafe = False
            if annotation_layer == 2:
                expert_has_unsafe = any(
                    (p.get("unsafe_act") or "").strip().lower() not in ("", "none", "none_visible")
                    for p in persons
                )
            elif annotation_layer == 1:
                expert_has_unsafe = bool((scene_unsafe_act or "").strip())
            if expert_has_unsafe:
                add_to_tier3_queue(
                    clip_id,
                    f"Expert {annotator_id} confirmed unsafe_act on Layer {annotation_layer}"
                )

        # Tier_3 submit clears their own queue entry
        if annotator_role == "tier_3" and status == "submitted":
            remove_from_tier3_queue(clip_id)

        # Add to tier_2 queue if flagged
        if flagged:
            queue_reason = f"Flagged by {annotator_id} ({flag_category}): {flag_reason_text.strip()[:80]}"
            add_to_tier2_queue(clip_id, queue_reason)

        # Auto-detect disagreements: tier_1 only, on submit
        if status == "submitted" and annotator_role == "tier_1":
            if os.path.exists(ANNOTATIONS_DIR):
                for other_dir in os.listdir(ANNOTATIONS_DIR):
                    if other_dir == annotator_id or other_dir.startswith("."):
                        continue
                    other_path = os.path.join(ANNOTATIONS_DIR, other_dir, f"{clip_id}.json")
                    if not os.path.exists(other_path):
                        continue
                    try:
                        with open(other_path) as f:
                            other_ann = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue
                    if other_ann.get("annotator_role") != "tier_1":
                        continue
                    if other_ann.get("status") != "submitted":
                        continue
                    # Layer-aware disagreement detection (D14)
                    if annotation_layer == 2 and persons and other_ann.get("persons"):
                        my_action = persons[0].get("action_code", "")
                        other_action = other_ann["persons"][0].get("action_code", "")
                        if my_action and other_action and my_action != other_action:
                            add_to_tier2_queue(
                                clip_id,
                                f"Disagreement: {annotator_id}={my_action} vs {other_dir}={other_action}"
                            )
                    elif annotation_layer == 1:
                        # Layer 1: Jaccard on dominant_actions
                        my_da = set(dominant_actions or [])
                        other_da = set(other_ann.get("dominant_actions") or [])
                        union = my_da | other_da
                        if union:
                            jaccard = len(my_da & other_da) / len(union)
                            if jaccard < 0.5:
                                add_to_tier2_queue(
                                    clip_id,
                                    f"Disagreement: dominant_actions overlap {jaccard:.0%} ({annotator_id} vs {other_dir})"
                                )

            # Auto-escalate safety-critical clips with unsafe acts (D14)
            if annotator_data:
                clip_entry = next(
                    (c for c in annotator_data.get("clips", []) if c.get("clip_id") == clip_id), None
                )
                if clip_entry and clip_entry.get("is_safety_critical"):
                    has_unsafe = False
                    if annotation_layer == 2:
                        # Layer 2: scan persons[].unsafe_act
                        has_unsafe = any(
                            (p.get("unsafe_act") or "").strip().lower() not in ("", "none")
                            for p in persons
                        )
                    elif annotation_layer == 1:
                        # Layer 1: check the new scene_unsafe_act field
                        has_unsafe = bool((scene_unsafe_act or "").strip())
                    if has_unsafe:
                        add_to_tier2_queue(
                            clip_id,
                            f"Safety-critical clip with unsafe act reported by {annotator_id}"
                        )

        # ---------- Notifications (D9) ----------
        if status == "submitted":
            st.toast("Annotation submitted ✓", icon="✅")
            st.success(f"Saved: {clip_id}")
        elif status == "skipped":
            st.toast("Skipped", icon="⏭️")
            st.warning(f"Skipped: {clip_id}")
        elif status == "discarded":
            st.toast(f"Discarded: {discard_reason}", icon="🗑️")
            st.error(f"Discarded: {clip_id} — {discard_reason}")
        else:
            st.toast("Flagged for expert review", icon="⚠️")
            st.warning(f"Flagged: {clip_id} — {flag_category}")

        st.session_state.clip_index += 1
        st.rerun()


if __name__ == "__main__":
    main()
