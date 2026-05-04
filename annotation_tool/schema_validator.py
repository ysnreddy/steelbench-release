"""SteelBench annotation schema validator.

Single source of truth for what constitutes a "valid" annotation record. Used by:
- annotation_tool/app.py — save guard, layer derivation, expert prefill
- scripts/audit_and_repair_annotations.py — categorize and repair existing records
- annotation_tool/qc_dashboard.py — Schema Health tab

Schema invariant: (num_workers ≤ 5) ↔ (annotation_layer == 2) ↔ (persons populated)

Layer 1 = scene-level only, 6+ workers, persons == []
Layer 2 = per-person, ≤5 workers, len(persons) == num_workers

Exception: legacy "bonus" records carry `bonus_per_person: True` and may have a
populated persons array for >5 workers (per-person fields are NOT strictly
validated for these — they pre-date the strict rule).
"""
from __future__ import annotations

from typing import Any

from annotation_tool.agreement import (
    ACTION_TAXONOMY,
    GROUP_FLAGS,
    OCCLUSION_LEVELS,
    PPE_ITEMS,
    PPE_ITEMS_WITH_NA,
    PPE_VALUES,
    PPE_VALUES_EXTENDED,
    SCENE_TYPES,
    SPATIAL_TAGS,
    VISIBILITY_CONDITIONS,
)

# ---------- Constants ----------

OVERALL_PPE_VALUES = {
    "compliant",
    "partial",
    "non_compliant",
    "cannot_determine",
}

FLAG_CATEGORIES = {
    "vlm_count_wrong",
    "image_unclear",
    "missing_frames",
    "ambiguous_scene",
    "wrong_content",
    "other",
}

LAYER_1_NUM_WORKERS_MIN = 6
LAYER_1_NUM_WORKERS_MAX = 30
LAYER_2_NUM_WORKERS_MIN = 1
LAYER_2_NUM_WORKERS_MAX = 5

# Required text-field min lengths (after .strip())
SCENE_DESCRIPTION_MIN_LEN = 10
POSITION_MIN_LEN = 3
PHYSICAL_DESCRIPTION_MIN_LEN = 5
FREE_TEXT_DESCRIPTION_MIN_LEN = 5
FLAG_REASON_MIN_LEN = 5


# ---------- Layer derivation ----------


def derive_annotation_layer(num_workers: int) -> int:
    """Return the layer that *should* apply for a given num_workers count.

    Layer 2 = ≤5 workers (per-person annotation enabled)
    Layer 1 = >5 workers (scene-level only)

    This is the single source of truth for layer derivation. The form, the
    save guard, and the audit script all use this function — never compute
    `2 if nw <= 5 else 1` inline anywhere else.
    """
    try:
        nw = int(num_workers)
    except (TypeError, ValueError):
        nw = 0
    return 2 if nw <= 5 else 1


def num_workers_bounds_for_layer(layer: int) -> tuple[int, int]:
    """Return (min, max) num_workers bounds for a given layer."""
    if layer == 1:
        return (LAYER_1_NUM_WORKERS_MIN, LAYER_1_NUM_WORKERS_MAX)
    return (LAYER_2_NUM_WORKERS_MIN, LAYER_2_NUM_WORKERS_MAX)


# ---------- Field validators ----------


def _is_nonempty_str(v: Any, min_len: int = 1) -> bool:
    return isinstance(v, str) and len(v.strip()) >= min_len


def validate_scene_fields(rec: dict) -> list[str]:
    """Check all required scene-level fields. Returns list of error messages."""
    errors: list[str] = []

    # scene_type
    st = rec.get("scene_type")
    if not _is_nonempty_str(st):
        errors.append("scene_type is required")
    elif st not in SCENE_TYPES:
        errors.append(f"scene_type must be one of {SCENE_TYPES} (got {st!r})")

    # num_workers
    nw_raw = rec.get("num_workers")
    try:
        nw = int(nw_raw)
    except (TypeError, ValueError):
        errors.append("num_workers must be an integer ≥ 1")
        nw = None
    else:
        if nw < 1:
            errors.append("num_workers must be ≥ 1")

    # annotation_layer must be derived correctly from num_workers
    if nw is not None and nw >= 1:
        expected_layer = derive_annotation_layer(nw)
        actual_layer = rec.get("annotation_layer")
        if actual_layer != expected_layer:
            # Allow mismatch only if bonus_per_person legacy flag is set AND
            # the record is a bonus Layer 1 with persons populated.
            if not (
                rec.get("bonus_per_person") is True
                and actual_layer == 1
                and nw > 5
            ):
                errors.append(
                    f"annotation_layer={actual_layer} does not match "
                    f"num_workers={nw} (expected layer={expected_layer})"
                )

        # num_workers must be within layer bounds
        if actual_layer in (1, 2):
            lo, hi = num_workers_bounds_for_layer(actual_layer)
            if not (lo <= nw <= hi):
                errors.append(
                    f"num_workers={nw} outside Layer {actual_layer} bounds [{lo},{hi}]"
                )

    # dominant_actions
    da = rec.get("dominant_actions")
    if not isinstance(da, list) or len(da) == 0:
        errors.append("dominant_actions must have ≥1 entry")
    else:
        for a in da:
            if a not in ACTION_TAXONOMY:
                errors.append(f"dominant_actions contains invalid code {a!r}")

    # scene_description
    sd = rec.get("scene_description")
    if not _is_nonempty_str(sd, SCENE_DESCRIPTION_MIN_LEN):
        errors.append(
            f"scene_description must be ≥{SCENE_DESCRIPTION_MIN_LEN} characters"
        )

    # overall_ppe_compliance
    ppe_overall = rec.get("overall_ppe_compliance")
    if not _is_nonempty_str(ppe_overall):
        errors.append("overall_ppe_compliance is required")
    elif ppe_overall not in OVERALL_PPE_VALUES:
        errors.append(
            f"overall_ppe_compliance must be one of {sorted(OVERALL_PPE_VALUES)} "
            f"(got {ppe_overall!r})"
        )

    # visibility_conditions
    vc = rec.get("visibility_conditions")
    if not isinstance(vc, list) or len(vc) == 0:
        errors.append("visibility_conditions must have ≥1 entry")
    else:
        for v in vc:
            if v not in VISIBILITY_CONDITIONS:
                errors.append(
                    f"visibility_conditions contains invalid value {v!r}"
                )

    # action_transition.detected
    at = rec.get("action_transition")
    if not isinstance(at, dict):
        errors.append("action_transition must be a dict")
    else:
        detected = at.get("detected")
        if not isinstance(detected, bool):
            errors.append("action_transition.detected must be bool")
        elif detected:
            transitions = at.get("transitions")
            if not isinstance(transitions, list) or len(transitions) == 0:
                errors.append(
                    "action_transition.transitions must be non-empty when detected=True"
                )

    # annotator_confidence
    conf = rec.get("annotator_confidence")
    if not isinstance(conf, (int, float)):
        errors.append("annotator_confidence must be a number in [0.0, 1.0]")
    elif not (0.0 <= float(conf) <= 1.0):
        errors.append(
            f"annotator_confidence={conf} outside [0.0, 1.0]"
        )

    return errors


def validate_person(person: dict, p_idx: int) -> list[str]:
    """Check all required per-person fields. Returns list of error messages.

    p_idx is the 0-based index used in error messages (rendered as P{p_idx+1}).
    """
    errors: list[str] = []
    pid = f"P{p_idx + 1}"

    if not isinstance(person, dict):
        return [f"{pid}: not a dict"]

    # action_code
    ac = person.get("action_code")
    if not _is_nonempty_str(ac):
        errors.append(f"{pid}.action_code is required")
    elif ac not in ACTION_TAXONOMY:
        errors.append(f"{pid}.action_code={ac!r} not in ACTION_TAXONOMY")

    # position
    pos = person.get("position")
    if not _is_nonempty_str(pos, POSITION_MIN_LEN):
        errors.append(
            f"{pid}.position must be ≥{POSITION_MIN_LEN} characters"
        )

    # NOTE: physical_description, free_text_description, and per-person
    # confidence are NOT strictly required by the validator. The original
    # render_person_form never captured them (only displayed VLM values as
    # headers/captions). Going forward, the new form adds them as OPTIONAL
    # text inputs/sliders that prefill from VLM and can be edited by the
    # annotator, but they are not mandatory. This matches the existing
    # ~89-record pilot data without forcing 74% re-annotation.

    # ppe — all 5 items required
    ppe = person.get("ppe")
    if not isinstance(ppe, dict):
        errors.append(f"{pid}.ppe must be a dict")
    else:
        for item in PPE_ITEMS:
            v = ppe.get(item)
            if not _is_nonempty_str(v):
                errors.append(f"{pid}.ppe.{item} is required")
                continue
            allowed = (
                PPE_VALUES_EXTENDED if item in PPE_ITEMS_WITH_NA else PPE_VALUES
            )
            if v not in allowed:
                errors.append(
                    f"{pid}.ppe.{item}={v!r} must be one of {allowed}"
                )

    # spatial_context
    sc = person.get("spatial_context")
    if not isinstance(sc, list) or len(sc) == 0:
        errors.append(f"{pid}.spatial_context must have ≥1 entry")
    else:
        for tag in sc:
            if tag not in SPATIAL_TAGS:
                errors.append(
                    f"{pid}.spatial_context contains invalid tag {tag!r}"
                )

    # occlusion_level
    ol = person.get("occlusion_level")
    if not _is_nonempty_str(ol):
        errors.append(f"{pid}.occlusion_level is required")
    elif ol not in OCCLUSION_LEVELS:
        errors.append(
            f"{pid}.occlusion_level={ol!r} must be one of {OCCLUSION_LEVELS}"
        )

    # group_flag
    gf = person.get("group_flag")
    if not _is_nonempty_str(gf):
        errors.append(f"{pid}.group_flag is required")
    elif gf not in GROUP_FLAGS:
        errors.append(
            f"{pid}.group_flag={gf!r} must be one of {GROUP_FLAGS}"
        )

    # unsafe_act: legacy form had placeholder "leave empty if none", so an
    # empty/None value is interpreted as "no unsafe act observed" — valid.
    # The new form normalizes empty → "none" at save time so the field is
    # always present in new records, but legacy records with "" are accepted.
    ua = person.get("unsafe_act")
    if ua is not None and not isinstance(ua, str):
        errors.append(f"{pid}.unsafe_act must be a string")

    # NOTE: per-person `confidence` is OPTIONAL (legacy form never captured
    # it; only the overall annotator_confidence slider exists). The new form
    # adds it as an optional slider prefilled from VLM.

    return errors


def _validate_flag_fields(rec: dict) -> list[str]:
    """Required fields when status='flagged'."""
    errors: list[str] = []

    fc = rec.get("flag_category")
    if not _is_nonempty_str(fc):
        errors.append("flag_category is required for flagged records")
    elif fc not in FLAG_CATEGORIES:
        errors.append(
            f"flag_category={fc!r} must be one of {sorted(FLAG_CATEGORIES)}"
        )

    fr = rec.get("flag_reason")
    if not _is_nonempty_str(fr, FLAG_REASON_MIN_LEN):
        errors.append(
            f"flag_reason must be ≥{FLAG_REASON_MIN_LEN} characters"
        )

    return errors


def _validate_discard_fields(rec: dict) -> list[str]:
    """Required fields when status='discarded'."""
    errors: list[str] = []
    dr = rec.get("discard_reason")
    if not _is_nonempty_str(dr):
        errors.append("discard_reason is required for discarded records")
    return errors


def _validate_safety_officer_fields(rec: dict) -> list[str]:
    """Validate a tier_3 (safety officer) submitted record.

    Safety officer records have a different schema: no scene_type, num_workers,
    persons, or annotation_layer. Instead they have a `safety_review` dict with
    overall_compliance, agree_with_vlm, per_rule judgments, etc.
    """
    errors: list[str] = []
    sr = rec.get("safety_review")
    if not isinstance(sr, dict):
        errors.append("safety_review dict is required for tier_3 records")
        return errors

    # overall_compliance
    oc = sr.get("overall_compliance")
    if not _is_nonempty_str(oc):
        errors.append("safety_review.overall_compliance is required")

    # agree_with_vlm
    agree = sr.get("agree_with_vlm")
    if not _is_nonempty_str(agree):
        errors.append("safety_review.agree_with_vlm is required")

    # disagree_reason required when disagreed
    if agree == "Disagree":
        dr = sr.get("disagree_reason", "")
        if not _is_nonempty_str(dr):
            errors.append(
                "safety_review.disagree_reason is required when agree_with_vlm='Disagree'"
            )

    # safety_description required when not compliant
    if oc in ("not_compliant", "cannot_determine"):
        sd = sr.get("safety_description", "")
        if not _is_nonempty_str(sd):
            errors.append(
                "safety_review.safety_description is required when "
                f"overall_compliance='{oc}'"
            )

    return errors


# ---------- Top-level validator ----------


def validate_annotation(rec: dict) -> list[str]:
    """Run the full validator on an annotation record. Returns list of error
    messages. Empty list means valid.

    Behavior depends on record `status`:
    - submitted → full strict validation (all required fields, schema invariant)
    - flagged → only flag_category and flag_reason required
    - skipped → valid (intentional non-annotation)
    - discarded → discard_reason required
    - other → returns ["unknown status"]
    """
    if not isinstance(rec, dict):
        return ["record is not a dict"]

    status = rec.get("status")
    if status == "skipped":
        return []  # intentional skip — always valid
    if status == "discarded":
        return _validate_discard_fields(rec)
    if status == "flagged":
        return _validate_flag_fields(rec)
    if status != "submitted":
        return [f"unknown status {status!r}"]

    # status == "submitted" — branch on annotator_role
    role = rec.get("annotator_role", "")

    # Tier 3 (safety officer) has a completely different record schema:
    # only safety_review dict + basic metadata. No scene_type, num_workers,
    # persons, etc. Validate the safety-specific fields instead.
    if role == "tier_3":
        return _validate_safety_officer_fields(rec)

    # Tier 1 / Tier 2: full strict validation
    errors = validate_scene_fields(rec)

    layer = rec.get("annotation_layer")
    persons = rec.get("persons") or []
    nw = rec.get("num_workers", 0) or 0
    is_bonus = rec.get("bonus_per_person") is True

    # Layer invariants
    if layer == 1:
        if not is_bonus:
            # Strict Layer 1: persons must be []
            if not isinstance(persons, list):
                errors.append("persons must be a list")
            elif len(persons) > 0:
                errors.append(
                    f"Layer 1 records must have persons=[] (got {len(persons)} persons). "
                    f"Use bonus_per_person=True for legacy records with captured per-person data."
                )
        else:
            # Bonus Layer 1: persons may be non-empty; do not strictly validate
            # per-person fields (legacy data, varying completeness)
            if not isinstance(persons, list):
                errors.append("persons must be a list (even for bonus records)")
    elif layer == 2:
        if not isinstance(persons, list):
            errors.append("persons must be a list")
        else:
            try:
                nw_int = int(nw)
            except (TypeError, ValueError):
                nw_int = 0
            if len(persons) != nw_int:
                errors.append(
                    f"Layer 2 records require len(persons)={len(persons)} == "
                    f"num_workers={nw_int}"
                )
            for p_idx, p in enumerate(persons):
                errors.extend(validate_person(p, p_idx))
    else:
        errors.append(
            f"annotation_layer={layer} must be 1 or 2"
        )

    return errors


def is_valid(rec: dict) -> bool:
    """Return True if the record passes validation."""
    return len(validate_annotation(rec)) == 0


# ---------- Categorization (for the audit script) ----------


def categorize_record(rec: dict) -> str:
    """Categorize a record for the audit/repair script.

    Returns one of:
    - 'non_submitted'       — status != 'submitted', not validated for content
    - 'valid'               — passes validate_annotation()
    - 'invalid_recoverable' — Layer 2 with >5 workers + persons populated
                              (the 6 bonus records). Audit script will set
                              annotation_layer=1 and bonus_per_person=True
                              in place; no re-queue needed.
    - 'invalid_requeue'     — fails validation in a way that requires re-annotation
                              (data loss, missing required fields, etc.).
                              Audit script will move to .invalid/ subdir.
    """
    if not isinstance(rec, dict):
        return "invalid_requeue"

    status = rec.get("status")
    if status != "submitted":
        return "non_submitted"

    # Check for the bonus pattern BEFORE running full validation
    # (these would otherwise show as invalid because of the layer mismatch)
    layer = rec.get("annotation_layer")
    persons = rec.get("persons") or []
    nw_raw = rec.get("num_workers", 0) or 0
    try:
        nw = int(nw_raw)
    except (TypeError, ValueError):
        nw = 0

    is_recoverable_bonus = (
        layer == 2
        and nw > 5
        and isinstance(persons, list)
        and len(persons) == nw
        and not rec.get("bonus_per_person")  # not yet normalized
    )

    if is_recoverable_bonus:
        return "invalid_recoverable"

    if is_valid(rec):
        return "valid"

    return "invalid_requeue"
