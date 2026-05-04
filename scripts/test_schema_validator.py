#!/usr/bin/env python3
"""Smoke tests for annotation_tool/schema_validator.py.

Run with:
    python scripts/test_schema_validator.py

Tests against:
- Real fixture files pulled from VPS (covers Layer 1, Layer 2, invalid types)
- Synthetic minimal records (test edge cases without depending on real data)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make annotation_tool importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from annotation_tool.schema_validator import (  # noqa: E402
    categorize_record,
    derive_annotation_layer,
    is_valid,
    num_workers_bounds_for_layer,
    validate_annotation,
    validate_person,
    validate_scene_fields,
)

FIXTURE_DIR = Path("/tmp/schema_test_fixtures")


# ---------- Helpers ----------

PASS = 0
FAIL = 0


def expect(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ---------- Layer derivation tests ----------


def test_derive_annotation_layer():
    section("derive_annotation_layer")
    expect("nw=1 → layer 2", derive_annotation_layer(1) == 2)
    expect("nw=5 → layer 2", derive_annotation_layer(5) == 2)
    expect("nw=6 → layer 1", derive_annotation_layer(6) == 1)
    expect("nw=12 → layer 1", derive_annotation_layer(12) == 1)
    expect("nw=0 → layer 2", derive_annotation_layer(0) == 2)  # edge
    expect("nw=None → layer 2", derive_annotation_layer(None) == 2)


def test_num_workers_bounds():
    section("num_workers_bounds_for_layer")
    expect("layer 1 bounds = (6, 30)", num_workers_bounds_for_layer(1) == (6, 30))
    expect("layer 2 bounds = (1, 5)", num_workers_bounds_for_layer(2) == (1, 5))


# ---------- Synthetic record tests ----------


def make_valid_layer2_record(num_workers: int = 2) -> dict:
    """Construct a synthetic Layer 2 record that passes strict validation."""
    return {
        "clip_id": "clip_synthetic_test_001",
        "annotator_id": "test",
        "annotator_role": "tier_1",
        "annotator_timestamp": "2026-04-09T00:00:00.000000",
        "status": "submitted",
        "source": "human",
        "annotation_layer": 2,
        "scene_type": "MAI",
        "num_workers": num_workers,
        "dominant_actions": ["B1", "F1"],
        "overall_ppe_compliance": "compliant",
        "visibility": "clear",
        "visibility_conditions": ["clear"],
        "visible_equipment": "",
        "persons": [
            {
                "person_id": f"P{i + 1}",
                "position": "center foreground",
                "physical_description": "male worker in blue shirt and white helmet",
                "action_code": "B1",
                "free_text_description": "operating a pneumatic tool on a steel beam",
                "spatial_context": ["ground_level", "near_machinery"],
                "ppe": {
                    "helmet": "worn",
                    "high_vis_vest": "worn",
                    "welding_protection": "not_applicable",
                    "harness": "not_applicable",
                    "safety_shoes": "worn",
                },
                "occlusion_level": "fully_visible",
                "group_flag": "solo",
                "unsafe_act": "none",
                "confidence": 0.9,
            }
            for i in range(num_workers)
        ],
        "action_transition": {"detected": False, "transitions": []},
        "group_activity": "",
        "scene_description": "Workers operating pneumatic tools on steel beams in workshop",
        "annotator_confidence": 0.85,
    }


def make_valid_layer1_record(num_workers: int = 8) -> dict:
    """Construct a synthetic Layer 1 record that passes strict validation."""
    return {
        "clip_id": "clip_synthetic_test_002",
        "annotator_id": "test",
        "annotator_role": "tier_1",
        "annotator_timestamp": "2026-04-09T00:00:00.000000",
        "status": "submitted",
        "source": "human",
        "annotation_layer": 1,
        "scene_type": "MAC",
        "num_workers": num_workers,
        "dominant_actions": ["B1", "F1", "A1"],
        "overall_ppe_compliance": "partial",
        "visibility": "clear",
        "visibility_conditions": ["clear"],
        "visible_equipment": "rotary feeder",
        "persons": [],  # Layer 1 — no per-person data
        "action_transition": {"detected": False, "transitions": []},
        "group_activity": "Team maintenance work",
        "scene_description": "Eight workers gathered around a large rotary feeder for maintenance work",
        "annotator_confidence": 0.85,
    }


def test_synthetic_valid_records():
    section("synthetic valid records")
    rec_l2 = make_valid_layer2_record()
    errs = validate_annotation(rec_l2)
    expect(
        "synthetic Layer 2 (2 workers) passes",
        len(errs) == 0,
        detail=f"errors: {errs}",
    )
    expect("synthetic Layer 2 → categorize='valid'", categorize_record(rec_l2) == "valid")

    rec_l1 = make_valid_layer1_record()
    errs = validate_annotation(rec_l1)
    expect(
        "synthetic Layer 1 (8 workers) passes",
        len(errs) == 0,
        detail=f"errors: {errs}",
    )
    expect("synthetic Layer 1 → categorize='valid'", categorize_record(rec_l1) == "valid")


def test_synthetic_invalid_records():
    section("synthetic invalid records")

    # Layer 1 with ≤5 workers (the bug pattern)
    bad = make_valid_layer1_record()
    bad["num_workers"] = 3
    errs = validate_annotation(bad)
    expect(
        "Layer 1 with num_workers=3 → invalid",
        len(errs) > 0,
        detail=f"errors: {errs}",
    )
    expect(
        "Layer 1 with num_workers=3 → categorize='invalid_requeue'",
        categorize_record(bad) == "invalid_requeue",
    )

    # Layer 2 with persons=[]
    bad = make_valid_layer2_record(num_workers=3)
    bad["persons"] = []
    errs = validate_annotation(bad)
    expect(
        "Layer 2 with persons=[] → invalid",
        len(errs) > 0,
        detail=f"errors: {errs}",
    )

    # Layer 2 with mismatched persons count
    bad = make_valid_layer2_record(num_workers=3)
    bad["num_workers"] = 5  # but persons has 3
    errs = validate_annotation(bad)
    expect(
        "Layer 2 with mismatched persons count → invalid",
        len(errs) > 0,
        detail=f"errors: {errs}",
    )

    # Missing scene_type
    bad = make_valid_layer2_record()
    bad["scene_type"] = ""
    errs = validate_annotation(bad)
    expect(
        "Missing scene_type → invalid",
        any("scene_type" in e for e in errs),
        detail=f"errors: {errs}",
    )

    # Missing scene_description
    bad = make_valid_layer2_record()
    bad["scene_description"] = "short"  # too short
    errs = validate_annotation(bad)
    expect(
        "Short scene_description → invalid",
        any("scene_description" in e for e in errs),
        detail=f"errors: {errs}",
    )

    # physical_description, free_text_description, per-person confidence are
    # NOT required by the validator (legacy form never captured them). They
    # are OPTIONAL fields prefilled from VLM in the new form. Verify the
    # validator does NOT flag them as missing.
    bad = make_valid_layer2_record()
    bad["persons"][0]["physical_description"] = None
    bad["persons"][0]["free_text_description"] = None
    bad["persons"][0]["confidence"] = None
    expect(
        "Missing P1 physical_description/free_text/confidence → still valid (optional)",
        is_valid(bad),
        detail=f"errors: {validate_annotation(bad)}",
    )

    # Missing P1 action_code (still required)
    bad = make_valid_layer2_record()
    bad["persons"][0]["action_code"] = ""
    errs = validate_annotation(bad)
    expect(
        "Missing P1 action_code → invalid",
        any("action_code" in e for e in errs),
        detail=f"errors: {errs}",
    )

    # Bonus per-person should be VALID
    bonus = make_valid_layer1_record(num_workers=8)
    bonus["persons"] = [{"person_id": "P1", "action_code": "B1"}]  # legacy partial data
    bonus["bonus_per_person"] = True
    errs = validate_annotation(bonus)
    expect(
        "Bonus per-person Layer 1 → valid (legacy)",
        len(errs) == 0,
        detail=f"errors: {errs}",
    )
    expect(
        "Bonus per-person → categorize='valid'",
        categorize_record(bonus) == "valid",
    )


def test_synthetic_status_handling():
    section("non-submitted statuses")

    skipped = {"status": "skipped", "clip_id": "x"}
    expect("skipped → valid", is_valid(skipped))
    expect("skipped → categorize='non_submitted'", categorize_record(skipped) == "non_submitted")

    discarded = {"status": "discarded", "discard_reason": "no_persons_visible", "clip_id": "x"}
    expect("discarded with reason → valid", is_valid(discarded))

    discarded_no_reason = {"status": "discarded", "clip_id": "x"}
    errs = validate_annotation(discarded_no_reason)
    expect(
        "discarded without reason → invalid",
        any("discard_reason" in e for e in errs),
        detail=f"errors: {errs}",
    )

    flagged = {
        "status": "flagged",
        "clip_id": "x",
        "flag_category": "vlm_count_wrong",
        "flag_reason": "VLM said 8 but actually 3 workers visible",
    }
    expect("flagged with category+reason → valid", is_valid(flagged))

    flagged_bad_cat = {
        "status": "flagged",
        "clip_id": "x",
        "flag_category": "not_a_valid_category",
        "flag_reason": "blah blah",
    }
    errs = validate_annotation(flagged_bad_cat)
    expect(
        "flagged with bad category → invalid",
        any("flag_category" in e for e in errs),
        detail=f"errors: {errs}",
    )


# ---------- Real fixture tests ----------


def load_fixture(name: str) -> dict:
    p = FIXTURE_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def test_real_fixtures():
    section("real fixtures from VPS")

    if not FIXTURE_DIR.exists():
        print(f"  SKIP  fixtures dir not found: {FIXTURE_DIR}")
        return

    # invalid_layer1_with_few_workers — must categorize as invalid_requeue
    rec = load_fixture("clip_CRM_Work_Shop_20251115_000945_0039.json")
    if rec:
        cat = categorize_record(rec)
        expect(
            "fixture invalid_layer1_with_few_workers → invalid_requeue",
            cat == "invalid_requeue",
            detail=f"got {cat}",
        )

    # invalid_layer2_with_many_workers — must categorize as invalid_recoverable (the bonus pattern)
    rec = load_fixture("clip_RERS_RERS-1_20251212_000615_0025.json")
    if rec:
        cat = categorize_record(rec)
        expect(
            "fixture invalid_layer2_with_many_workers → invalid_recoverable",
            cat == "invalid_recoverable",
            detail=f"got {cat}",
        )

    # missing_scene_type — must categorize as invalid_requeue
    rec = load_fixture("clip_RED_LRS-1_20251111_000145_0007.json")
    if rec:
        cat = categorize_record(rec)
        expect(
            "fixture missing_scene_type → invalid_requeue",
            cat == "invalid_requeue",
            detail=f"got {cat}",
        )

    # missing_scene_description_layer1 — must categorize as invalid_requeue
    rec = load_fixture("clip_CRM_MOTER_STORE_20260403_000530_0022.json")
    if rec:
        cat = categorize_record(rec)
        expect(
            "fixture missing_scene_description_layer1 → invalid_requeue",
            cat == "invalid_requeue",
            detail=f"got {cat}",
        )


# ---------- Run all ----------


def main():
    test_derive_annotation_layer()
    test_num_workers_bounds()
    test_synthetic_valid_records()
    test_synthetic_invalid_records()
    test_synthetic_status_handling()
    test_real_fixtures()

    print()
    print(f"=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
