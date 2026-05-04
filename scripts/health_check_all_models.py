#!/usr/bin/env python3
"""Comprehensive health check for ALL model inference results.

Validates every successful result across all models against the expected schema.
Reports issues per model, per field, and generates a re-run list for failed clips.

Usage:
    python scripts/health_check_all_models.py
    python scripts/health_check_all_models.py --model gpt4o
    python scripts/health_check_all_models.py --rerun-list  # generates rerun JSONs

Run AFTER inference completes, BEFORE metrics computation.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = Path(__file__).parent.parent / "eval_data" / "results"
GT_CLIPS_FILE = Path(__file__).parent.parent / "eval_data" / "gt_clips.json"

VALID_ACTIONS = {
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
    "C1", "C2", "C3", "C4",
    "D1", "D2", "D3",
    "E1", "E2",
    "F1", "F2",
    "X1",
}
VALID_PPE_STATUS = {"worn", "not_worn", "cannot_determine", "not_applicable"}
VALID_SCENE_TYPES = {"SA", "MAI", "MAC", "SEQ"}
VALID_OCCLUSION = {"fully_visible", "partially_occluded", "heavily_occluded"}
PPE_ITEMS = ["helmet", "safety_shoes", "high_vis_vest", "welding_protection", "harness"]

SKIP_PATTERNS = ["_test", "_batch_input", "_batch_error", "_chunk", "_batch_v2"]


def load_gt_clips():
    """Load GT clip IDs."""
    if GT_CLIPS_FILE.exists():
        return set(json.load(open(GT_CLIPS_FILE)))
    return set()


def validate_record(record):
    """Validate a single successful inference record. Returns list of issues."""
    issues = []
    clip_id = record.get("clip_id", "?")
    norm = record.get("normalized", {})
    parsed = record.get("parsed", {})

    if not norm:
        issues.append("no_normalized_output")
        return issues
    if not parsed:
        issues.append("no_parsed_output")

    # Scene type
    st = norm.get("scene_type", "")
    if st not in VALID_SCENE_TYPES:
        issues.append(f"invalid_scene_type={st}")

    # Num workers
    nw = norm.get("num_workers", 0)
    if nw == 0:
        issues.append("num_workers=0")
    elif nw < 0 or nw > 30:
        issues.append(f"num_workers_out_of_range={nw}")

    # Annotation layer
    layer = norm.get("annotation_layer")
    if layer not in (1, 2):
        issues.append(f"invalid_layer={layer}")

    # Dominant actions
    da = norm.get("dominant_actions", [])
    if not da:
        issues.append("empty_dominant_actions")
    for a in da:
        if a not in VALID_ACTIONS:
            issues.append(f"invalid_dominant_action={a}")

    # Visibility conditions
    vis = norm.get("visibility_conditions", [])
    if not vis:
        issues.append("missing_visibility_conditions")

    # Overall PPE compliance
    opc = norm.get("overall_ppe_compliance", "")
    if not opc:
        issues.append("missing_overall_ppe_compliance")

    # Annotator confidence
    ac = norm.get("annotator_confidence")
    if ac is None:
        issues.append("missing_annotator_confidence")
    elif not (0 <= ac <= 1):
        issues.append(f"confidence_out_of_range={ac}")

    # Persons (Layer 2)
    persons = norm.get("persons", [])
    if layer == 2 and not persons:
        issues.append("layer2_but_no_persons")

    for i, p in enumerate(persons):
        prefix = f"p[{i}]"

        # Action code
        ac = p.get("action_code", "")
        if not ac:
            issues.append(f"{prefix}_missing_action_code")
        elif ac not in VALID_ACTIONS:
            issues.append(f"{prefix}_invalid_action={ac}")

        # Confidence
        conf = p.get("confidence")
        if conf is None:
            issues.append(f"{prefix}_missing_confidence")
        elif not (0 <= conf <= 1):
            issues.append(f"{prefix}_confidence_out_of_range={conf}")

        # PPE
        ppe = p.get("ppe", {})
        if not ppe:
            issues.append(f"{prefix}_missing_ppe")
        else:
            for item in ["helmet", "safety_shoes", "high_vis_vest"]:
                val = ppe.get(item, "")
                if val and val not in VALID_PPE_STATUS:
                    issues.append(f"{prefix}_invalid_ppe_{item}={val}")

        # Spatial context
        sc = p.get("spatial_context", [])
        if not sc:
            issues.append(f"{prefix}_missing_spatial_context")

        # Occlusion
        occ = p.get("occlusion_level", "")
        if occ and occ not in VALID_OCCLUSION:
            issues.append(f"{prefix}_invalid_occlusion={occ}")

        # Unsafe act (should exist as field, can be "none" or empty)
        if "unsafe_act" not in p:
            issues.append(f"{prefix}_missing_unsafe_act_field")

    return issues


def check_model(model_name, gt_clips=None):
    """Run health check on a single model's results."""
    result_file = RESULTS_DIR / f"{model_name}.jsonl"
    if not result_file.exists():
        return None

    records = []
    for line in open(result_file, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    total = len(records)
    successful = [r for r in records if r.get("success")]
    failed = [r for r in records if not r.get("success")]

    # Deduplicate by clip_id (keep last successful)
    unique_success = {}
    for r in successful:
        unique_success[r.get("clip_id", "")] = r

    # Validate each successful record
    clip_issues = {}  # clip_id -> list of issues
    clean_clips = set()
    issue_type_counts = Counter()

    for clip_id, r in unique_success.items():
        issues = validate_record(r)
        if issues:
            clip_issues[clip_id] = issues
            for iss in issues:
                # Extract issue type (before = sign)
                issue_type = iss.split("=")[0].split("_")[-1] if "=" in iss else iss
                issue_type_counts[iss.split("=")[0]] += 1
        else:
            clean_clips.add(clip_id)

    # Coverage check against GT
    coverage = {}
    if gt_clips:
        covered = set(unique_success.keys()) & gt_clips
        missing = gt_clips - set(unique_success.keys())
        coverage = {
            "gt_total": len(gt_clips),
            "covered": len(covered),
            "missing": len(missing),
            "coverage_pct": len(covered) / len(gt_clips) * 100 if gt_clips else 0,
        }

    # Failure analysis
    failure_types = Counter()
    for r in failed:
        err = r.get("error", r.get("raw_response", "unknown"))
        if isinstance(err, str):
            if "parse" in err.lower():
                failure_types["parse_fail"] += 1
            elif "429" in err or "rate" in err.lower() or "throttl" in err.lower():
                failure_types["rate_limit"] += 1
            elif "timeout" in err.lower():
                failure_types["timeout"] += 1
            elif "safety" in err.lower() or "block" in err.lower():
                failure_types["safety_block"] += 1
            elif "token" in err.lower() or "length" in err.lower():
                failure_types["token_limit"] += 1
            else:
                failure_types["other"] += 1
        else:
            failure_types["unknown"] += 1

    # Stats from successful records
    confs = []
    workers_list = []
    for r in unique_success.values():
        n = r.get("normalized", {})
        workers_list.append(n.get("num_workers", 0))
        for p in n.get("persons", []):
            c = p.get("confidence")
            if c is not None:
                confs.append(c)

    return {
        "model": model_name,
        "total_records": total,
        "unique_successful": len(unique_success),
        "total_failed": len(failed),
        "clean_clips": len(clean_clips),
        "clips_with_issues": len(clip_issues),
        "issue_details": clip_issues,
        "issue_type_counts": dict(issue_type_counts),
        "failure_types": dict(failure_types),
        "coverage": coverage,
        "stats": {
            "avg_workers": sum(workers_list) / len(workers_list) if workers_list else 0,
            "avg_confidence": sum(confs) / len(confs) if confs else 0,
            "confidence_range": [min(confs), max(confs)] if confs else [0, 0],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Health check all model inference results")
    parser.add_argument("--model", default=None, help="Check specific model only")
    parser.add_argument("--rerun-list", action="store_true",
                        help="Generate rerun clip lists for failed/missing clips")
    parser.add_argument("--verbose", action="store_true", help="Show per-clip issues")
    args = parser.parse_args()

    gt_clips = load_gt_clips()
    print(f"GT clips: {len(gt_clips)}")

    # Find all model result files
    if args.model:
        model_names = [args.model]
    else:
        model_names = []
        for f in sorted(RESULTS_DIR.glob("*.jsonl")):
            name = f.stem
            if any(p in name for p in SKIP_PATTERNS):
                continue
            model_names.append(name)

    print(f"Models to check: {len(model_names)}")
    print("=" * 90)

    all_results = {}
    for model_name in model_names:
        result = check_model(model_name, gt_clips)
        if result is None:
            print(f"\n{model_name}: FILE NOT FOUND")
            continue

        all_results[model_name] = result
        r = result

        # Print summary
        clean_pct = r["clean_clips"] / r["unique_successful"] * 100 if r["unique_successful"] else 0
        coverage_str = ""
        if r["coverage"]:
            coverage_str = f"  coverage={r['coverage']['covered']}/{r['coverage']['gt_total']} ({r['coverage']['coverage_pct']:.1f}%)"

        verdict = "PASS" if r["clips_with_issues"] == 0 and clean_pct == 100 else "ISSUES"
        if r["unique_successful"] == 0:
            verdict = "NO DATA"

        print(f"\n{model_name}:")
        print(f"  Records: {r['total_records']} total, {r['unique_successful']} unique successful, {r['total_failed']} failed")
        print(f"  Clean: {r['clean_clips']}/{r['unique_successful']} ({clean_pct:.1f}%){coverage_str}")
        print(f"  Stats: avg_workers={r['stats']['avg_workers']:.1f}  avg_conf={r['stats']['avg_confidence']:.3f}  conf_range={r['stats']['confidence_range']}")

        if r["clips_with_issues"]:
            print(f"  Issues ({r['clips_with_issues']} clips):")
            for issue_type, count in sorted(r["issue_type_counts"].items(), key=lambda x: -x[1])[:10]:
                print(f"    {issue_type}: {count}")

        if r["failure_types"]:
            print(f"  Failures ({r['total_failed']} records):")
            for ftype, count in sorted(r["failure_types"].items(), key=lambda x: -x[1]):
                print(f"    {ftype}: {count}")

        if r["coverage"] and r["coverage"]["missing"] > 0:
            print(f"  MISSING: {r['coverage']['missing']} GT clips not covered")

        print(f"  Verdict: {verdict}")

        if args.verbose and r["issue_details"]:
            print(f"  Per-clip issues (first 5):")
            for clip_id, issues in list(r["issue_details"].items())[:5]:
                print(f"    {clip_id[:45]}: {issues}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Model':<25} {'Success':>8} {'Clean':>8} {'Issues':>8} {'Failed':>8} {'Missing':>8} {'Verdict':>8}")
    print("-" * 90)

    for model_name, r in sorted(all_results.items()):
        missing = r["coverage"].get("missing", 0) if r["coverage"] else 0
        verdict = "PASS" if r["clips_with_issues"] == 0 and missing == 0 else "CHECK"
        if r["unique_successful"] == 0:
            verdict = "NO DATA"
        print(f"{model_name:<25} {r['unique_successful']:>8} {r['clean_clips']:>8} {r['clips_with_issues']:>8} {r['total_failed']:>8} {missing:>8} {verdict:>8}")

    # Generate rerun lists
    if args.rerun_list:
        print("\n" + "=" * 90)
        print("RERUN LISTS")
        print("=" * 90)

        for model_name, r in all_results.items():
            # Clips to rerun = failed + missing + clips with critical issues
            rerun_clips = set()

            # Missing from GT
            if r["coverage"]:
                covered = set()
                result_file = RESULTS_DIR / f"{model_name}.jsonl"
                for line in open(result_file, encoding="utf-8", errors="replace"):
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("success") and rec.get("clip_id"):
                            covered.add(rec["clip_id"])
                    except:
                        pass
                rerun_clips |= (gt_clips - covered)

            # Clips with critical issues (no normalized, no action codes)
            for clip_id, issues in r.get("issue_details", {}).items():
                critical = any(
                    iss in ["no_normalized_output", "no_parsed_output", "num_workers=0",
                            "empty_dominant_actions", "layer2_but_no_persons"]
                    or iss.startswith("p[") and "missing_action_code" in iss
                    for iss in issues
                )
                if critical:
                    rerun_clips.add(clip_id)

            if rerun_clips:
                rerun_path = RESULTS_DIR / f"{model_name}_rerun.json"
                with open(rerun_path, "w") as f:
                    json.dump(sorted(rerun_clips), f, indent=2)
                print(f"  {model_name}: {len(rerun_clips)} clips to rerun -> {rerun_path}")
            else:
                print(f"  {model_name}: no rerun needed")


if __name__ == "__main__":
    main()
