#!/usr/bin/env python3
"""Compute evaluation metrics for SteelBench VLM predictions against GT.

Standard metrics:
- Top-1 accuracy (weighted + macro), per-class accuracy
- ECE (expected calibration error)
- F1-magnet analysis
- Worker count MAE

Novel metrics (M1-M4, M6-M7):
- M1: AUDC/nAUDC (degradation robustness — accuracy across visibility bins)
- M2: CRG (compositional reasoning gap — safety reasoning when perception correct)
- M3: SR@K (safety-weighted recall stratified by severity tier)
- M4: DWA (distance-weighted accuracy by taxonomic distance)
- M6: F2-detect (worker detection F-beta with beta=2)
- M7: DRS (deployment readiness score — composite pass/fail)

Usage:
    python eval_metrics_compute.py
    python eval_metrics_compute.py --model qwen35_122b
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from annotation_tool.agreement import (
    ACTION_GROUPS,
    ACTION_TAXONOMY,
    CODE_TO_GROUP,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent / "eval_data"
RESULTS_DIR = EVAL_DIR / "results"
GT_DIR = EVAL_DIR / "gt"
CONFIG_DIR = EVAL_DIR / "config"





# ---------------------------------------------------------------------------
# Taxonomic distance for DWA (M4)
# ---------------------------------------------------------------------------
def taxonomic_distance(code_a, code_b):
    """Compute taxonomic distance between two action codes."""
    if code_a == code_b:
        return 0.0
    group_a = CODE_TO_GROUP.get(code_a, "X")
    group_b = CODE_TO_GROUP.get(code_b, "X")
    if group_a == group_b:
        return 0.33
    else:
        group_order = ["A", "B", "C", "D", "E", "F", "X"]
        try:
            idx_a = group_order.index(group_a)
            idx_b = group_order.index(group_b)
            dist = abs(idx_a - idx_b)
            return min(0.50 + dist * 0.10, 1.0)
        except ValueError:
            return 1.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_predictions(model_name):
    """Load model predictions from JSONL."""
    path = RESULTS_DIR / f"{model_name}.jsonl"
    predictions = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                clip_id = record.get("clip_id", "")
                if clip_id and record.get("success"):
                    predictions[clip_id] = record
            except json.JSONDecodeError:
                pass
    return predictions


def load_gt_annotations():
    """Load GT annotations. Priority: expert > tier_1."""
    gt = {}
    for expert_dir in ["expert_1", "expert_2"]:
        expert_path = GT_DIR / expert_dir
        if not expert_path.exists():
            continue
        for f in sorted(expert_path.glob("*.json")):
            try:
                data = json.load(open(f))
                if data.get("status") != "submitted":
                    continue
                clip_id = data.get("clip_id", "")
                if clip_id:
                    gt[clip_id] = data
            except (json.JSONDecodeError, KeyError):
                continue

    for annotator_dir in ["annotator_1", "annotator_2", "annotator_3", "annotator_4"]:
        ann_path = GT_DIR / annotator_dir
        if not ann_path.exists():
            continue
        for f in sorted(ann_path.glob("*.json")):
            try:
                data = json.load(open(f))
                if data.get("status") != "submitted":
                    continue
                clip_id = data.get("clip_id", "")
                if clip_id and clip_id not in gt:
                    gt[clip_id] = data
            except (json.JSONDecodeError, KeyError):
                continue
    return gt


def load_manifest():
    """Load batch manifest for visibility/site info."""
    manifest = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row.get("clip_id", "")] = row
    return manifest


def extract_person_pairs(pred_record, gt_record):
    """Extract matched person-level pairs."""
    pairs = []
    pred_norm = pred_record.get("normalized", {})
    gt_persons = gt_record.get("persons", [])
    pred_persons = pred_norm.get("persons", [])
    gt_layer = gt_record.get("annotation_layer", 2)

    if gt_layer == 2 and gt_persons:
        for i, gt_p in enumerate(gt_persons):
            gt_action = gt_p.get("action_code", "")
            if not gt_action:
                continue
            if i < len(pred_persons):
                pred_action = pred_persons[i].get("action_code", "")
                conf = pred_persons[i].get("confidence", 0.5)
                pred_unsafe = pred_persons[i].get("unsafe_act", "none")
            else:
                pred_action = ""
                conf = 0.5
                pred_unsafe = "none"

            if pred_action:
                pairs.append({
                    "gt_action": gt_action,
                    "pred_action": pred_action,
                    "confidence": conf,
                    "gt_unsafe": gt_p.get("unsafe_act", "none") or "none",
                    "pred_unsafe": pred_unsafe or "none",
                })
    elif gt_layer == 1:
        gt_actions = gt_record.get("dominant_actions", [])
        pred_actions = pred_norm.get("dominant_actions", [])
        if gt_actions and pred_actions:
            pairs.append({
                "gt_action": gt_actions[0],
                "pred_action": pred_actions[0],
                "confidence": pred_norm.get("annotator_confidence", 0.5),
                "gt_unsafe": gt_record.get("scene_unsafe_act", "none") or "none",
                "pred_unsafe": pred_norm.get("unsafe_act", "none") or "none",
            })
    return pairs


# ---------------------------------------------------------------------------
# Novel Metrics
# ---------------------------------------------------------------------------
def compute_audc(clip_data, visibility_bins=None):
    """M1: Area Under Degradation Curve.

    Computes accuracy at each visibility condition level,
    then AUDC = mean accuracy across bins.
    nAUDC = AUDC / best_bin_accuracy (normalized).
    """
    if visibility_bins is None:
        visibility_bins = ["clear", "low_light", "dust", "glare", "steam", "smoke"]

    bin_correct = defaultdict(int)
    bin_total = defaultdict(int)

    for item in clip_data:
        vis_conditions = item.get("visibility_conditions", ["clear"])
        correct = item.get("correct", False)
        for vc in vis_conditions:
            if vc in visibility_bins:
                bin_total[vc] += 1
                if correct:
                    bin_correct[vc] += 1

    bin_acc = {}
    for vc in visibility_bins:
        if bin_total[vc] > 0:
            bin_acc[vc] = bin_correct[vc] / bin_total[vc]

    if not bin_acc:
        return {"audc": None, "naudc": None, "per_bin": {}}

    audc = sum(bin_acc.values()) / len(bin_acc)
    best = max(bin_acc.values())
    naudc = audc / best if best > 0 else 0

    return {
        "audc": round(audc, 4),
        "naudc": round(naudc, 4),
        "per_bin": {k: round(v, 4) for k, v in bin_acc.items()},
    }


def compute_crg(pairs):
    """M2: Compositional Reasoning Gap.

    CRG = 1 - (correctly_perceived AND correctly_judged / correctly_perceived)
    When the model gets the action right, does it also get the safety judgment right?
    """
    # Correctly perceived = action prediction matches GT
    correct_perception = [p for p in pairs if p["gt_action"] == p["pred_action"]]
    if not correct_perception:
        return {"crg": None, "n_correct_perception": 0, "n_correct_both": 0}

    # Of those, how many also got safety right?
    correct_both = 0
    for p in correct_perception:
        gt_has_violation = bool(p["gt_unsafe"] != "none" and p["gt_unsafe"])
        pred_has_violation = bool(p["pred_unsafe"] != "none" and p["pred_unsafe"])
        # Both agree on violation status (binary: violation present or not)
        if gt_has_violation == pred_has_violation:
            correct_both += 1

    crg = 1.0 - (correct_both / len(correct_perception))
    return {
        "crg": round(crg, 4),
        "n_correct_perception": len(correct_perception),
        "n_correct_both": correct_both,
    }


def compute_safety_recall(pairs):
    """M3: Safety Recall — does the model detect violations?

    All violations are equally important (no severity tiers).
    Returns: recall, false-safe rate, false-alarm rate, N.
    """
    gt_violations = [p for p in pairs if p["gt_unsafe"] != "none" and p["gt_unsafe"]]
    gt_safe = [p for p in pairs if p["gt_unsafe"] == "none" or not p["gt_unsafe"]]

    n_violations = len(gt_violations)
    if n_violations > 0:
        tp = sum(1 for p in gt_violations if p["pred_unsafe"] != "none" and p["pred_unsafe"])
        recall = tp / n_violations
        false_safe_rate = 1.0 - recall
    else:
        recall = None
        false_safe_rate = None

    if gt_safe:
        false_alarms = sum(1 for p in gt_safe if p["pred_unsafe"] != "none" and p["pred_unsafe"])
        false_alarm_rate = false_alarms / len(gt_safe)
    else:
        false_alarm_rate = None

    return {
        "recall": round(recall, 4) if recall is not None else None,
        "false_safe_rate": round(false_safe_rate, 4) if false_safe_rate is not None else None,
        "false_alarm_rate": round(false_alarm_rate, 4) if false_alarm_rate is not None else None,
        "n_violations": n_violations,
        "n_safe": len(gt_safe),
    }


def compute_f2_detect(clip_results):
    """M6: F2-detect — worker detection with beta=2 (recall-weighted).

    Compares predicted worker count vs GT worker count per clip.
    TP = min(pred, gt), FP = max(0, pred - gt), FN = max(0, gt - pred)
    F2 = (1 + 4) * precision * recall / (4 * precision + recall)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in clip_results:
        pred_count = item.get("pred_count", 0)
        gt_count = item.get("gt_count", 0)
        if gt_count == 0:
            continue
        tp = min(pred_count, gt_count)
        fp = max(0, pred_count - gt_count)
        fn = max(0, gt_count - pred_count)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    beta = 2
    if precision + recall > 0:
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        f2 = 0

    return {
        "f2_detect": round(f2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "worker_count_mae": round(
            sum(abs(i["pred_count"] - i["gt_count"]) for i in clip_results if i["gt_count"] > 0)
            / max(sum(1 for i in clip_results if i["gt_count"] > 0), 1), 3),
    }


def compute_drs(metrics):
    """M7: Deployment Readiness Score — composite of 5 threshold checks.

    DRS = passes / 5
    Thresholds:
    - DWA >= 0.80
    - nAUDC >= 0.85
    - CRG <= 0.20
    - SR@critical >= 0.90
    - F2-detect >= 0.70
    """
    checks = {}

    # DWA
    dwa = metrics.get("dwa", 0)
    checks["dwa"] = {"value": dwa, "threshold": 0.80, "pass": dwa >= 0.80}

    # nAUDC
    naudc = metrics.get("audc", {}).get("naudc")
    if naudc is not None:
        checks["naudc"] = {"value": naudc, "threshold": 0.85, "pass": naudc >= 0.85}
    else:
        checks["naudc"] = {"value": None, "threshold": 0.85, "pass": False}

    # CRG
    crg = metrics.get("crg", {}).get("crg")
    if crg is not None:
        checks["crg"] = {"value": crg, "threshold": 0.20, "pass": crg <= 0.20}
    else:
        checks["crg"] = {"value": None, "threshold": 0.20, "pass": False}

    # Safety Recall
    sr = metrics.get("safety_recall", {}).get("recall")
    if sr is not None:
        checks["safety_recall"] = {"value": sr, "threshold": 0.90, "pass": sr >= 0.90}
    else:
        checks["safety_recall"] = {"value": None, "threshold": 0.90, "pass": False}

    # F2-detect
    f2 = metrics.get("f2_detect", {}).get("f2_detect", 0)
    checks["f2_detect"] = {"value": f2, "threshold": 0.70, "pass": f2 >= 0.70}

    passes = sum(1 for c in checks.values() if c["pass"])
    return {
        "drs": round(passes / 5, 2),
        "passes": passes,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
def compute_metrics(predictions, gt, manifest=None):
    """Compute all standard + novel metrics."""
    all_pairs = []
    clip_results = []
    clip_data_for_audc = []

    matched_clips = set(predictions.keys()) & set(gt.keys())
    if not matched_clips:
        return {}

    for clip_id in sorted(matched_clips):
        pred = predictions[clip_id]
        gt_ann = gt[clip_id]

        pairs = extract_person_pairs(pred, gt_ann)
        all_pairs.extend(pairs)

        # Worker count
        pred_count = pred.get("normalized", {}).get("num_workers", 0)
        gt_count = gt_ann.get("num_workers", len(gt_ann.get("persons", [])))

        clip_results.append({
            "clip_id": clip_id,
            "n_pairs": len(pairs),
            "pred_count": pred_count,
            "gt_count": gt_count,
        })

        # Visibility for AUDC
        vis = gt_ann.get("visibility_conditions", [])
        if isinstance(vis, str):
            vis = [vis]
        for p in pairs:
            clip_data_for_audc.append({
                "visibility_conditions": vis,
                "correct": p["gt_action"] == p["pred_action"],
            })

    if not all_pairs:
        return {}

    total = len(all_pairs)

    # ---- Standard: Accuracy ----
    correct = sum(1 for p in all_pairs if p["gt_action"] == p["pred_action"])
    weighted_acc = correct / total

    class_correct = Counter()
    class_total = Counter()
    for p in all_pairs:
        class_total[p["gt_action"]] += 1
        if p["gt_action"] == p["pred_action"]:
            class_correct[p["gt_action"]] += 1

    per_class_acc = {}
    for cls in sorted(class_total.keys()):
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        per_class_acc[cls] = {"accuracy": acc, "correct": class_correct[cls],
                              "total": class_total[cls]}

    macro_acc = sum(v["accuracy"] for v in per_class_acc.values()) / len(per_class_acc) if per_class_acc else 0

    # ---- Standard: ECE ----
    n_bins = 10
    bin_correct = [0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_count = [0] * n_bins
    for p in all_pairs:
        conf = p["confidence"]
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_count[bin_idx] += 1
        bin_conf[bin_idx] += conf
        if p["gt_action"] == p["pred_action"]:
            bin_correct[bin_idx] += 1

    ece = sum(
        (bin_count[i] / total) * abs(bin_conf[i] / bin_count[i] - bin_correct[i] / bin_count[i])
        for i in range(n_bins) if bin_count[i] > 0
    )
    avg_confidence = sum(p["confidence"] for p in all_pairs) / total

    # ---- Standard: F1-Magnet ----
    wrong_preds = [p for p in all_pairs if p["gt_action"] != p["pred_action"]]
    wrong_pred_counts = Counter(p["pred_action"] for p in wrong_preds)
    f1_as_wrong = wrong_pred_counts.get("F1", 0)
    f1_magnet_ratio = f1_as_wrong / len(wrong_preds) if wrong_preds else 0
    confusion_pairs = Counter((p["gt_action"], p["pred_action"]) for p in wrong_preds)

    # ---- Standard: Group-level ----
    group_correct = Counter()
    group_total = Counter()
    for p in all_pairs:
        gt_group = CODE_TO_GROUP.get(p["gt_action"], "X")
        pred_group = CODE_TO_GROUP.get(p["pred_action"], "X")
        group_total[gt_group] += 1
        if gt_group == pred_group:
            group_correct[gt_group] += 1
    group_acc = {g: {"accuracy": group_correct[g] / group_total[g],
                     "correct": group_correct[g], "total": group_total[g]}
                 for g in sorted(group_total.keys())}

    # ---- Standard: Safety basic ----
    safety_pairs = [p for p in all_pairs if p["gt_unsafe"] != "none" and p["gt_unsafe"]]
    safe_pairs = [p for p in all_pairs if p["gt_unsafe"] == "none" or not p["gt_unsafe"]]
    n_violations = len(safety_pairs)
    if n_violations > 0:
        tp = sum(1 for p in safety_pairs if p["pred_unsafe"] != "none" and p["pred_unsafe"])
        false_safe_rate = (n_violations - tp) / n_violations
        safety_recall = tp / n_violations
    else:
        false_safe_rate = None
        safety_recall = None
    if safe_pairs:
        false_alarms = sum(1 for p in safe_pairs if p["pred_unsafe"] != "none" and p["pred_unsafe"])
        false_alarm_rate = false_alarms / len(safe_pairs)
    else:
        false_alarm_rate = None

    # ---- M1: AUDC ----
    audc_result = compute_audc(clip_data_for_audc)

    # ---- M2: CRG ----
    crg_result = compute_crg(all_pairs)

    # ---- M3: Safety Recall ----
    safety_recall_result = compute_safety_recall(all_pairs)

    # ---- M4: DWA ----
    total_distance = sum(taxonomic_distance(p["gt_action"], p["pred_action"]) for p in all_pairs)
    dwa = 1.0 - (total_distance / total)

    # ---- M6: F2-detect ----
    f2_result = compute_f2_detect(clip_results)

    # Build metrics dict
    metrics = {
        "n_clips": len(matched_clips),
        "n_pairs": total,
        # Standard
        "weighted_accuracy": round(weighted_acc, 4),
        "macro_accuracy": round(macro_acc, 4),
        "ece": round(ece, 4),
        "avg_confidence": round(avg_confidence, 4),
        "confidence_accuracy_gap": round(avg_confidence - weighted_acc, 4),
        "f1_magnet": {
            "f1_as_wrong_pred": f1_as_wrong,
            "total_wrong": len(wrong_preds),
            "ratio": round(f1_magnet_ratio, 4),
        },
        "safety": {
            "n_violations_in_gt": n_violations,
            "recall": round(safety_recall, 4) if safety_recall is not None else None,
            "false_safe_rate": round(false_safe_rate, 4) if false_safe_rate is not None else None,
            "false_alarm_rate": round(false_alarm_rate, 4) if false_alarm_rate is not None else None,
        },
        "per_class": per_class_acc,
        "per_group": group_acc,
        "top_confusions": [(f"{gt}->{pred}", count)
                           for (gt, pred), count in confusion_pairs.most_common(10)],
        # Novel metrics
        "dwa": round(dwa, 4),  # M4
        "audc": audc_result,  # M1
        "crg": crg_result,  # M2
        "safety_recall": safety_recall_result,  # M3
        "f2_detect": f2_result,  # M6
    }

    # ---- M7: DRS (depends on all above) ----
    metrics["drs"] = compute_drs(metrics)

    return metrics


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_results(model_name, m):
    """Pretty-print all metrics."""
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} — {m['n_clips']} clips, {m['n_pairs']} person-level pairs")
    print(f"{'='*70}")

    # Standard
    print(f"\n  --- Standard Metrics ---")
    print(f"  Weighted Accuracy:  {m['weighted_accuracy']*100:.1f}%")
    print(f"  Macro Accuracy:     {m['macro_accuracy']*100:.1f}%")
    print(f"  ECE:                {m['ece']:.3f}")
    print(f"  Avg Confidence:     {m['avg_confidence']*100:.1f}%")
    print(f"  Conf-Acc Gap:       {m['confidence_accuracy_gap']*100:.1f}pp")

    # F1-magnet
    fm = m["f1_magnet"]
    print(f"  F1-Magnet:          {fm['f1_as_wrong_pred']}/{fm['total_wrong']}"
          f" ({fm['ratio']*100:.1f}%)")

    # Novel metrics
    print(f"\n  --- Novel Metrics ---")
    print(f"  M4 DWA:             {m['dwa']:.3f}  {'PASS' if m['dwa'] >= 0.80 else 'FAIL'} (>= 0.80)")

    audc = m.get("audc", {})
    if audc.get("naudc") is not None:
        print(f"  M1 nAUDC:           {audc['naudc']:.3f}  {'PASS' if audc['naudc'] >= 0.85 else 'FAIL'} (>= 0.85)")
        if audc.get("per_bin"):
            bins_str = ", ".join(f"{k}:{v*100:.0f}%" for k, v in sorted(audc["per_bin"].items()))
            print(f"     Visibility bins: {bins_str}")
    else:
        print(f"  M1 nAUDC:           N/A (insufficient visibility data)")

    crg = m.get("crg", {})
    if crg.get("crg") is not None:
        print(f"  M2 CRG:             {crg['crg']:.3f}  {'PASS' if crg['crg'] <= 0.20 else 'FAIL'} (<= 0.20)"
              f"  (n={crg['n_correct_perception']} correct perceptions)")
    else:
        print(f"  M2 CRG:             N/A (no correct perceptions)")

    sr = m.get("safety_recall", {})
    if sr.get("recall") is not None:
        print(f"  M3 Safety Recall:   {sr['recall']*100:.1f}%  {'PASS' if sr['recall'] >= 0.90 else 'FAIL'} (>= 90%)"
              f"  (n={sr.get('n_violations', 0)} violations)")
        print(f"     False-safe rate: {sr['false_safe_rate']*100:.1f}%")
        if sr.get("false_alarm_rate") is not None:
            print(f"     False-alarm:     {sr['false_alarm_rate']*100:.1f}%")
    else:
        print(f"  M3 Safety Recall:   N/A (no violations in GT)")

    f2 = m.get("f2_detect", {})
    print(f"  M6 F2-detect:       {f2.get('f2_detect', 0):.3f}  "
          f"{'PASS' if f2.get('f2_detect', 0) >= 0.70 else 'FAIL'} (>= 0.70)"
          f"  (P={f2.get('precision',0):.2f} R={f2.get('recall',0):.2f})")
    print(f"     Worker MAE:      {f2.get('worker_count_mae', 0):.2f}")

    # DRS
    drs = m.get("drs", {})
    print(f"\n  --- M7 Deployment Readiness Score ---")
    print(f"  DRS = {drs.get('drs', 0):.2f} ({drs.get('passes', 0)}/5 checks passed)")
    for check_name, check in drs.get("checks", {}).items():
        val = check.get("value")
        val_str = f"{val:.3f}" if val is not None else "N/A"
        status = "PASS" if check["pass"] else "FAIL"
        print(f"    {check_name:<15} {val_str:>8}  {check.get('threshold'):>6}  {status}")

    # (Safety details already shown under M3 above)

    # Top confusions
    print(f"\n  Top 5 Confusion Pairs (GT->Pred):")
    for pair, count in m["top_confusions"][:5]:
        gt_code, pred_code = pair.split("->")
        gt_name = ACTION_TAXONOMY.get(gt_code, gt_code)
        pred_name = ACTION_TAXONOMY.get(pred_code, pred_code)
        print(f"    {pair:>8} ({count:3d}x)  {gt_name} -> {pred_name}")

    # Group accuracy
    print(f"\n  Group-Level Accuracy:")
    for g, info in sorted(m["per_group"].items()):
        bar = "#" * int(info["accuracy"] * 20)
        print(f"    Group {g}: {info['accuracy']*100:5.1f}%  {bar}  "
              f"({info['correct']}/{info['total']})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute SteelBench eval metrics")
    parser.add_argument("--model", default=None,
                        help="Specific model (default: all in results/)")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip test files")
    args = parser.parse_args()

    print("Loading GT annotations...")
    gt = load_gt_annotations()
    print(f"Loaded GT for {len(gt)} clips")

    manifest = load_manifest()

    if args.model:
        model_names = [args.model]
    else:
        model_names = []
        for f in sorted(RESULTS_DIR.glob("*.jsonl")):
            name = f.stem
            if "_test" in name or "_batch_input" in name or "_chunk" in name:
                continue
            model_names.append(name)

    if not model_names:
        print("No model results found")
        return

    all_metrics = {}

    for model_name in model_names:
        try:
            predictions = load_predictions(model_name)
            if len(predictions) < 3:
                continue
            print(f"\n{model_name}: {len(predictions)} predictions")
        except FileNotFoundError:
            continue

        metrics = compute_metrics(predictions, gt, manifest)
        if metrics:
            all_metrics[model_name] = metrics
            print_results(model_name, metrics)

    # Comparison table
    if len(all_metrics) > 1:
        print(f"\n{'='*110}")
        print(f"  MODEL COMPARISON (sorted by accuracy)")
        print(f"{'='*110}")
        header = (f"  {'Model':<20} {'N':>4} {'Acc':>6} {'Macro':>6} {'DWA':>6} "
                  f"{'nAUDC':>6} {'CRG':>6} {'SfRec':>6} {'F2det':>6} {'DRS':>5} "
                  f"{'ECE':>6} {'Conf':>5} {'F1Mag':>6}")
        print(header)
        print(f"  {'-'*108}")

        sorted_models = sorted(all_metrics.items(),
                                key=lambda x: x[1]["weighted_accuracy"], reverse=True)
        for name, m in sorted_models:
            audc = m.get("audc", {})
            naudc_str = f"{audc['naudc']:.3f}" if audc.get("naudc") is not None else "  N/A"
            crg = m.get("crg", {})
            crg_str = f"{crg['crg']:.3f}" if crg.get("crg") is not None else "  N/A"
            sr = m.get("safety_recall", {})
            sr_str = f"{sr['recall']*100:.0f}%" if sr.get("recall") is not None else " N/A"
            f2 = m.get("f2_detect", {})
            drs = m.get("drs", {})

            print(f"  {name:<20} {m['n_clips']:>4} "
                  f"{m['weighted_accuracy']*100:5.1f}% "
                  f"{m['macro_accuracy']*100:5.1f}% "
                  f"{m['dwa']:.3f} "
                  f"{naudc_str} "
                  f"{crg_str} "
                  f"{sr_str:>5} "
                  f"{f2.get('f2_detect',0):.3f} "
                  f"{drs.get('drs',0):.2f} "
                  f"{m['ece']:.3f} "
                  f"{m['avg_confidence']*100:4.0f}% "
                  f"{m['f1_magnet']['ratio']*100:5.1f}%")

    # Save
    output_path = RESULTS_DIR / "metrics_summary.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
