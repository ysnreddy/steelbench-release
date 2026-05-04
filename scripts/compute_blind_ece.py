#!/usr/bin/env python3
"""
Compute ECE on independent GT subsets:
1. Blind annotations (calibration_condition=blind, no VLM pre-fill shown)
2. Proper-chain expert annotations (source_type=tier_1, expert reviewed human work)
3. Standard gold ECE (all expert annotations, for comparison)

Usage:
    python scripts/compute_blind_ece.py \
        --vlm annotation_tool/data/vlm_results/vlm_unified.jsonl \
        --annotations active_batch/annotations \
        --calibration active_batch/data/calibration_set.json \
        --output paper_reference/anchoring_analysis/ece_audit.json
"""

import argparse
import json
import glob
import os
import math
from collections import defaultdict
from pathlib import Path


def load_vlm(vlm_path):
    """Load VLM predictions: {clip_id: {action, confidence, scene_type, ...}}"""
    vlm = {}
    with open(vlm_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            cid = rec.get("clip_id", "")
            if not cid or not rec.get("vlm", {}).get("success"):
                continue
            norm = rec["vlm"]["normalized"]
            persons = norm.get("persons", [])
            nw = norm.get("num_workers", 0)

            entry = {}
            # Layer 2
            if persons and nw <= 5:
                p0 = persons[0]
                action = p0.get("action_code", "")
                try:
                    conf = float(p0.get("confidence", 0.0))
                except (TypeError, ValueError):
                    conf = None
                if action and conf is not None and 0.0 <= conf <= 1.0:
                    entry["l2_action"] = action
                    entry["l2_conf"] = conf

            # Layer 1
            scene_type = (norm.get("scene_type") or "").strip()
            if scene_type and nw > 5 and persons:
                confs = []
                for p in persons:
                    try:
                        c = float(p.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= c <= 1.0:
                        confs.append(c)
                if confs:
                    entry["l1_scene_type"] = scene_type
                    entry["l1_conf"] = sum(confs) / len(confs)

            if entry:
                vlm[cid] = entry
    return vlm


def load_annotations(annotations_dir, annotator_filter=None):
    """Load all submitted annotations.
    Returns: {clip_id: [{annotator, annotation, source_type, cal_condition}]}
    """
    clips = defaultdict(list)
    for ann_dir_name in os.listdir(annotations_dir):
        ann_dir = os.path.join(annotations_dir, ann_dir_name)
        if not os.path.isdir(ann_dir) or ann_dir_name.startswith("."):
            continue
        if annotator_filter and ann_dir_name not in annotator_filter:
            continue
        for fpath in glob.glob(os.path.join(ann_dir, "*.json")):
            with open(fpath) as f:
                ann = json.load(f)
            if ann.get("status") != "submitted":
                continue
            et = ann.get("edit_tracking", {})
            clips[ann["clip_id"]].append({
                "annotator": ann_dir_name,
                "annotation": ann,
                "source_type": et.get("source_type"),
                "cal_condition": ann.get("calibration_condition"),
                "role": ann.get("annotator_role", ann.get("tier", "")),
            })
    return clips


def compute_ece(predictions, labels, confidences, n_bins=10):
    """Standard ECE computation."""
    if not predictions:
        return {"ece": None, "n": 0}
    n = len(predictions)
    bin_edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total_ece = 0.0
    bins = []
    for lo, hi in bin_edges:
        indices = []
        for j in range(n):
            c = confidences[j]
            if lo == 0:
                if c >= lo and c <= hi:
                    indices.append(j)
            else:
                if c > lo and c <= hi:
                    indices.append(j)
        if not indices:
            bins.append(None)
            continue
        bin_acc = sum(1 for j in indices if predictions[j] == labels[j]) / len(indices)
        bin_conf = sum(confidences[j] for j in indices) / len(indices)
        gap = abs(bin_acc - bin_conf)
        total_ece += gap * len(indices) / n
        bins.append({
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "gap": round(gap, 4),
            "count": len(indices),
        })

    accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / n
    avg_conf = sum(confidences) / n

    return {
        "ece": round(total_ece, 4),
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_conf, 4),
        "n": n,
        "bins": bins,
    }


def build_ece_triples(vlm, annotations, clip_filter=None):
    """Build (prediction, label, confidence) triples for ECE.
    For Layer 2: VLM action vs human action.
    For Layer 1: VLM scene_type vs human scene_type.
    """
    preds, labels, confs = [], [], []

    for cid, ann_list in annotations.items():
        if clip_filter and cid not in clip_filter:
            continue
        if cid not in vlm:
            continue

        v = vlm[cid]
        # Use first annotation (if multiple, they're from different annotators)
        ann = ann_list[0]["annotation"]
        layer = ann.get("annotation_layer")

        if layer == 2 and "l2_action" in v:
            persons = ann.get("persons", [])
            if persons:
                h_action = persons[0].get("action_code", "")
                if h_action:
                    preds.append(v["l2_action"])
                    labels.append(h_action)
                    confs.append(v["l2_conf"])
        elif layer == 1 and "l1_scene_type" in v:
            h_scene = (ann.get("scene_type") or "").strip()
            if h_scene:
                preds.append(v["l1_scene_type"])
                labels.append(h_scene)
                confs.append(v["l1_conf"])

    return preds, labels, confs


def main():
    parser = argparse.ArgumentParser(description="Compute ECE on independent GT subsets")
    parser.add_argument("--vlm", default="annotation_tool/data/vlm_results/vlm_unified.jsonl")
    parser.add_argument("--annotations", default="active_batch/annotations")
    parser.add_argument("--calibration", default="active_batch/data/calibration_set.json")
    parser.add_argument("--output", default="paper_reference/anchoring_analysis/ece_audit.json")
    args = parser.parse_args()

    print("Loading data...")
    vlm = load_vlm(args.vlm)
    all_anns = load_annotations(args.annotations)
    print(f"  VLM: {len(vlm)} clips")
    print(f"  Annotations: {len(all_anns)} clips")

    # Load calibration set
    cal_blind_ids = set()
    if os.path.exists(args.calibration):
        with open(args.calibration) as f:
            cal = json.load(f)
        cal_blind_ids = set(cal.get("blind_clips", {}).keys())
        print(f"  Calibration blind clips: {len(cal_blind_ids)}")

    results = {}

    # ── ECE 1: Blind annotations only ──
    # Use tier_1 annotations where calibration_condition=blind AND source_type is None
    # (truly blind — annotator never saw VLM)
    blind_anns = {}
    for cid, ann_list in all_anns.items():
        for entry in ann_list:
            cc = entry["cal_condition"]
            src = entry["source_type"]
            role = entry.get("role", "")
            # Truly blind: calibration_condition is "blind" and source was NOT vlm
            if cc == "blind" and src != "vlm" and "annotator" in entry["annotator"]:
                if cid not in blind_anns:
                    blind_anns[cid] = [entry]
                break
            # Also accept: clip in blind_ids, no prefill shown (source_type is None)
            if cid in cal_blind_ids and src is None and "annotator" in entry["annotator"]:
                if cid not in blind_anns:
                    blind_anns[cid] = [entry]
                break

    p, l, c = build_ece_triples(vlm, blind_anns)
    results["blind_ece"] = compute_ece(p, l, c)
    print(f"\n  Blind ECE: {results['blind_ece']['ece']} (n={results['blind_ece']['n']})")

    # ── ECE 2: Proper-chain expert annotations ──
    # Expert annotations where source_type=tier_1 (expert reviewed human work)
    proper_chain_anns = {}
    for cid, ann_list in all_anns.items():
        for entry in ann_list:
            if "expert" in entry["annotator"] and entry["source_type"] == "tier_1":
                if cid not in proper_chain_anns:
                    proper_chain_anns[cid] = [entry]
                break

    p, l, c = build_ece_triples(vlm, proper_chain_anns)
    results["proper_chain_ece"] = compute_ece(p, l, c)
    print(f"  Proper-chain ECE: {results['proper_chain_ece']['ece']} (n={results['proper_chain_ece']['n']})")

    # ── ECE 3: All expert annotations (standard gold, for comparison) ──
    expert_anns = {}
    for cid, ann_list in all_anns.items():
        for entry in ann_list:
            if "expert" in entry["annotator"]:
                if cid not in expert_anns:
                    expert_anns[cid] = [entry]
                break

    p, l, c = build_ece_triples(vlm, expert_anns)
    results["gold_ece"] = compute_ece(p, l, c)
    print(f"  Gold ECE (all expert): {results['gold_ece']['ece']} (n={results['gold_ece']['n']})")

    # ── ECE 4: VLM-sourced expert only (anchored GT) ──
    vlm_sourced_anns = {}
    for cid, ann_list in all_anns.items():
        for entry in ann_list:
            if "expert" in entry["annotator"] and entry["source_type"] == "vlm":
                if cid not in vlm_sourced_anns:
                    vlm_sourced_anns[cid] = [entry]
                break

    p, l, c = build_ece_triples(vlm, vlm_sourced_anns)
    results["vlm_sourced_ece"] = compute_ece(p, l, c)
    print(f"  VLM-sourced ECE: {results['vlm_sourced_ece']['ece']} (n={results['vlm_sourced_ece']['n']})")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("ECE COMPARISON")
    print("=" * 60)
    print(f"{'GT Source':<25s} {'ECE':>8s} {'Accuracy':>10s} {'Avg Conf':>10s} {'N':>6s}")
    print("-" * 60)
    for name, label in [
        ("blind_ece", "Blind (no VLM shown)"),
        ("proper_chain_ece", "Expert (reviewed tier_1)"),
        ("vlm_sourced_ece", "Expert (edited VLM)"),
        ("gold_ece", "All expert (standard)"),
    ]:
        r = results[name]
        ece = f"{r['ece']:.4f}" if r['ece'] is not None else "N/A"
        acc = f"{r['accuracy']:.1%}" if r.get('accuracy') is not None else "N/A"
        conf = f"{r['avg_confidence']:.1%}" if r.get('avg_confidence') is not None else "N/A"
        print(f"{label:<25s} {ece:>8s} {acc:>10s} {conf:>10s} {r['n']:>6d}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
