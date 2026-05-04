#!/usr/bin/env python3
"""Extract per-VLM predictions from vlm_annotations.jsonl into eval-ready files.

Reads the dual-VLM annotation results and produces standalone prediction files
for each model, compatible with eval_metrics.evaluate_all().

Also reports parse failure rates and X1/open-vocabulary usage statistics.

Usage:
    python -m annotation_tool.extract_predictions
    python -m annotation_tool.extract_predictions --output-dir eval_results
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_VLM_RESULTS = os.path.join(
    PROJECT_ROOT, "annotation_tool", "data", "vlm_results", "vlm_annotations.jsonl"
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "eval_results")


def load_vlm_annotations(path: str) -> list:
    """Load all records from vlm_annotations.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_model_predictions(records: list, vlm_key: str = "vlm") -> tuple:
    """Extract predictions for a VLM from annotation records.

    Args:
        records: list of annotation records from vlm_annotations.jsonl
        vlm_key: 'vlm' (single-VLM format) or legacy 'vlm_a'/'vlm_b'

    Returns:
        (predictions_layer1, predictions_layer2, stats) where predictions
        follow the schema expected by eval_metrics.evaluate_all()
    """
    predictions_l1 = []
    predictions_l2 = []
    stats = {
        "total_clips": 0,
        "success": 0,
        "parse_failures": 0,
        "model_name": "",
        "x1_count": 0,
        "action_distribution": Counter(),
        "scene_type_distribution": Counter(),
        "confidence_scores": [],
    }

    for record in records:
        stats["total_clips"] += 1
        clip_id = record.get("clip_id", "")
        vlm_data = record.get(vlm_key, {})

        if not vlm_data:
            stats["parse_failures"] += 1
            continue

        if not stats["model_name"]:
            stats["model_name"] = vlm_data.get("model", vlm_key)

        if not vlm_data.get("success", False):
            stats["parse_failures"] += 1
            continue

        normalized = vlm_data.get("normalized")
        if not normalized:
            stats["parse_failures"] += 1
            continue

        stats["success"] += 1

        # Build prediction dict in eval_metrics-compatible format
        scene_type = normalized.get("scene_type", "SA")
        stats["scene_type_distribution"][scene_type] += 1

        persons = normalized.get("persons", [])
        has_layer2 = False

        for person in persons:
            action_code = person.get("action_code", "")
            stats["action_distribution"][action_code] += 1
            if action_code == "X1":
                stats["x1_count"] += 1
                has_layer2 = True

            confidence = person.get("confidence")
            if confidence is not None:
                try:
                    stats["confidence_scores"].append(float(confidence))
                except (ValueError, TypeError):
                    pass

        pred = {
            "clip_id": clip_id,
            "scene_type": scene_type,
            "num_workers": normalized.get("num_workers", len(persons)),
            "persons": persons,
            "action_transition": normalized.get("action_transition", {}),
            "visibility_conditions": normalized.get("visibility_conditions", []),
        }

        # Preserve raw response for failure mode analysis
        raw_response = vlm_data.get("raw_response", "")
        if raw_response:
            pred["_raw_response"] = raw_response

        if has_layer2:
            predictions_l2.append(pred)
        else:
            predictions_l1.append(pred)

    # Convert Counter to dict for JSON serialization
    stats["action_distribution"] = dict(stats["action_distribution"])
    stats["scene_type_distribution"] = dict(stats["scene_type_distribution"])
    stats["parse_failure_rate"] = round(
        stats["parse_failures"] / stats["total_clips"], 4
    ) if stats["total_clips"] > 0 else 0.0

    # Confidence summary
    conf_scores = stats["confidence_scores"]
    if conf_scores:
        stats["confidence_summary"] = {
            "mean": round(sum(conf_scores) / len(conf_scores), 4),
            "min": round(min(conf_scores), 4),
            "max": round(max(conf_scores), 4),
            "count": len(conf_scores),
        }
    else:
        stats["confidence_summary"] = {}
    # Remove raw list from stats (too large for summary)
    del stats["confidence_scores"]

    return predictions_l1, predictions_l2, stats


def compute_x1_summary(records: list) -> dict:
    """Compute open vocabulary (X1) usage statistics."""
    summary = {
        "x1_clips": 0,
        "total_clips": len(records),
        "x1_workers": 0,
        "x1_descriptions": [],
        "x1_closest_classes": Counter(),
    }

    for record in records:
        vlm_data = record.get("vlm", {})
        normalized = vlm_data.get("normalized", {})
        if not normalized:
            continue
        persons = normalized.get("persons", [])
        clip_has_x1 = False
        for person in persons:
            if person.get("action_code") == "X1":
                clip_has_x1 = True
                summary["x1_workers"] += 1

                unlisted = person.get("unlisted_action", {})
                desc = unlisted.get("free_text_description",
                                    person.get("free_text_description", ""))
                closest = unlisted.get("closest_existing_class",
                                       person.get("closest_existing_class", ""))
                if desc:
                    summary["x1_descriptions"].append({
                        "clip_id": record.get("clip_id", ""),
                        "description": desc,
                        "closest_class": closest,
                    })
                if closest:
                    summary["x1_closest_classes"][closest] += 1

        if clip_has_x1:
            summary["x1_clips"] += 1

    summary["x1_closest_classes"] = dict(summary["x1_closest_classes"])

    if summary["total_clips"] > 0:
        summary["taxonomy_coverage"] = round(
            1 - summary["x1_clips"] / summary["total_clips"], 4
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-VLM predictions from vlm_annotations.jsonl"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_VLM_RESULTS,
        help="Path to vlm_annotations.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for prediction files"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No VLM annotations found at {args.input}")
        print("Run vlm_annotator.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading annotations from {args.input}...")
    records = load_vlm_annotations(args.input)
    print(f"Loaded {len(records)} annotation records")

    # Extract predictions (single VLM format)
    print(f"\n{'='*60}")
    print("Extracting predictions...")
    preds_l1, preds_l2, stats = extract_model_predictions(records, "vlm")

    model_name = stats["model_name"].replace("/", "_").replace(" ", "_")
    if not model_name:
        model_name = "vlm"

    # Save predictions
    pred_path = os.path.join(args.output_dir, f"{model_name}_predictions.json")
    with open(pred_path, "w") as f:
        json.dump({
            "model": stats["model_name"],
            "layer1_predictions": preds_l1,
            "layer2_predictions": preds_l2,
        }, f, indent=2)
    print(f"  Predictions saved to {pred_path}")
    print(f"  Layer 1: {len(preds_l1)} clips, Layer 2: {len(preds_l2)} clips")

    # Save stats
    stats_path = os.path.join(args.output_dir, f"{model_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")

    # Print summary
    print(f"\n  Model: {stats['model_name']}")
    print(f"  Total clips: {stats['total_clips']}")
    print(f"  Success: {stats['success']} ({100*stats['success']/max(stats['total_clips'],1):.1f}%)")
    print(f"  Parse failures: {stats['parse_failures']} "
          f"(rate: {stats['parse_failure_rate']:.2%})")
    print(f"  X1 (unlisted) workers: {stats['x1_count']}")
    if stats.get("confidence_summary"):
        cs = stats["confidence_summary"]
        print(f"  Confidence: mean={cs['mean']}, min={cs['min']}, max={cs['max']}")

    # X1 summary
    print(f"\n{'='*60}")
    print("Open Vocabulary (X1) Summary:")
    x1_summary = compute_x1_summary(records)
    x1_path = os.path.join(args.output_dir, "x1_summary.json")
    with open(x1_path, "w") as f:
        json.dump(x1_summary, f, indent=2)
    print(f"  X1 clips: {x1_summary['x1_clips']}")
    print(f"  X1 workers: {x1_summary['x1_workers']}")
    print(f"  Taxonomy coverage: {x1_summary.get('taxonomy_coverage', 'N/A')}")
    print(f"  Unique X1 descriptions: {len(x1_summary['x1_descriptions'])}")
    print(f"  X1 summary saved to {x1_path}")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
