#!/usr/bin/env python3
"""OpenAI Batch API runner for SteelBench evaluation.

Prepares batch JSONL, submits to OpenAI Batch API (50% off),
polls for completion, and downloads results.

Usage:
    # Prepare + submit batch
    python eval_runners/openai_batch.py --model gpt-5.4 --submit

    # Check status
    python eval_runners/openai_batch.py --status BATCH_ID

    # Download results
    python eval_runners/openai_batch.py --download BATCH_ID
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from annotation_tool.vlm_annotator import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    normalize_annotation,
    parse_json_response,
    validate_and_fix_response,
    validate_response_completeness,
)
from annotation_tool.safety_rules import (
    build_safety_prompt_block,
    load_camera_zones,
    load_safety_rules,
)

EVAL_DIR = Path(__file__).parent.parent / "eval_data"
FRAMES_DIR = EVAL_DIR / "frames"
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_DIR = EVAL_DIR / "config"

BATCH_MODELS = {
    "gpt4o": {
        "model": "gpt-4o",
        "response_format": {"type": "json_object"},
    },
    "gpt54": {
        "model": "gpt-5.4",
        "response_format": {"type": "json_object"},
    },
    "gpt54_mini": {
        "model": "gpt-5.4-mini",
        "response_format": {"type": "json_object"},
    },
}


def prepare_batch(model_name):
    """Prepare batch JSONL file for OpenAI Batch API."""
    if model_name not in BATCH_MODELS:
        print(f"Unknown model: {model_name}. Available: {list(BATCH_MODELS.keys())}")
        return None

    config = BATCH_MODELS[model_name]
    model_id = config["model"]

    clips = json.load(open(EVAL_DIR / "gt_clips.json"))
    print(f"Preparing batch for {model_id}: {len(clips)} clips")

    # Load safety rules
    try:
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
    except Exception:
        rules_config = None
        camera_zones = None

    # Load manifest
    manifest = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row.get("clip_id", "")] = row

    # Skip already-completed clips
    output_path = RESULTS_DIR / f"{model_name}.jsonl"
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("success"):
                        done.add(r.get("clip_id", ""))
                except:
                    pass
    if done:
        print(f"  Skipping {len(done)} already completed clips")
        clips = [c for c in clips if c not in done]

    batch_path = RESULTS_DIR / f"{model_name}_batch_input.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    with open(batch_path, "w") as f:
        for clip_id in clips:
            frame_dir = FRAMES_DIR / clip_id
            if not frame_dir.exists():
                skipped += 1
                continue

            frames = sorted(frame_dir.glob("frame_*.jpg"))
            if len(frames) != 8:
                skipped += 1
                continue

            # Build safety block
            row = manifest.get(clip_id, {})
            safety_block = None
            if rules_config and row.get("site"):
                try:
                    safety_block = build_safety_prompt_block(
                        row.get("camera_id", ""),
                        row.get("site", ""),
                        row.get("work_area", ""),
                        rules_config=rules_config,
                        camera_zones=camera_zones,
                    )
                except Exception:
                    pass

            prompt_text = USER_PROMPT
            if safety_block:
                prompt_text = USER_PROMPT + "\n\n" + safety_block

            # Build content with S3 image URLs (not base64)
            s3_bucket = "steelbench-eval-frames"
            content = [{"type": "text", "text": prompt_text}]
            for frame_path in frames:
                fname = os.path.basename(frame_path)
                s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{clip_id}/{fname}"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": s3_url,
                        "detail": "high",
                    },
                })

            # Batch API request format
            request = {
                "custom_id": clip_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    "max_completion_tokens": 6000,
                    "temperature": 0.1,
                    "response_format": config.get("response_format"),
                },
            }

            f.write(json.dumps(request) + "\n")
            written += 1

    print(f"  Written: {written} requests")
    print(f"  Skipped: {skipped} (missing frames)")
    print(f"  Batch file: {batch_path}")
    print(f"  File size: {batch_path.stat().st_size / 1024 / 1024:.1f} MB")
    return batch_path


def submit_batch(batch_path, model_name):
    """Upload batch file and submit to OpenAI Batch API."""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Uploading {batch_path}...")
    with open(batch_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    print("Submitting batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"model": model_name, "project": "steelbench"},
    )
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    print(f"  Created: {batch.created_at}")
    print(f"\nSave this batch ID: {batch.id}")
    print(f"Check status: python eval_runners/openai_batch.py --status {batch.id}")
    print(f"Download:     python eval_runners/openai_batch.py --download {batch.id} --model {model_name}")
    return batch.id


def check_status(batch_id):
    """Check batch job status."""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch = client.batches.retrieve(batch_id)
    print(f"Batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created: {batch.created_at}")
    if batch.request_counts:
        print(f"Requests: total={batch.request_counts.total}, "
              f"completed={batch.request_counts.completed}, "
              f"failed={batch.request_counts.failed}")
    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file: {batch.error_file_id}")
    return batch


def download_results(batch_id, model_name):
    """Download and parse batch results."""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not complete. Status: {batch.status}")
        return

    if not batch.output_file_id:
        print("No output file available")
        return

    print(f"Downloading results from {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id)

    output_path = RESULTS_DIR / f"{model_name}.jsonl"
    success = 0
    failed = 0

    with open(output_path, "a") as out_f:
        for line in content.text.strip().split("\n"):
            try:
                result = json.loads(line)
                clip_id = result.get("custom_id", "")
                response = result.get("response", {})
                body = response.get("body", {})
                choices = body.get("choices", [])

                if choices:
                    raw_text = choices[0].get("message", {}).get("content", "")
                    parsed = parse_json_response(raw_text)

                    if parsed:
                        parsed = validate_and_fix_response(parsed) or parsed
                        parsed, _ = validate_response_completeness(parsed)
                        normalized = normalize_annotation(parsed)

                        record = {
                            "clip_id": clip_id,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "model": BATCH_MODELS.get(model_name, {}).get("model", model_name),
                            "success": True,
                            "parsed": parsed,
                            "normalized": normalized,
                            "raw_response": raw_text[:5000],
                        }
                        out_f.write(json.dumps(record) + "\n")
                        success += 1
                    else:
                        record = {
                            "clip_id": clip_id,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "model": BATCH_MODELS.get(model_name, {}).get("model", model_name),
                            "success": False,
                            "error": "parse_fail",
                            "raw_response": raw_text[:5000],
                        }
                        out_f.write(json.dumps(record) + "\n")
                        failed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

    print(f"Results: {success} success, {failed} failed")
    print(f"Output: {output_path}")

    # Download errors if any
    if batch.error_file_id:
        errors = client.files.content(batch.error_file_id)
        error_path = RESULTS_DIR / f"{model_name}_batch_errors.jsonl"
        with open(error_path, "w") as f:
            f.write(errors.text)
        print(f"Errors: {error_path}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API runner")
    parser.add_argument("--model", choices=list(BATCH_MODELS.keys()),
                        help="Model to run")
    parser.add_argument("--submit", action="store_true",
                        help="Prepare and submit batch")
    parser.add_argument("--status", metavar="BATCH_ID",
                        help="Check batch status")
    parser.add_argument("--download", metavar="BATCH_ID",
                        help="Download batch results")
    args = parser.parse_args()

    load_dotenv()

    if args.status:
        check_status(args.status)
    elif args.download:
        if not args.model:
            print("--model required with --download")
            return
        download_results(args.download, args.model)
    elif args.submit:
        if not args.model:
            print("--model required with --submit")
            return
        batch_path = prepare_batch(args.model)
        if batch_path:
            submit_batch(batch_path, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
