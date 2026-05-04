#!/usr/bin/env python3
"""Chunked OpenAI Batch API runner — submits in batches of N clips.

Handles token queue limits by submitting small chunks, waiting for
completion, then submitting the next chunk. Runs automatically overnight.

Usage:
    # Run GPT-5.4 in chunks of 50
    python eval_runners/openai_batch_chunked.py --model gpt54 --chunk-size 50

    # Run both GPT-4o and GPT-5.4 in parallel
    python eval_runners/openai_batch_chunked.py --model gpt4o --chunk-size 50 &
    python eval_runners/openai_batch_chunked.py --model gpt54 --chunk-size 50 &
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
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
S3_BUCKET = "steelbench-eval-frames"

BATCH_MODELS = {
    "gpt4o": {"model": "gpt-4o", "response_format": {"type": "json_object"}},
    "gpt54": {"model": "gpt-5.4", "response_format": {"type": "json_object"}},
    "gpt54_mini": {"model": "gpt-5.4-mini", "response_format": {"type": "json_object"}},
}


def get_completed_clips(model_name):
    """Get clip IDs already completed."""
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
    return done


def prepare_chunk(clip_ids, model_config, rules_config, camera_zones, manifest):
    """Prepare a batch JSONL for a chunk of clips."""
    model_id = model_config["model"]
    lines = []

    for clip_id in clip_ids:
        frame_dir = FRAMES_DIR / clip_id
        if not frame_dir.exists():
            continue
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frames) != 8:
            continue

        # Safety block
        row = manifest.get(clip_id, {})
        safety_block = None
        if rules_config and row.get("site"):
            try:
                safety_block = build_safety_prompt_block(
                    row.get("camera_id", ""), row.get("site", ""),
                    row.get("work_area", ""),
                    rules_config=rules_config, camera_zones=camera_zones,
                )
            except:
                pass

        prompt_text = USER_PROMPT
        if safety_block:
            prompt_text = USER_PROMPT + "\n\n" + safety_block

        # S3 image URLs
        content = [{"type": "text", "text": prompt_text}]
        for fp in frames:
            s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{clip_id}/{fp.name}"
            content.append({
                "type": "image_url",
                "image_url": {"url": s3_url, "detail": "high"},
            })

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
                "response_format": model_config.get("response_format"),
            },
        }
        lines.append(json.dumps(request))

    return lines


def submit_and_wait(client, chunk_lines, model_name, chunk_num):
    """Submit a chunk and wait for completion."""
    # Write temp batch file
    batch_path = RESULTS_DIR / f"{model_name}_chunk_{chunk_num}.jsonl"
    with open(batch_path, "w") as f:
        f.write("\n".join(chunk_lines))

    file_size_kb = batch_path.stat().st_size / 1024
    print(f"  Chunk {chunk_num}: {len(chunk_lines)} clips, {file_size_kb:.0f} KB")

    # Upload
    with open(batch_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    # Submit
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"model": model_name, "chunk": str(chunk_num), "project": "steelbench"},
    )
    print(f"  Submitted: {batch.id} (status: {batch.status})")

    # Poll for completion
    poll_interval = 30  # seconds
    start_time = time.time()
    while True:
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        elapsed = (time.time() - start_time) / 60
        rc = batch.request_counts
        completed = rc.completed if rc else 0
        total = rc.total if rc else len(chunk_lines)
        failed = rc.failed if rc else 0

        if batch.status == "completed":
            print(f"  Completed in {elapsed:.1f} min: {completed}/{total} success, {failed} failed")
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            print(f"  FAILED ({batch.status}) after {elapsed:.1f} min")
            if batch.errors:
                for err in batch.errors.data[:3]:
                    print(f"    Error: {err.message}")
            # Clean up temp file
            batch_path.unlink(missing_ok=True)
            return 0
        else:
            print(f"  [{elapsed:.0f}m] {batch.status}: {completed}/{total}", end="\r")

    # Download results
    success_count = 0
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id)
        output_path = RESULTS_DIR / f"{model_name}.jsonl"

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
                                "timestamp": datetime.now().isoformat(),
                                "model": BATCH_MODELS[model_name]["model"],
                                "success": True,
                                "parsed": parsed,
                                "normalized": normalized,
                                "raw_response": raw_text[:5000],
                            }
                            out_f.write(json.dumps(record) + "\n")
                            success_count += 1
                        else:
                            record = {
                                "clip_id": clip_id,
                                "timestamp": datetime.now().isoformat(),
                                "model": BATCH_MODELS[model_name]["model"],
                                "success": False,
                                "error": "parse_fail",
                                "raw_response": raw_text[:5000],
                            }
                            out_f.write(json.dumps(record) + "\n")
                except:
                    pass

    # Clean up temp file
    batch_path.unlink(missing_ok=True)
    return success_count


def run_chunked(model_name, chunk_size=50):
    """Run model in chunks via batch API."""
    load_dotenv()

    if model_name not in BATCH_MODELS:
        print(f"Unknown model: {model_name}")
        return

    config = BATCH_MODELS[model_name]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load clips
    all_clips = json.load(open(EVAL_DIR / "gt_clips.json"))
    done = get_completed_clips(model_name)
    remaining = [c for c in all_clips if c not in done]

    print(f"{'='*60}")
    print(f"  {config['model']} — Chunked Batch Runner")
    print(f"{'='*60}")
    print(f"  Total clips: {len(all_clips)}")
    print(f"  Already done: {len(done)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunks needed: {(len(remaining) + chunk_size - 1) // chunk_size}")
    print()

    if not remaining:
        print("All clips done!")
        return

    # Load config
    try:
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
    except:
        rules_config = None
        camera_zones = None

    manifest = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest[row.get("clip_id", "")] = row

    # Process chunks
    total_success = len(done)
    chunk_num = 0
    start_time = time.time()

    while remaining:
        chunk_num += 1
        chunk = remaining[:chunk_size]
        remaining = remaining[chunk_size:]

        print(f"\n--- Chunk {chunk_num} ({len(chunk)} clips, {len(remaining)} remaining) ---")

        chunk_lines = prepare_chunk(chunk, config, rules_config, camera_zones, manifest)
        if not chunk_lines:
            print("  No valid clips in chunk, skipping")
            continue

        success = submit_and_wait(client, chunk_lines, model_name, chunk_num)
        total_success += success

        elapsed = (time.time() - start_time) / 60
        print(f"  Total progress: {total_success}/{len(all_clips)} "
              f"({elapsed:.0f} min elapsed)")

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"  DONE: {total_success}/{len(all_clips)} in {elapsed:.0f} min")
    print(f"  Output: {RESULTS_DIR / f'{model_name}.jsonl'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Chunked OpenAI Batch runner")
    parser.add_argument("--model", required=True, choices=list(BATCH_MODELS.keys()))
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    run_chunked(args.model, args.chunk_size)


if __name__ == "__main__":
    main()
