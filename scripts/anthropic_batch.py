#!/usr/bin/env python3
"""Anthropic Message Batches API for Claude inference on SteelBench.

50% cheaper than real-time API. Submits all clips as a batch, polls for
completion, then converts results to the same JSONL format as eval_inference.py.

Usage:
    # Submit batch for Claude Opus 4.6
    python scripts/anthropic_batch.py --model claude_opus --submit

    # Check batch status
    python scripts/anthropic_batch.py --model claude_opus --status

    # Download results when batch is done
    python scripts/anthropic_batch.py --model claude_opus --results

    # Submit, poll, and download in one go
    python scripts/anthropic_batch.py --model claude_opus --submit --poll --results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent.parent / "eval_data"
RESULTS_DIR = EVAL_DIR / "results"
FRAMES_DIR = EVAL_DIR / "frames"
GT_CLIPS_FILE = EVAL_DIR / "gt_clips.json"
CONFIG_DIR = EVAL_DIR / "config"
BATCH_STATE_DIR = RESULTS_DIR  # Store batch IDs here
S3_BUCKET = "steelbench-eval-frames"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"

MODELS = {
    "claude_opus": "claude-opus-4-7",
}


def load_manifest():
    """Load batch manifest for safety rule lookup."""
    import csv
    manifest = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                clip_id = row.get("clip_id", "")
                if clip_id:
                    manifest[clip_id] = row
    return manifest


def load_existing_success(output_path):
    """Load clip IDs that already have successful results."""
    done = set()
    if output_path.exists():
        for line in open(output_path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("success") and r.get("clip_id"):
                    done.add(r["clip_id"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return done


def build_message_content(clip_id, manifest, rules_config, camera_zones):
    """Build the Anthropic message content for a single clip using S3 URLs."""
    # Safety block
    row = manifest.get(clip_id, {})
    safety_block = None
    site = row.get("site", "")
    camera_id = row.get("camera_id", "")
    work_area = row.get("work_area", "")
    if site and rules_config:
        try:
            safety_block = build_safety_prompt_block(
                camera_id, site, work_area,
                rules_config=rules_config,
                camera_zones=camera_zones,
                config_dir=str(CONFIG_DIR),
            )
        except Exception:
            pass

    prompt_text = USER_PROMPT
    if safety_block:
        prompt_text = USER_PROMPT + "\n\n" + safety_block

    # Build content: S3 URL images + text
    content = []
    from urllib.parse import quote
    for i in range(8):  # 8 frames per clip: frame_0.jpg .. frame_7.jpg
        url = f"{S3_BASE_URL}/{quote(clip_id, safe='')}/frame_{i}.jpg"
        content.append({
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        })
    content.append({"type": "text", "text": prompt_text})
    return content


def submit_batch(model_name, frames_dir=None):
    """Build and submit a single batch using S3 URLs (no base64, no OOM)."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    model_id = MODELS[model_name]
    output_path = RESULTS_DIR / f"{model_name}.jsonl"

    # Load clips
    clip_ids = json.load(open(GT_CLIPS_FILE))
    print(f"Total GT clips: {len(clip_ids)}")

    # Skip already-done clips
    already_done = load_existing_success(output_path)
    remaining = [c for c in clip_ids if c not in already_done]
    print(f"Already done: {len(already_done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All clips already processed!")
        return

    # Load safety rules
    rules_config = None
    camera_zones = None
    try:
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
        print(f"Loaded safety rules from {CONFIG_DIR}")
    except Exception as e:
        print(f"Warning: Could not load safety rules: {e}")

    manifest = load_manifest()

    # Build all requests (lightweight — just URLs, no image data)
    requests = []
    id_map = {}  # custom_id -> clip_id (for reversing sanitization)
    for clip_id in remaining:
        content = build_message_content(clip_id, manifest, rules_config, camera_zones)
        # Sanitize: only alphanumeric, hyphens, underscores allowed
        import re
        custom_id = re.sub(r'[^a-zA-Z0-9_-]', '_', clip_id)[:64]
        id_map[custom_id] = clip_id
        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=model_id,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                ),
            )
        )

    print(f"Built {len(requests)} requests (using S3 URLs, no base64)")

    # Submit single batch
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"Submitting batch of {len(requests)} requests for {model_id}...")

    message_batch = client.messages.batches.create(requests=requests)

    print(f"Batch created: {message_batch.id}")
    print(f"Status: {message_batch.processing_status}")
    print(f"Request counts: {message_batch.request_counts}")

    # Save batch ID and ID mapping
    state_file = BATCH_STATE_DIR / f"{model_name}_batch_id.txt"
    with open(state_file, "w") as f:
        f.write(message_batch.id)

    map_file = BATCH_STATE_DIR / f"{model_name}_id_map.json"
    with open(map_file, "w") as f:
        json.dump(id_map, f)

    print(f"Batch ID saved to {state_file}")
    print(f"ID mapping saved to {map_file}")
    return message_batch.id


def _load_batch_ids(model_name):
    """Load batch IDs from state file (supports multiple batches)."""
    state_file = BATCH_STATE_DIR / f"{model_name}_batch_id.txt"
    if not state_file.exists():
        return []
    return [line.strip() for line in state_file.read_text().strip().split("\n") if line.strip()]


def check_status(model_name):
    """Check the status of all submitted batches."""
    import anthropic

    batch_ids = _load_batch_ids(model_name)
    if not batch_ids:
        print(f"No batch IDs found for {model_name}. Submit a batch first.")
        return None

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    total_succeeded = 0
    total_processing = 0
    total_errored = 0
    total_expired = 0

    for batch_id in batch_ids:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        total_succeeded += c.succeeded
        total_processing += c.processing
        total_errored += c.errored
        total_expired += c.expired
        ended = f"  ended={batch.ended_at}" if batch.ended_at else ""
        print(f"  {batch.id}: {batch.processing_status} "
              f"ok={c.succeeded} err={c.errored} exp={c.expired} proc={c.processing}{ended}")

    print(f"\nTotal: succeeded={total_succeeded} processing={total_processing} "
          f"errored={total_errored} expired={total_expired}")
    all_done = total_processing == 0
    print(f"All done: {all_done}")
    return all_done


def poll_until_done(model_name, interval=60):
    """Poll all batches until completion."""
    import anthropic

    batch_ids = _load_batch_ids(model_name)
    if not batch_ids:
        print(f"No batch IDs found for {model_name}")
        return

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"Polling {len(batch_ids)} batches every {interval}s...")

    while True:
        all_done = True
        total_ok = 0
        total_proc = 0
        for batch_id in batch_ids:
            batch = client.messages.batches.retrieve(batch_id)
            c = batch.request_counts
            total_ok += c.succeeded
            total_proc += c.processing
            if batch.processing_status != "ended":
                all_done = False

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
              f"succeeded={total_ok} processing={total_proc}")

        if all_done:
            print("All batches complete!")
            return

        time.sleep(interval)


def download_results(model_name):
    """Download batch results from all batches and convert to eval JSONL format."""
    import anthropic

    batch_ids = _load_batch_ids(model_name)
    if not batch_ids:
        print(f"No batch IDs found for {model_name}")
        return

    model_id = MODELS[model_name]
    output_path = RESULTS_DIR / f"{model_name}.jsonl"
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load ID mapping (custom_id -> clip_id)
    map_file = BATCH_STATE_DIR / f"{model_name}_id_map.json"
    id_map = {}
    if map_file.exists():
        id_map = json.load(open(map_file))

    success_count = 0
    fail_count = 0

    with open(output_path, "a") as out_f:
        for batch_id in batch_ids:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                print(f"Batch {batch_id} not yet complete "
                      f"(processing={batch.request_counts.processing}), skipping")
                continue

            c = batch.request_counts
            print(f"Downloading {batch_id}: ok={c.succeeded} err={c.errored} exp={c.expired}")

            for result in client.messages.batches.results(batch_id):
                # Reverse-map sanitized custom_id back to original clip_id
                clip_id = id_map.get(result.custom_id, result.custom_id)

                if result.result.type == "succeeded":
                    raw_text = ""
                    for block in result.result.message.content:
                        if block.type == "text":
                            raw_text += block.text

                    parsed = parse_json_response(raw_text)
                    normalized = None
                    success = False

                    if parsed:
                        parsed = validate_and_fix_response(parsed) or parsed
                        parsed, _ = validate_response_completeness(parsed)
                        success = True
                        try:
                            normalized = normalize_annotation(parsed)
                        except Exception:
                            normalized = None

                    record = {
                        "clip_id": clip_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id,
                        "success": success,
                        "parsed": parsed,
                        "normalized": normalized,
                        "raw_response": (raw_text or "")[:5000],
                    }

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        print(f"  Parse fail: {clip_id}")

                elif result.result.type == "errored":
                    fail_count += 1
                    error_msg = str(result.result.error) if hasattr(result.result, 'error') else "batch_error"
                    record = {
                        "clip_id": clip_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id,
                        "success": False,
                        "error": error_msg,
                    }
                    print(f"  Error: {clip_id}: {error_msg[:80]}")

                elif result.result.type == "expired":
                    fail_count += 1
                    record = {
                        "clip_id": clip_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id,
                        "success": False,
                        "error": "batch_expired",
                    }
                    print(f"  Expired: {clip_id}")

                else:
                    fail_count += 1
                    record = {
                        "clip_id": clip_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id,
                        "success": False,
                        "error": f"unknown_result_type={result.result.type}",
                    }

                out_f.write(json.dumps(record) + "\n")

    print(f"\nResults written to {output_path}")
    print(f"Total: success={success_count}, failed={fail_count}")


def main():
    parser = argparse.ArgumentParser(description="Anthropic batch inference for SteelBench")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model to run")
    parser.add_argument("--submit", action="store_true", help="Submit batch")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--poll", action="store_true",
                        help="Poll until batch completes")
    parser.add_argument("--results", action="store_true",
                        help="Download and convert results")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Polling interval in seconds (default: 60)")
    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        sys.exit(1)

    if not any([args.submit, args.status, args.poll, args.results]):
        print("Specify at least one action: --submit, --status, --poll, --results")
        sys.exit(1)

    if args.submit:
        submit_batch(args.model)

    if args.status:
        check_status(args.model)

    if args.poll:
        poll_until_done(args.model, interval=args.poll_interval)

    if args.results:
        download_results(args.model)


if __name__ == "__main__":
    main()
