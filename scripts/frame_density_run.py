#!/usr/bin/env python3
"""Frame density ablation runner.

Runs evaluation at 1, 2, 4, 15 frames per clip on 150 stratified clips.
8-frame results already exist from main evaluation (V1).
15 frames = 1fps for 15-second clips (natural temporal cadence; sampled at
midpoints 0.5s, 1.5s, ..., 14.5s to avoid EOF-seek issues).

Usage:
    # Run DeepInfra model (Gemma) at specific frame count
    python scripts/frame_density_run.py --provider deepinfra --frames 1
    python scripts/frame_density_run.py --provider deepinfra --frames 4
    python scripts/frame_density_run.py --provider deepinfra --frames 15

    # Generate OpenAI batch input for GPT-4o
    python scripts/frame_density_run.py --provider openai --frames 1 --batch-input
    python scripts/frame_density_run.py --provider openai --frames 15 --batch-submit

    # Extract 15 frames at 1fps from mp4 (desktop PC only)
    python scripts/frame_density_run.py --extract-15 --clips-dir /path/to/clips --output-dir /path/to/frames15
"""

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

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
from annotation_tool.safety_rules import build_safety_prompt_block

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent.parent / "eval_data"
RESULTS_DIR = EVAL_DIR / "results" / "frame_density"
FRAMES_DIR = EVAL_DIR / "frames"
CONFIG_DIR = EVAL_DIR / "config"
CLIPS_FILE = EVAL_DIR / "prompt_sensitivity_clips.json"  # same 150 clips

S3_BUCKET = "steelbench-eval-frames"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"

# VPS frames path (for running on VPS)
VPS_FRAMES_DIR = Path("/opt/steelbench/active_batch/frames")

# Frame selection for each density.
# Indices 0..7 reference the 8-frame canonical set in active_batch/frames/.
# Index lists with len > 8 (e.g. 15) reference the dense set in eval_data/frames_15/.
FRAME_SELECTION = {
    1: [3],                    # center frame (from 8-frame set)
    2: [0, 7],                 # first, last (from 8-frame set)
    4: [0, 2, 5, 7],           # evenly spaced (from 8-frame set)
    8: [0, 1, 2, 3, 4, 5, 6, 7],  # all (existing V1)
    15: list(range(15)),       # 1fps for 15s clips (from 15-frame dense set)
}


def load_manifest():
    manifest = {}
    with open(CONFIG_DIR / "batch_manifest.csv") as f:
        for row in csv.DictReader(f):
            if row.get("clip_id"):
                manifest[row["clip_id"]] = row
    return manifest


def get_safety_block(clip_id, manifest):
    row = manifest.get(clip_id, {})
    site = row.get("site", "")
    camera_id = row.get("camera_id", "")
    work_area = row.get("work_area", "")
    if not site:
        return ""
    try:
        from annotation_tool.safety_rules import load_safety_rules, load_camera_zones
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
        return build_safety_prompt_block(
            camera_id, site, work_area,
            rules_config=rules_config, camera_zones=camera_zones,
            config_dir=str(CONFIG_DIR),
        )
    except Exception:
        return ""


def get_frame_indices(num_frames):
    """Get frame indices for a given frame count."""
    if num_frames in FRAME_SELECTION:
        return FRAME_SELECTION[num_frames]
    raise ValueError(f"Unsupported frame count: {num_frames}")


def _is_dense(num_frames):
    """Dense densities use zero-padded filenames (frame_NN.jpg) and live in
    a separate per-density directory like eval_data/frames_15/."""
    return num_frames >= 10


def get_s3_urls(clip_id, num_frames):
    """Get S3 URLs for selected frames.
    Dense densities (>=10) use zero-padded filenames matching the local naming
    produced by extract_dense_frames()."""
    indices = get_frame_indices(num_frames)
    pad = _is_dense(num_frames)
    urls = []
    for i in indices:
        name = f"frame_{i:02d}.jpg" if pad else f"frame_{i}.jpg"
        url = f"{S3_BASE_URL}/{quote(clip_id, safe='')}/{name}"
        urls.append(url)
    return urls


def get_base64_images(clip_id, num_frames, frames_dir=None):
    """Load selected frames as base64 data URLs.
    Dense densities (>=10) use zero-padded filenames frame_NN.jpg."""
    if frames_dir is None:
        frames_dir = VPS_FRAMES_DIR if VPS_FRAMES_DIR.exists() else FRAMES_DIR

    frame_dir = frames_dir / clip_id
    indices = get_frame_indices(num_frames)
    pad = _is_dense(num_frames)

    images = []
    for i in indices:
        name = f"frame_{i:02d}.jpg" if pad else f"frame_{i}.jpg"
        frame_path = frame_dir / name

        if not frame_path.exists():
            continue
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        images.append(f"data:image/jpeg;base64,{b64}")
    return images


def build_prompt(clip_id, manifest, num_frames):
    """Build the V1 prompt with safety block, substituting the hardcoded
    '8 frames' string in USER_PROMPT with the actual frame count for this
    density. The original USER_PROMPT (used by the canonical 8-frame main
    eval) has '8 frames' in 4 places; we substitute all of them so the
    model is told it has the same number of frames it's being shown.
    Without this substitution the 1/2/4/15-frame ablations would be biased
    by a prompt claiming 8 frames at every density (see
    scripts/ab_prompt_frame_count.py: ~33% disagreement on primary_person
    unsafe_act between '8 frames' and density-aware prompts at 4 frames)."""
    safety_block = get_safety_block(clip_id, manifest)
    prompt_text = USER_PROMPT.replace("8 frames", f"{num_frames} frames")
    if safety_block:
        prompt_text = prompt_text + "\n\n" + safety_block
    return SYSTEM_PROMPT, prompt_text


# ---------------------------------------------------------------------------
# Dense-frame extraction from mp4 (15-frame at 1fps default)
# ---------------------------------------------------------------------------
def extract_dense_frames(clips_dir, output_dir, clips_file, num_frames=15, fps=1.0):
    """Extract num_frames frames per clip using midpoint sampling at the
    given fps. For 15-second clips at 1fps with num_frames=15, this samples
    at t = 0.5, 1.5, ..., 14.5 — natural one-frame-per-second cadence with
    no endpoint-seek artefacts. Falls back to even spacing across duration
    if num_frames * (1/fps) > duration. Output: frame_00.jpg .. frame_NN.jpg.
    """
    clips = json.load(open(clips_file))
    output_dir = Path(output_dir)

    extracted = 0
    missing = 0
    for clip_id in clips:
        mp4_path = Path(clips_dir) / f"{clip_id}.mp4"
        if not mp4_path.exists():
            missing += 1
            continue

        out_dir = output_dir / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(list(out_dir.glob("frame_*.jpg"))) >= num_frames:
            extracted += 1
            continue

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
            capture_output=True, text=True
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = 15.0

        spacing = 1.0 / fps
        if num_frames * spacing <= duration:
            # 1fps midpoint sampling: 0.5, 1.5, ..., (num_frames - 0.5)
            offsets = [(i + 0.5) * spacing for i in range(num_frames)]
        else:
            # Clip shorter than expected — even-space across duration with
            # midpoint sampling so neither endpoint is at exact 0 or duration.
            offsets = [(i + 0.5) * duration / num_frames for i in range(num_frames)]

        for i, t in enumerate(offsets):
            frame_path = out_dir / f"frame_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", f"{t:.3f}", "-i", str(mp4_path),
                 "-frames:v", "1", "-q:v", "2", str(frame_path),
                 "-y", "-loglevel", "error"],
                capture_output=True
            )

        actual = len(list(out_dir.glob("frame_*.jpg")))
        if actual >= num_frames:
            extracted += 1
        else:
            print(f"  WARNING: {clip_id} only got {actual}/{num_frames} frames")

    print(f"Extracted: {extracted}, Missing mp4: {missing}")


# ---------------------------------------------------------------------------
# DeepInfra runner
# ---------------------------------------------------------------------------
def run_deepinfra(num_frames, frames_dir=None):
    from openai import OpenAI
    from tqdm import tqdm

    api_key = os.getenv("DeepInfra_API_KEY_2")
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    clips = json.load(open(CLIPS_FILE))
    manifest = load_manifest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"gemma4_31b_{num_frames}f.jsonl"

    # Resume
    done = set()
    if output_path.exists():
        for line in open(output_path):
            try:
                r = json.loads(line.strip())
                if r.get("success") and r.get("clip_id"):
                    done.add(r["clip_id"])
            except:
                pass

    remaining = [c for c in clips if c not in done]
    print(f"Gemma {num_frames}f: {len(done)} done, {len(remaining)} remaining")

    if not remaining:
        print("All done!")
        return

    success = 0
    fail = 0

    with open(output_path, "a") as out_f:
        for clip_id in tqdm(remaining, desc=f"gemma_{num_frames}f"):
            system_prompt, user_prompt = build_prompt(clip_id, manifest, num_frames)
            b64_images = get_base64_images(clip_id, num_frames, frames_dir)

            if not b64_images:
                fail += 1
                record = {
                    "clip_id": clip_id, "variant": f"{num_frames}f",
                    "model": "google/gemma-4-31B-it", "success": False,
                    "error": "no_frames", "timestamp": datetime.now().isoformat(),
                }
                out_f.write(json.dumps(record) + "\n")
                continue

            content = []
            for data_url in b64_images:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            content.append({"type": "text", "text": user_prompt})

            try:
                response = client.chat.completions.create(
                    model="google/gemma-4-31B-it",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=8000,
                    temperature=0.1,
                )

                raw_text = response.choices[0].message.content or ""
                parsed = parse_json_response(raw_text)
                normalized = None
                ok = False

                if parsed:
                    parsed = validate_and_fix_response(parsed) or parsed
                    parsed, _ = validate_response_completeness(parsed)
                    ok = True
                    try:
                        normalized = normalize_annotation(parsed)
                    except:
                        normalized = None

                record = {
                    "clip_id": clip_id, "variant": f"{num_frames}f",
                    "model": "google/gemma-4-31B-it", "success": ok,
                    "parsed": parsed, "normalized": normalized,
                    "raw_response": (raw_text or "")[:8000],
                    "usage": {
                        "input": response.usage.prompt_tokens if response.usage else 0,
                        "output": response.usage.completion_tokens if response.usage else 0,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                if ok:
                    success += 1
                else:
                    fail += 1

            except Exception as e:
                fail += 1
                record = {
                    "clip_id": clip_id, "variant": f"{num_frames}f",
                    "model": "google/gemma-4-31B-it", "success": False,
                    "error": str(e)[:500], "timestamp": datetime.now().isoformat(),
                }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"Done: {success} ok, {fail} fail. Output: {output_path}")


# ---------------------------------------------------------------------------
# OpenAI batch input generator
# ---------------------------------------------------------------------------
def generate_openai_batch_input(num_frames):
    clips = json.load(open(CLIPS_FILE))
    manifest = load_manifest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"gpt4o_{num_frames}f_batch_input.jsonl"

    count = 0
    with open(output_path, "w") as f:
        for clip_id in clips:
            system_prompt, user_prompt = build_prompt(clip_id, manifest, num_frames)
            image_urls = get_s3_urls(clip_id, num_frames)

            content = []
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({"type": "text", "text": user_prompt})

            import re
            custom_id = re.sub(r'[^a-zA-Z0-9_-]', '_', clip_id)[:64]

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    "max_tokens": 8000,
                    "temperature": 0.1,
                },
            }
            f.write(json.dumps(request) + "\n")
            count += 1

    print(f"Generated {count} batch requests: {output_path}")
    return output_path


def submit_openai_batch(num_frames):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path = RESULTS_DIR / f"gpt4o_{num_frames}f_batch_input.jsonl"
    if not input_path.exists():
        print("Generate batch input first: --batch-input")
        return

    print(f"Uploading {input_path}...")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"File: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"SteelBench frame density {num_frames}f"},
    )
    print(f"Batch: {batch.id}, status: {batch.status}")

    state_file = RESULTS_DIR / f"gpt4o_{num_frames}f_batch_id.txt"
    with open(state_file, "w") as f:
        f.write(batch.id)
    print(f"Batch ID saved to {state_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Frame density ablation")
    parser.add_argument("--provider", choices=["deepinfra", "openai"])
    parser.add_argument("--frames", type=int, choices=[1, 2, 4, 8, 15])
    parser.add_argument("--frames-dir", default=None, help="Override frames directory")
    parser.add_argument("--batch-input", action="store_true")
    parser.add_argument("--batch-submit", action="store_true")
    parser.add_argument("--extract-15", action="store_true",
                        help="Extract 15 frames at 1fps (midpoint sampling)")
    parser.add_argument("--clips-dir", default=None, help="Directory with mp4 clips")
    parser.add_argument("--output-dir", default=None, help="Output dir for 15-frame extraction")
    args = parser.parse_args()

    load_dotenv()

    if args.extract_15:
        clips_dir = args.clips_dir or "active_batch/clips"
        output_dir = args.output_dir or "eval_data/frames_15"
        extract_dense_frames(clips_dir, output_dir, CLIPS_FILE, num_frames=15, fps=1.0)
        return

    if args.provider == "deepinfra":
        frames_dir = Path(args.frames_dir) if args.frames_dir else None
        run_deepinfra(args.frames, frames_dir)
    elif args.provider == "openai":
        if args.batch_input:
            generate_openai_batch_input(args.frames)
        elif args.batch_submit:
            submit_openai_batch(args.frames)
        else:
            print("For OpenAI, use --batch-input then --batch-submit")


if __name__ == "__main__":
    main()
