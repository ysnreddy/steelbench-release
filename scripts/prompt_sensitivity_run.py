#!/usr/bin/env python3
"""Prompt sensitivity ablation runner.

Runs V2 (minimal/open-ended) and V3 (structured observation) on 150 stratified clips
for 3 models: GPT-4o (OpenAI batch), Gemma 4-31B (DeepInfra), Qwen 3.5-122B (DeepInfra).

V1 results already exist from main evaluation — extracted separately.

Usage:
    # Run DeepInfra models (Gemma + Qwen) real-time
    python scripts/prompt_sensitivity_run.py --provider deepinfra --variant v2
    python scripts/prompt_sensitivity_run.py --provider deepinfra --variant v3

    # Generate OpenAI batch input files
    python scripts/prompt_sensitivity_run.py --provider openai --variant v2 --batch-input
    python scripts/prompt_sensitivity_run.py --provider openai --variant v3 --batch-input

    # Submit OpenAI batch
    python scripts/prompt_sensitivity_run.py --provider openai --variant v2 --batch-submit
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation_tool.vlm_annotator import (
    normalize_annotation,
    parse_json_response,
    validate_and_fix_response,
    validate_response_completeness,
)
from annotation_tool.safety_rules import build_safety_prompt_block

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent.parent / "eval_data"
RESULTS_DIR = EVAL_DIR / "results" / "prompt_sensitivity"
FRAMES_DIR = EVAL_DIR / "frames"
CONFIG_DIR = EVAL_DIR / "config"
CLIPS_FILE = EVAL_DIR / "prompt_sensitivity_clips.json"

S3_BUCKET = "steelbench-eval-frames"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

V2_SYSTEM = """You are analyzing surveillance footage from an industrial facility. The camera is fixed and mounted at a distance, so some small details may be unclear. Describe only what is visible. Do not force a specific action or safety conclusion when the visual evidence is ambiguous."""

V2_USER = """These are 8 frames sampled evenly from a 15-second surveillance clip in an industrial steel plant.
Analyze the scene using open descriptions. Do NOT use a predefined action taxonomy. Describe what each visible worker appears to be doing in natural language.
Return ONLY valid JSON in the schema below.
Tasks:
1. Count all visible workers across the 8 frames.
2. For each worker, provide:
   - A short physical description so the person can be identified across frames.
   - Their approximate position in the scene.
   - A free-text description of what they appear to be doing.
   - Any visible tool, object, machine, vehicle, panel, load, or equipment associated with that worker.
   - Their visible PPE status.
   - Whether any visible safety concern is present.
   - The reason for the safety concern if one is present.
3. Assess only visible evidence. For small or unclear PPE items such as safety shoes, use "cannot_determine" when the item is not clearly visible.
4. For safety assessment:
   - Use "none" when no visible unsafe act is present.
   - Use "cannot_determine" when the visual evidence is insufficient.
   - Do not infer a violation only from the type of industrial area unless the unsafe condition is visible.
5. Identify visibility conditions in the scene.
6. Classify the scene type:
   - SA: exactly one visible worker.
   - MAC: two or more workers visibly coordinating on the same task, object, equipment, or signal.
   - MAI: two or more workers present but doing independent activities. Use MAI as the default for multi-worker scenes unless coordination is clearly visible.
JSON schema:
{
  "scene_type": "SA|MAI|MAC",
  "total_workers": 0,
  "visibility_conditions": ["clear|dust|smoke|steam|low_light|glare"],
  "scene_description": "",
  "workers": [
    {
      "id": 1,
      "position": "",
      "physical_description": "",
      "action_description": "",
      "tool_or_equipment": "",
      "confidence": 0.0,
      "ppe": {
        "helmet": "worn|not_worn|cannot_determine",
        "high_vis_vest": "worn|not_worn|cannot_determine",
        "welding_protection": "worn|not_worn|cannot_determine|not_applicable",
        "harness": "worn|not_worn|cannot_determine|not_applicable",
        "safety_shoes": "worn|not_worn|cannot_determine"
      },
      "spatial_context_description": "",
      "unsafe_act": "none|cannot_determine|description of visible unsafe act",
      "safety_reason": "",
      "occlusion_level": "fully_visible|partially_occluded|heavily_occluded"
    }
  ]
}"""

V3_SYSTEM = """You are a trained safety officer reviewing surveillance footage from an integrated steel plant. The camera is fixed and mounted at a distance above the work area. Use the required analysis order internally, but output only the final JSON. Base all conclusions on visible evidence. When evidence is unclear, use "cannot_determine" rather than guessing."""

V3_USER = """These are 8 frames sampled evenly from a 15-second surveillance clip in an industrial steel plant.
Use the following analysis order internally before producing the final JSON:
STEP 1 — Observe the scene:
Identify the work area, visible equipment, lighting, visibility conditions, and any dust, smoke, steam, glare, or low-light conditions.
STEP 2 — Count and locate workers:
Identify every visible worker across the 8 frames. For each worker, provide a physical description and position so that a human reviewer can identify the same person.
STEP 3 — Describe before classifying:
For each worker, first describe what the worker is physically doing in free text. Then classify the worker's primary action using the SteelBench action taxonomy below.
SteelBench action taxonomy:
GROUP A — LOCOMOTION:
A1 walking, A2 climbing_up, A3 climbing_down, A4 carrying_walking, A5 pushing_pulling
GROUP B — STATIONARY WORK:
B1 standing_operation_tool_use, B2 standing_panel_operation, B3 hot_work_welding, B4 hot_work_non_welding, B5 crouching_floor_work, B6 overhead_reaching, B7 inspection_stationary, B8 lifting_placing, B9 sitting_operation
GROUP C — CRANE AND EQUIPMENT:
C1 crane_signalling, C2 crane_hook_interaction, C3 crane_guiding_load, C4 vehicle_operating
GROUP D — MATERIAL HANDLING:
D1 manual_team_carry, D2 loading_unloading, D3 cylinder_handling
GROUP E — SOCIAL:
E1 communicating, E2 supervising
GROUP F — IDLE:
F1 idle_standing, F2 idle_sitting
GROUP X — UNLISTED:
X1 unlisted_action
Use X1 only when the worker's action clearly does not fit the listed classes. Do not use X1 merely because the image is unclear. If the image is unclear but the closest class is still plausible, select the closest class and lower the confidence.
STEP 4 — Assess PPE:
For each worker, assess only visible PPE: helmet, high_vis_vest, welding_protection, harness, safety_shoes.
Use "worn" if visibly present, "not_worn" if clearly absent, "cannot_determine" if too small/occluded/unclear, "not_applicable" only for irrelevant PPE.
STEP 5 — Apply safety-rule reasoning:
Using the worker's action, location, PPE, and visible hazards, determine whether a visible unsafe act is present.
Use the applicable safety-rule context provided for this clip. Do not invent rules. If a rule may apply but the visual evidence is insufficient, use "cannot_determine" and explain why.
STEP 6 — Check coordination:
solo / concurrent / coordinated. When uncertain, choose concurrent.
STEP 7 — Check action transition:
If a worker clearly changes action across the 8 frames, record it.

{safety_block}

Return ONLY valid JSON in the schema below:
{{
  "scene_type": "SA|MAI|MAC",
  "total_workers": 0,
  "visibility": "clear|moderate|harsh",
  "visibility_conditions": ["clear|dust|smoke|steam|low_light|glare"],
  "visible_equipment": "",
  "scene_description": "",
  "group_activity": "",
  "workers": [
    {{
      "id": 1,
      "position": "",
      "physical_description": "",
      "free_text_description": "",
      "action_class": "A1|...|X1",
      "action_group": "GROUP A|...|GROUP X",
      "confidence": 0.0,
      "tool_or_equipment": "",
      "spatial_context": [],
      "ppe": {{
        "helmet": "worn|not_worn|cannot_determine",
        "high_vis_vest": "worn|not_worn|cannot_determine",
        "welding_protection": "worn|not_worn|cannot_determine|not_applicable",
        "harness": "worn|not_worn|cannot_determine|not_applicable",
        "safety_shoes": "worn|not_worn|cannot_determine"
      }},
      "occlusion_level": "fully_visible|partially_occluded|heavily_occluded",
      "occlusion_source": [],
      "unsafe_act": "none|cannot_determine|description",
      "safety_rule": "",
      "safety_reason": "",
      "group_flag": "solo|coordinated|concurrent"
    }}
  ],
  "action_transition": {{
    "detected": false,
    "transitions": []
  }}
}}
Scene type rules:
1. SA: exactly one worker visible.
2. MAC: two or more workers visibly coordinating on a shared task.
3. MAI: two or more workers present but acting independently (DEFAULT)."""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DEEPINFRA_MODELS = {
    "gemma4_31b": {
        "model": "google/gemma-4-31B-it",
        "api_key_env": "DeepInfra_API_KEY_2",
    },
    "qwen35_122b": {
        "model": "Qwen/Qwen3.5-122B-A10B",
        "api_key_env": "DeepInfra_API_KEY_3",
        "disable_thinking": True,
    },
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


def get_s3_image_urls(clip_id):
    from urllib.parse import quote
    urls = []
    for i in range(8):
        urls.append(f"{S3_BASE_URL}/{quote(clip_id, safe='')}/frame_{i}.jpg")
    return urls


# VPS local frames path
VPS_FRAMES_DIR = Path("/opt/steelbench/active_batch/frames")


def get_base64_images(clip_id, max_images=8):
    """Load frames from local disk and return as base64 data URLs.

    If max_images < 8, selects evenly spaced frames.
    """
    frame_dir = VPS_FRAMES_DIR / clip_id
    if not frame_dir.exists():
        frame_dir = FRAMES_DIR / clip_id
    all_frames = sorted(frame_dir.glob("frame_*.jpg"))

    # Subsample if needed
    if max_images < len(all_frames):
        indices = [int(i * (len(all_frames) - 1) / (max_images - 1)) for i in range(max_images)]
        all_frames = [all_frames[i] for i in indices]

    images = []
    for f in all_frames:
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        images.append(f"data:image/jpeg;base64,{b64}")
    return images


def build_prompt(variant, clip_id, manifest):
    safety_block = get_safety_block(clip_id, manifest) if variant == "v3" else ""

    if variant == "v2":
        return V2_SYSTEM, V2_USER
    elif variant == "v3":
        user = V3_USER.format(safety_block=safety_block if safety_block else "No site-specific safety rules available for this clip.")
        return V3_SYSTEM, user
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# DeepInfra runner
# ---------------------------------------------------------------------------
def run_deepinfra(variant, model_name):
    from openai import OpenAI

    config = DEEPINFRA_MODELS[model_name]
    api_key = os.getenv(config["api_key_env"])
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    clips = json.load(open(CLIPS_FILE))
    manifest = load_manifest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_name}_{variant}.jsonl"

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
    print(f"{model_name} {variant}: {len(done)} done, {len(remaining)} remaining")

    if not remaining:
        print("All done!")
        return

    from tqdm import tqdm
    success = 0
    fail = 0

    with open(output_path, "a") as out_f:
        for clip_id in tqdm(remaining, desc=f"{model_name}_{variant}"):
            system_prompt, user_prompt = build_prompt(variant, clip_id, manifest)

            # Use base64 from local frames (DeepInfra limits URL-based images)
            # Qwen 3.5 on DeepInfra: max 4 images (provider constraint)
            max_img = 4 if "qwen" in model_name.lower() else 8
            b64_images = get_base64_images(clip_id, max_images=max_img)

            content = []
            for data_url in b64_images:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            content.append({"type": "text", "text": user_prompt})

            extra = {}
            if config.get("disable_thinking"):
                extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            try:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=8000,
                    temperature=0.1,
                    **extra,
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
                    "clip_id": clip_id,
                    "variant": variant,
                    "model": config["model"],
                    "success": ok,
                    "parsed": parsed,
                    "normalized": normalized,
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
                    tqdm.write(f"  PARSE FAIL: {clip_id}")

            except Exception as e:
                fail += 1
                tqdm.write(f"  ERROR: {clip_id}: {e}")
                record = {
                    "clip_id": clip_id,
                    "variant": variant,
                    "model": config["model"],
                    "success": False,
                    "error": str(e)[:500],
                    "timestamp": datetime.now().isoformat(),
                }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"Done: {success} ok, {fail} fail. Output: {output_path}")


# ---------------------------------------------------------------------------
# OpenAI batch input generator
# ---------------------------------------------------------------------------
def generate_openai_batch_input(variant):
    clips = json.load(open(CLIPS_FILE))
    manifest = load_manifest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"gpt4o_{variant}_batch_input.jsonl"

    count = 0
    with open(output_path, "w") as f:
        for clip_id in clips:
            system_prompt, user_prompt = build_prompt(variant, clip_id, manifest)
            image_urls = get_s3_image_urls(clip_id)

            content = []
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            content.append({"type": "text", "text": user_prompt})

            request = {
                "custom_id": clip_id.replace("&", "_").replace("#", "_")[:64],
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


def submit_openai_batch(variant):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path = RESULTS_DIR / f"gpt4o_{variant}_batch_input.jsonl"
    if not input_path.exists():
        print(f"Generate batch input first: --batch-input")
        return

    # Upload file
    print(f"Uploading {input_path}...")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"File uploaded: {uploaded.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"SteelBench prompt sensitivity {variant}"},
    )
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.status}")

    # Save batch ID
    state_file = RESULTS_DIR / f"gpt4o_{variant}_batch_id.txt"
    with open(state_file, "w") as f:
        f.write(batch.id)
    print(f"Batch ID saved to {state_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prompt sensitivity ablation")
    parser.add_argument("--provider", choices=["deepinfra", "openai"], required=True)
    parser.add_argument("--variant", choices=["v2", "v3"], required=True)
    parser.add_argument("--model", default=None, help="Specific DeepInfra model (gemma4_31b or qwen35_122b)")
    parser.add_argument("--batch-input", action="store_true", help="Generate OpenAI batch input")
    parser.add_argument("--batch-submit", action="store_true", help="Submit OpenAI batch")
    args = parser.parse_args()

    load_dotenv()

    if args.provider == "deepinfra":
        models = [args.model] if args.model else list(DEEPINFRA_MODELS.keys())
        for m in models:
            run_deepinfra(args.variant, m)

    elif args.provider == "openai":
        if args.batch_input:
            generate_openai_batch_input(args.variant)
        elif args.batch_submit:
            submit_openai_batch(args.variant)
        else:
            print("For OpenAI, use --batch-input then --batch-submit")


if __name__ == "__main__":
    main()
