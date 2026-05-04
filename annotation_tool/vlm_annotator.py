#!/usr/bin/env python3
"""Single-VLM pre-annotation runner for SteelBench.

Sends 8 representative frames per clip to a VLM for structured annotation prefill
with the unified evaluation prompt. Collects structured JSON annotations.

Usage:
    python -m annotation_tool.vlm_annotator --single-vlm b --vlm-b openrouter_qwen --limit 50
    python -m annotation_tool.vlm_annotator --single-vlm b --vlm-b openrouter_qwen --limit 247 --offset 0 --output batch_a.jsonl

Environment variables:
    OPENROUTER_API_KEY   - OpenRouter API key (primary)
    OPENROUTER_API_KEY_2 - OpenRouter API key (for parallel runs)
"""

import argparse
import base64
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.agreement import compute_clip_agreement

# ---------- Configuration ----------

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
VLM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results")
TIER_A_MANIFEST = os.path.join(OUTPUT_DIR, "metadata", "tier_a_manifest.csv")

# VLM model configs
VLM_CONFIGS = {
    "openai": {
        "base_url": None,  # Default OpenAI
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "response_format": {"type": "json_object"},  # Enforce JSON output
    },
    "local_qwen": {
        "model_path": "/<MODEL_DIR>/qwen_3.5-9B",
        "model": "Qwen3.5-VL-9B-local",
        "backend": "local",
    },
    "openrouter_qwen": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.5-35b-a3b",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openrouter_qwen36plus": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.6-plus",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepinfra_qwen": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    "deepinfra_qwen_235b": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "api_key_env": "DEEPINFRA_API_KEY",
        "disable_thinking": True,
    },
    "openrouter_gemma": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemma-4-31b-it",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openrouter_nemotron": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openrouter_llama": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

# Unified evaluation prompt (addendum v2 Section 3.1)
SYSTEM_PROMPT = """You are a trained safety officer reviewing surveillance footage from an \
integrated steel plant. The camera is mounted 7-10 meters above the work area. The footage is \
from a fixed CCTV camera. Your job is to identify what workers are doing AND whether any of \
them are violating safety rules specific to this site or the plant-wide general rules. \
A human annotator will review your output to verify and correct it, so be detailed, accurate, \
and conservative — when a safety rule may apply but you're unsure, cite it rather than omit it."""

USER_PROMPT = """These are 8 frames sampled evenly from a 15-second surveillance clip in an \
industrial steel plant. Analyze the COMPLETE scene and every visible worker.

1. IDENTIFY all visible workers. For EACH worker provide:
   - A brief PHYSICAL DESCRIPTION so a human reviewer can identify which person you mean \
across all 8 frames (e.g., "worker in blue shirt and yellow helmet, stocky build" or \
"person in dark clothing near left edge"). Mention clothing color, headwear, build, or \
any distinguishing feature.
   - Their POSITION in the scene (e.g., "left foreground near conveyor", "center background \
on elevated platform", "right midground beside crane").

2. CLASSIFY each worker's PRIMARY action from this taxonomy (27 classes):
   GROUP A - LOCOMOTION: walking (A1), climbing_up (A2), climbing_down (A3), \
carrying_walking (A4), pushing_pulling (A5)
   GROUP B - STATIONARY WORK: standing_operation_tool_use (B1), standing_panel_operation (B2), \
hot_work_welding (B3), hot_work_non_welding (B4), crouching_floor_work (B5), \
overhead_reaching (B6), inspection_stationary (B7), lifting_placing (B8), sitting_operation (B9)
   GROUP C - CRANE & EQUIPMENT: crane_signalling (C1), crane_hook_interaction (C2), \
crane_guiding_load (C3), vehicle_operating (C4)
   GROUP D - MATERIAL HANDLING: manual_team_carry (D1), loading_unloading (D2), \
cylinder_handling (D3)
   GROUP E - SOCIAL: communicating (E1), supervising (E2)
   GROUP F - IDLE: idle_standing (F1), idle_sitting (F2)
   GROUP X - UNLISTED ACTION (X1): Use ONLY when the action clearly does not fit ANY of \
the 27 classes above. Do NOT force-fit an ambiguous action into a standard class — if you \
are unsure, prefer X1 with closest_existing_class filled. The human annotator will make the \
final decision. When using X1, you MUST fill ALL of these fields:
     - closest_existing_class: which of the 27 classes is closest (e.g., "B1")
     - why_not_existing: why it doesn't fit (1 sentence)
     - free_text_description: what the worker is actually doing (1-2 sentences)
     - tool_or_equipment: tool or equipment involved (or "" if none)
     - industry_specific: true if this is a steel-plant-specific action, false otherwise

3. For EACH worker, provide free_text_description (MANDATORY, even for standard classes): \
1-2 sentences describing what the worker is physically doing. This is essential for human \
annotators to verify your action classification. Example: "Worker crouching beside a large \
motor, using a wrench to tighten bolts on the base plate."

3b. For EACH worker, populate tool_or_equipment whenever a tool, hand-held object, or piece \
of equipment is visible in or near their hands — even if tangential. Examples: "wrench", \
"hammer", "gas cylinder", "mobile phone", "clipboard", "hose", "metal rod", "control lever". \
Use "" only if the worker is clearly empty-handed AND not operating anything. Do NOT leave \
this empty just because the item is common.

4. For each worker, classify spatial_context using one or more tags from this list ONLY: \
ground_level, elevated, vehicle_area, crane_zone, furnace_zone, confined_space, restricted, \
near_machinery, walkway, storage_area, locomotive_area, hot_area. Do NOT invent new tags.

5. For each worker, assess VISIBLE PPE: helmet, high_vis_vest, welding_protection, \
harness, safety_shoes. Values: worn / not_worn / cannot_determine / not_applicable. \
("worn" = item is present and being used; "not_worn" = item is absent or clearly not in use.)

6. COORDINATION — for each worker set group_flag to one of:
   - "solo": only worker in the scene OR working alone with no other workers visibly nearby. \
This is the DEFAULT for single-worker scenes.
   - "coordinated": actively working together with one or more SPECIFIC other workers on a \
SHARED task (e.g., team carry of one object, one welder + one holder, crane signaller + \
crane operator). MUST also fill coordinated_with with the specific worker IDs they are \
working with, and role_in_coordination explaining their role.
   - "concurrent": multiple workers in the same area but each doing their own independent \
task with no coordination (e.g., walking past each other, working at separate stations).
   When in doubt, prefer "concurrent" over "coordinated". Coordination requires evidence of \
shared activity, not just proximity.

7. UNSAFE ACTS — for each worker note any visible safety concern. Examples: "no_helmet", \
"near_moving_equipment_without_vest", "improper_lifting_posture", "working_at_height_no_harness". \
Use "none" if no concern visible. Look if the worker is violating any safety protocols followed in steel plants. Do NOT leave this field empty.

8. ACTION TRANSITIONS — if a worker's action CHANGES across the 8 frames, report it with \
worker_id, from_action, to_action, and which frames (e.g., "frames 1-4 to frames 5-8").

9. OCCLUSION — for each worker:
   - occlusion_level: fully_visible / partially_occluded / heavily_occluded
   - occlusion_source (if occluded): equipment / structure / other_person / atmospheric / frame_edge

10. VISIBILITY CONDITIONS for the scene: one or more of clear / steam / dust / smoke / low_light / glare.

Respond ONLY with valid JSON matching this schema (see SCENE TYPE rules below for which value to use):
{
  "scene_type": "SA|MAI|MAC|SEQ",
  "total_workers": 0,
  "visibility": "clear|moderate|harsh",
  "visibility_conditions": ["clear"],
  "visible_equipment": "",
  "scene_description": "1-2 sentence overview of the scene",
  "group_activity": "description if MAC, else empty",
  "workers": [
    {
      "id": 1,
      "position": "location in scene (e.g. left foreground near conveyor)",
      "physical_description": "clothing, build, distinguishing features",
      "action_class": "one of the 27 class names or unlisted_action",
      "action_group": "GROUP A|B|C|D|E|F|X",
      "confidence": 0.85,
      "free_text_description": "MANDATORY: what the worker is physically doing",
      "closest_existing_class": "",
      "why_not_existing": "",
      "unlisted_description": "",
      "tool_or_equipment": "",
      "industry_specific": false,
      "spatial_context": ["ground_level"],
      "ppe": {
        "helmet": "worn|not_worn|cannot_determine",
        "high_vis_vest": "worn|not_worn|cannot_determine",
        "welding_protection": "worn|not_worn|cannot_determine|not_applicable",
        "harness": "worn|not_worn|cannot_determine|not_applicable",
        "safety_shoes": "worn|not_worn|cannot_determine"
      },
      "occlusion_level": "fully_visible|partially_occluded|heavily_occluded",
      "occlusion_source": [],
      "unsafe_act": "none or description of safety concern",
      "group_flag": "solo|coordinated|concurrent",
      "coordinated_with": [],
      "role_in_coordination": ""
    }
  ],
  "action_transition": {
    "detected": false,
    "transitions": []
  }
}

SCENE TYPE — apply these rules in order, pick the FIRST that matches:

1. SA (Single Actor): Exactly ONE worker visible in the scene. If total_workers == 1, scene_type \
MUST be "SA". Do NOT use MAI or MAC for single-worker clips.

2. SEQ (Sequential action change): The same worker(s) clearly switch from one action to a \
different action across the 8 frames (e.g., walking → standing → operating tool). The action \
itself changes, not just position. Only use SEQ if action_transition.detected is true.

3. MAC (Multi-Actor Coordinated): Two or more workers performing a SHARED task with VISIBLE \
coordination — they are working on the SAME object/equipment together, OR one is signalling \
to another (e.g., crane signaller + crane operator), OR they are doing complementary roles \
(e.g., one holds material while another welds, team carry of one large object). Mere proximity, \
similar activity, or being in the same area is NOT coordination.

4. MAI (Multi-Actor Independent): Two or more workers each doing their own task in the same \
scene without coordinating with each other. This is the DEFAULT for multi-worker scenes \
unless you can clearly identify shared activity. Examples: multiple workers each at their own \
station, workers walking past each other, several workers idle in different positions.

CRITICAL: When in doubt between MAC and MAI, prefer MAI. MAC requires evidence of \
coordination, not just multiple workers in the frame. Most multi-worker industrial scenes \
are MAI, not MAC."""


def encode_image_base64(image_path):
    """Encode an image file to base64 string at original resolution.

    Do NOT resize — VLMs handle resolution internally:
    - GPT-4o: scales shortest side to 768px + 512x512 tiling (detail:high)
    - Qwen2.5-VL: dynamic resolution 480-2560px natively
    Resizing loses detail critical for distant workers at 7-10m surveillance distance.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(frame_paths, safety_block=None):
    """Build the API messages with representative frames as images.

    Args:
        frame_paths: List of image file paths
        safety_block: Optional safety rules text to append to the prompt.
                      Generated by safety_rules.build_safety_prompt_block().
    """
    prompt_text = USER_PROMPT
    if safety_block:
        prompt_text = USER_PROMPT + "\n\n" + safety_block
    content = [{"type": "text", "text": prompt_text}]

    for i, path in enumerate(frame_paths):
        if os.path.exists(path):
            b64 = encode_image_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_json_response(response_text):
    """Extract JSON from VLM response, handling markdown code blocks."""
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if '```' in pattern else match.group(0))
            except json.JSONDecodeError:
                continue

    return None


# PPE value normalization — maps common VLM variants to canonical values
_PPE_VALUE_MAP = {
    # present variants
    "yes": "present", "true": "present", "1": "present",
    "worn": "present", "visible": "present", "detected": "present",
    "wearing": "present",
    # absent variants
    "no": "absent", "false": "absent", "0": "absent",
    "not_worn": "absent", "missing": "absent", "not_visible": "absent",
    "not_wearing": "absent", "none": "absent",
    # indeterminate variants
    "unknown": "cannot_determine", "unsure": "cannot_determine",
    "unclear": "cannot_determine", "indeterminate": "cannot_determine",
    "partially_visible": "cannot_determine", "obscured": "cannot_determine",
    # not applicable
    "n/a": "not_applicable", "na": "not_applicable",
    # canonical values pass through
    "present": "present", "absent": "absent",
    "cannot_determine": "cannot_determine", "not_applicable": "not_applicable",
}

# Expected top-level keys in the unified prompt response
_EXPECTED_TOP_KEYS = {
    "scene_type", "total_workers", "workers", "persons",
    "action_transition", "group_activity", "scene_description",
}

# Expected per-worker keys
_EXPECTED_WORKER_KEYS = {
    "id", "position", "physical_description", "action_class", "action_code",
    "action_group", "confidence", "unlisted_description", "ppe",
    "spatial_context", "coordinated_with", "role_in_coordination",
    "occlusion_level", "occlusion_source", "unsafe_act", "group_flag",
    # X1 fields
    "closest_existing_class", "closest_existing_label",
    "why_not_existing", "free_text_description",
    "tool_or_equipment", "industry_specific", "taxonomy_layer",
}

# Required fields that must be present and non-empty in each worker
_REQUIRED_WORKER_FIELDS = {
    "position": "",
    "action_class": "",
    "confidence": 0.5,
    "ppe": {},
    "spatial_context": ["ground_level"],
    "free_text_description": "",
    "group_flag": "solo",
    "occlusion_level": "fully_visible",
    "unsafe_act": "none",
}


def validate_and_fix_response(parsed_json):
    """Validate and fix VLM response against expected schema.

    Handles common open-weight model issues:
    - Extra/hallucinated fields → stripped
    - Wrong PPE enum values → normalized
    - String instead of list for spatial_context → wrapped
    - Missing required fields → filled with defaults
    - workers vs persons key → unified

    Returns cleaned dict or None if unfixable.
    """
    if not isinstance(parsed_json, dict):
        return None

    result = {}

    # Scene type — normalize
    scene_type = str(parsed_json.get("scene_type", "SA")).upper().strip()
    # Fix common variants
    scene_type = scene_type.replace("-", "").replace("_", "").replace(" ", "")
    if scene_type in ("SINGLEACTOR", "SINGLE"):
        scene_type = "SA"
    elif scene_type in ("MULTIACTORINDEPENDENT", "MULTI", "MAI"):
        scene_type = "MAI"
    elif scene_type in ("MULTIACTORCOORDINATED", "MAC"):
        scene_type = "MAC"
    elif scene_type in ("SEQUENTIAL", "SEQ"):
        scene_type = "SEQ"
    result["scene_type"] = scene_type

    # Workers/persons list
    persons_raw = parsed_json.get("workers", parsed_json.get("persons", []))
    if isinstance(persons_raw, dict):
        # Some models return a single worker as dict instead of list
        persons_raw = [persons_raw]
    if not isinstance(persons_raw, list):
        persons_raw = []

    result["total_workers"] = parsed_json.get(
        "total_workers", parsed_json.get("num_workers", len(persons_raw))
    )

    # Validate and clean each worker
    cleaned_persons = []
    for p in persons_raw:
        if not isinstance(p, dict):
            continue

        worker = {}
        # Copy expected keys only (strip hallucinated fields)
        for k, v in p.items():
            k_lower = k.lower().replace(" ", "_")
            if k_lower in _EXPECTED_WORKER_KEYS or k in _EXPECTED_WORKER_KEYS:
                worker[k] = v

        # Ensure action_code or action_class exists
        if "action_code" not in worker and "action_class" not in worker:
            # Try to extract from other fields
            action = p.get("action", p.get("primary_action", ""))
            if action:
                worker["action_class"] = str(action)

        # Fix spatial_context — should be a list
        spatial = worker.get("spatial_context", [])
        if isinstance(spatial, str):
            worker["spatial_context"] = [spatial] if spatial else []

        # Fix PPE values — normalize to canonical enums
        ppe = worker.get("ppe", {})
        if isinstance(ppe, dict):
            cleaned_ppe = {}
            for item, value in ppe.items():
                item_lower = item.lower().replace(" ", "_")
                # Normalize item names
                if item_lower in ("hard_hat", "helmet", "safety_helmet"):
                    item_key = "helmet"
                elif item_lower in ("safety_vest", "high_vis_vest", "vest",
                                    "hi_vis_vest", "reflective_vest"):
                    item_key = "high_vis_vest"
                elif item_lower in ("welding_protection", "welding_mask",
                                    "face_shield", "welding_shield"):
                    item_key = "welding_protection"
                elif item_lower in ("harness", "safety_harness", "fall_protection"):
                    item_key = "harness"
                elif item_lower in ("safety_shoes", "safety_boots", "boots",
                                    "steel_toe", "footwear"):
                    item_key = "safety_shoes"
                elif item_lower in ("gloves", "safety_gloves"):
                    item_key = "gloves"
                else:
                    item_key = item_lower

                # Normalize value
                val_str = str(value).lower().strip()
                cleaned_ppe[item_key] = _PPE_VALUE_MAP.get(val_str, val_str)
            worker["ppe"] = cleaned_ppe

        # Fix coordinated_with — should be a list
        coord = worker.get("coordinated_with", [])
        if isinstance(coord, str):
            worker["coordinated_with"] = [coord] if coord else []
        elif isinstance(coord, (int, float)):
            worker["coordinated_with"] = [int(coord)]

        # Fix confidence — should be float 0-1
        conf = worker.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
                if conf > 1.0:
                    conf = conf / 100.0  # Convert percentage to fraction
                worker["confidence"] = round(min(max(conf, 0.0), 1.0), 3)
            except (ValueError, TypeError):
                worker["confidence"] = 0.5

        cleaned_persons.append(worker)

    result["workers"] = cleaned_persons

    # Action transition
    transition = parsed_json.get("action_transition", {})
    if isinstance(transition, dict):
        result["action_transition"] = transition
    else:
        result["action_transition"] = {"detected": False, "transitions": []}

    # Copy other expected top-level fields
    for k in ("group_activity", "scene_description", "visibility",
              "visibility_conditions", "visible_equipment"):
        if k in parsed_json:
            result[k] = parsed_json[k]

    return result


def validate_response_completeness(parsed_json):
    """Check that all required fields are present and non-empty in each worker.

    Fills defaults for missing fields and returns list of warnings.
    """
    warnings = []
    workers = parsed_json.get("workers", parsed_json.get("persons", []))
    for i, w in enumerate(workers):
        for field, default in _REQUIRED_WORKER_FIELDS.items():
            val = w.get(field)
            if val is None or val == "" or val == []:
                warnings.append(f"Worker {i+1}: missing '{field}'")
                w[field] = default
    return parsed_json, warnings


def call_vlm(client, model, frame_paths, max_retries=4,
             use_openai_params=False, response_format=None,
             disable_thinking=False, safety_block=None):
    """Call a VLM API with the representative frames.

    Returns (parsed_json, raw_response, success_bool)
    """
    messages = build_messages(frame_paths, safety_block=safety_block)

    # OpenAI GPT-4o+ requires max_completion_tokens; OpenRouter uses max_tokens
    if use_openai_params:
        token_kwargs = {"max_completion_tokens": 6000}
    else:
        token_kwargs = {"max_tokens": 6000}

    # Add response_format if supported (OpenAI JSON mode)
    if response_format:
        token_kwargs["response_format"] = response_format

    # Disable thinking mode for Qwen3 reasoning models served by DeepInfra/etc.
    extra_body = None
    if disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    for attempt in range(max_retries + 1):
        try:
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=0.1,
                **token_kwargs,
            )
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**create_kwargs)

            raw_text = response.choices[0].message.content
            # Some thinking models put output in reasoning_content if not disabled
            if not raw_text:
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                if reasoning:
                    raw_text = reasoning
            parsed = parse_json_response(raw_text)

            if parsed:
                # Validate and fix schema issues from open-weight models
                parsed = validate_and_fix_response(parsed) or parsed
                # Check field completeness and fill defaults
                parsed, field_warnings = validate_response_completeness(parsed)
                if field_warnings:
                    print(f"    [{len(field_warnings)} missing fields filled with defaults]",
                          end=" ", flush=True)
                return parsed, raw_text, True
            elif attempt < max_retries:
                time.sleep(2)
                continue
            else:
                return None, raw_text, False

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if attempt < max_retries:
                # Longer backoff for rate limits (15s, 30s, 45s, 60s)
                wait = (attempt + 1) * 15 if is_rate_limit else (attempt + 1) * 5
                print(f"    API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None, err_str, False

    return None, "", False


# ---------------------------------------------------------------------------
# Local Qwen inference
# ---------------------------------------------------------------------------

_local_model = None
_local_processor = None


def load_local_model(model_path):
    """Load Qwen VL model locally on GPU. Cached after first call."""
    global _local_model, _local_processor
    if _local_model is not None:
        return _local_model, _local_processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"  Loading local model from {model_path}...")
    _local_processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    _local_model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    import torch as _torch
    print(f"  Model loaded on {_local_model.device} "
          f"({_torch.cuda.memory_allocated()/1024**3:.1f}GB VRAM)")
    return _local_model, _local_processor


def call_vlm_local(model_path, frame_paths, max_retries=2):
    """Run inference on local Qwen VL model with representative frames.

    Returns (parsed_json, raw_response, success_bool) — same interface as call_vlm.
    """
    from PIL import Image

    model, processor = load_local_model(model_path)

    # Build conversation with images
    images = []
    image_content = []
    for path in frame_paths:
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
            image_content.append({"type": "image"})

    if not images:
        return None, "No valid frames", False

    # Build messages in Qwen chat format
    # Local models need stronger JSON enforcement
    local_system = SYSTEM_PROMPT + (
        " You MUST respond with ONLY a valid JSON object. "
        "No explanation, no markdown, no text before or after the JSON. "
        "Start your response with { and end with }."
    )
    local_user = USER_PROMPT + (
        "\n\nIMPORTANT: Output ONLY the JSON object. No thinking, no explanation, "
        "no markdown code blocks. Start directly with {"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": local_system}]},
        {"role": "user", "content": image_content + [{"type": "text", "text": local_user}]},
    ]

    for attempt in range(max_retries + 1):
        try:
            # Process inputs
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Skip thinking phase — replace <think> with </think> to get
            # direct JSON output. Qwen 3.5 chat template forces thinking
            # mode which outputs long reasoning before the actual answer.
            text_input = text_input.replace("<think>\n", "</think>\n")

            inputs = processor(
                text=text_input, images=images, return_tensors="pt"
            ).to(model.device)

            # Generate
            import torch
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    temperature=0.1,
                    do_sample=True,
                )

            # Decode — only the generated tokens (skip input)
            input_len = inputs["input_ids"].shape[1]
            raw_text = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

            parsed = parse_json_response(raw_text)
            if parsed:
                parsed = validate_and_fix_response(parsed) or parsed
                return parsed, raw_text, True
            elif attempt < max_retries:
                print(f"    Local parse fail (attempt {attempt+1}), retrying...")
                continue
            else:
                return None, raw_text, False

        except Exception as e:
            if attempt < max_retries:
                print(f"    Local error (attempt {attempt+1}): {e}")
                continue
            return None, str(e), False

    return None, "", False


def normalize_annotation(parsed_json):
    """Normalize VLM output into consistent annotation format.

    Handles both old-format (persons key) and new unified prompt (workers key)
    with scene_type, taxonomy_layer, coordination, and action transitions.
    """
    if not parsed_json:
        return None

    from annotation_tool.agreement import normalize_action_code

    # Scene type
    scene_type = parsed_json.get("scene_type", "").upper().strip()

    persons_raw = parsed_json.get("persons", [])
    if not persons_raw:
        persons_raw = parsed_json.get("workers", [])

    persons = []
    for p in persons_raw:
        action_input = p.get("action_code", "") or p.get("action_class", "")
        action_code = normalize_action_code(action_input)

        # Determine taxonomy layer
        taxonomy_layer = p.get("taxonomy_layer", 1)
        if action_code == "X1":
            taxonomy_layer = 2

        # Normalize PPE keys (map alternative names to our 5-item set) AND
        # map VLM-raw values (present|absent) to UI enum (worn|not_worn).
        # Downstream agreement.py/schema_validator.py expect worn|not_worn.
        raw_ppe = p.get("ppe", {})
        _PPE_VAL_MAP = {
            "present": "worn", "absent": "not_worn",
            "worn": "worn", "not_worn": "not_worn",
            "cannot_determine": "cannot_determine",
            "not_applicable": "not_applicable",
        }
        def _map_ppe(v, default):
            return _PPE_VAL_MAP.get(v, default)
        ppe = {
            "helmet": _map_ppe(raw_ppe.get("helmet", raw_ppe.get("hard_hat")), "cannot_determine"),
            "high_vis_vest": _map_ppe(raw_ppe.get("high_vis_vest", raw_ppe.get("safety_vest")), "cannot_determine"),
            "welding_protection": _map_ppe(raw_ppe.get("welding_protection"), "not_applicable"),
            "harness": _map_ppe(raw_ppe.get("harness"), "not_applicable"),
            "safety_shoes": _map_ppe(raw_ppe.get("safety_shoes"), "cannot_determine"),
        }

        spatial = p.get("spatial_context", [])
        if isinstance(spatial, str):
            spatial = [spatial] if spatial else []

        # Unlisted action fields (when X1)
        unlisted_action = None
        if action_code == "X1":
            unlisted_action = {
                "closest_existing_class": p.get("closest_existing_class",
                    p.get("closest_existing_label", "")),
                "why_not_existing": p.get("why_not_existing", ""),
                "free_text_description": p.get("free_text_description",
                    p.get("unlisted_description", "")),
                "tool_or_equipment": p.get("tool_or_equipment", ""),
                "industry_specific": p.get("industry_specific", True),
            }

        # Occlusion fields
        occlusion_level = p.get("occlusion_level", "fully_visible")
        occlusion_source = p.get("occlusion_source", [])
        if isinstance(occlusion_source, str):
            occlusion_source = [occlusion_source] if occlusion_source else []

        person_dict = {
            "person_id": p.get("person_id", p.get("id", f"P{len(persons)+1}")),
            "position": p.get("position", ""),
            "physical_description": p.get("physical_description", ""),
            "action_code": action_code,
            "action_name": p.get("action_name", p.get("action_label", "")),
            "taxonomy_layer": taxonomy_layer,
            "confidence": p.get("confidence", 0.0),
            "spatial_context": spatial,
            "ppe": ppe,
            "occlusion_level": occlusion_level,
            "occlusion_source": occlusion_source,
            "unsafe_act": p.get("unsafe_act", ""),
            "interaction": p.get("interaction", "none_visible"),
            "interaction_targets": p.get("interaction_targets", []),
            "group_flag": p.get("group_flag", "solo"),
            "coordinated_with": p.get("coordinated_with", []),
            "role_in_coordination": p.get("role_in_coordination", ""),
            "free_text_description": p.get("free_text_description", ""),
            "tool_or_equipment": p.get("tool_or_equipment", ""),
        }
        if unlisted_action:
            person_dict["unlisted_action"] = unlisted_action
        persons.append(person_dict)

    # Action transitions
    raw_transition = parsed_json.get("action_transition", {})
    action_transition = {
        "detected": raw_transition.get("detected", False),
        "transitions": raw_transition.get("transitions", []),
    }

    # Visibility conditions (scene-level)
    visibility_conditions = parsed_json.get("visibility_conditions", [])
    if isinstance(visibility_conditions, str):
        visibility_conditions = [visibility_conditions] if visibility_conditions else []
    if not visibility_conditions:
        # Fall back from legacy "visibility" field
        vis = parsed_json.get("visibility", "clear")
        visibility_conditions = ["clear"] if vis == "clear" else [vis]

    num_workers = parsed_json.get("num_workers",
        parsed_json.get("total_workers", len(persons)))
    annotation_layer = 2 if num_workers <= 5 else 1

    # Post-processing fix for DeepInfra Qwen3-VL 1-worker SA bug.
    # The tightened prompt instructs "if total_workers == 1, scene_type MUST be SA"
    # but the model frequently emits MAI or SEQ. Force the rule. Idempotent.
    if num_workers == 1 and scene_type and scene_type != "SA":
        scene_type = "SA"
        if persons:
            persons[0]["group_flag"] = "solo"
            persons[0]["coordinated_with"] = []
            persons[0]["role_in_coordination"] = ""

    bonus_per_person = (annotation_layer == 1 and len(persons) > 0)

    # dominant_actions: union of per-person action codes (fallback X1 = unlisted_action)
    dominant_actions = sorted({p.get("action_code") for p in persons if p.get("action_code")})
    if not dominant_actions:
        dominant_actions = ["X1"]

    # overall_ppe_compliance: aggregate per-person PPE dicts into a scene-level verdict.
    # Values are UI enum (worn|not_worn|cannot_determine|not_applicable) after PPE map.
    def _overall_ppe(ps):
        if not ps:
            return "cannot_determine"
        vals = [v for p in ps for v in (p.get("ppe") or {}).values()]
        vals = [v for v in vals if v != "not_applicable"]
        if not vals:
            return "cannot_determine"
        absent = sum(1 for v in vals if v == "not_worn")
        worn = sum(1 for v in vals if v == "worn")
        if absent == 0 and worn > 0:
            return "compliant"
        if absent / len(vals) >= 0.5:
            return "non_compliant"
        if absent > 0:
            return "partial"
        return "cannot_determine"
    overall_ppe_compliance = _overall_ppe(persons)

    # annotator_confidence prefill: mean of per-person model confidences
    confs = [p.get("confidence") for p in persons if isinstance(p.get("confidence"), (int, float))]
    annotator_confidence = round(sum(confs) / len(confs), 3) if confs else 0.5

    return {
        "scene_type": scene_type,
        "num_workers": num_workers,
        "annotation_layer": annotation_layer,
        "bonus_per_person": bonus_per_person,
        "visibility": parsed_json.get("visibility", "clear"),
        "visibility_conditions": visibility_conditions,
        "visible_equipment": parsed_json.get("visible_equipment", ""),
        "persons": persons,
        "action_transition": action_transition,
        "group_activity": parsed_json.get("group_activity", ""),
        "scene_description": parsed_json.get("scene_description", ""),
        "dominant_actions": dominant_actions,
        "overall_ppe_compliance": overall_ppe_compliance,
        "annotator_confidence": annotator_confidence,
        # Backward-compatible flat fields
        "action_class": persons[0]["action_code"] if persons else "",
        "spatial_context": persons[0]["spatial_context"] if persons else [],
        "ppe": persons[0]["ppe"] if persons else {},
        "interaction": persons[0]["interaction"] if persons else "",
        "unsafe_act": persons[0]["unsafe_act"] if persons else "",
    }


def load_tier_a_clips(manifest_path, limit=0):
    """Load Tier A clips from manifest."""
    clips = []
    if not os.path.exists(manifest_path):
        return clips
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)
    if limit > 0:
        clips = clips[:limit]
    return clips


def load_existing_results(output_path=None):
    """Load already-processed clip IDs to avoid re-processing."""
    results_path = output_path or os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    processed = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get("clip_id", ""))
                except json.JSONDecodeError:
                    continue
    return processed


def save_result(result, output_path=None):
    """Append a single result to the JSONL file."""
    if output_path is None:
        os.makedirs(VLM_RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run VLM pre-annotation on clips")
    parser.add_argument("--vlm-a", type=str, default="openai",
                        choices=list(VLM_CONFIGS.keys()),
                        help="VLM-A config (default: openai/GPT-4o)")
    parser.add_argument("--vlm-b", type=str, default="local_qwen",
                        choices=list(VLM_CONFIGS.keys()),
                        help="VLM-B config (default: local_qwen)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of clips to process (default: 50)")
    parser.add_argument("--manifest", type=str, default=TIER_A_MANIFEST,
                        help="Path to Tier A manifest")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay between API calls in seconds (for rate limits)")
    parser.add_argument("--single-vlm", choices=["a", "b"], default=None,
                        help="Run only one VLM: 'a' for VLM-A only, 'b' for VLM-B only")
    parser.add_argument("--append-vlm", choices=["a", "b"], default=None,
                        help="Append a VLM to existing results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL filename (default: vlm_annotations.jsonl)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N clips in manifest (for parallel runs)")
    parser.add_argument("--api-key-env", type=str, default=None,
                        help="Override API key env var name (e.g., OPENROUTER_API_KEY_2)")
    parser.add_argument("--safety-enhanced", action="store_true",
                        help="Enable safety-enhanced prompt with rules from unsafe_acts.pdf "
                             "and per-clip site/zone context. Requires camera_zones.csv.")
    parser.add_argument("--health-check-interval", type=int, default=100,
                        help="Run health check every N clips (default: 100). "
                             "Checks field completeness and parse failure rate.")
    args = parser.parse_args()

    # Determine which VLMs are active
    run_a = args.single_vlm is None or args.single_vlm == "a"
    run_b = args.single_vlm is None or args.single_vlm == "b"

    # Load VLM configs (only for active VLMs)
    config_a = VLM_CONFIGS[args.vlm_a] if run_a else None
    config_b = VLM_CONFIGS[args.vlm_b] if run_b else None

    client_a = None
    client_b = None

    # Initialize active VLM clients
    if run_a and config_a:
        is_local_a = config_a.get("backend") == "local"
        if not is_local_a:
            key_a = os.environ.get(config_a["api_key_env"], "")
            if not key_a:
                print(f"Error: Set {config_a['api_key_env']} environment variable")
                sys.exit(1)
            client_a = OpenAI(api_key=key_a, base_url=config_a["base_url"])

    if run_b and config_b:
        is_local_b = config_b.get("backend") == "local"
        if not is_local_b:
            api_key_env = args.api_key_env or config_b["api_key_env"]
            key_b = os.environ.get(api_key_env, "")
            if not key_b:
                print(f"Error: Set {api_key_env} environment variable")
                sys.exit(1)
            client_b = OpenAI(api_key=key_b, base_url=config_b["base_url"])
        if is_local_b:
            load_local_model(config_b["model_path"])

    # Print active VLM(s)
    if args.single_vlm:
        active_config = config_a if args.single_vlm == "a" else config_b
        print(f"VLM: {active_config['model']} (single-VLM mode)")
    else:
        print(f"VLM A: {config_a['model']}")
        print(f"VLM B: {config_b['model']}")
    print()

    # Resolve output path
    output_path = None
    if args.output:
        output_path = os.path.join(VLM_RESULTS_DIR, args.output)

    # Load clips with offset support
    clips = load_tier_a_clips(args.manifest, limit=0)  # load all first
    if args.offset > 0:
        clips = clips[args.offset:]
    if args.limit > 0:
        clips = clips[:args.limit]
    if not clips:
        print(f"No clips found in {args.manifest}")
        print("Run extract_clips.py and filter_clips.py first.")
        sys.exit(1)

    # Handle append mode: add missing VLM to existing results
    if args.append_vlm:
        results_path = os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
        if not os.path.exists(results_path):
            print("No existing results to append to.")
            sys.exit(1)

        vlm_key = f"vlm_{args.append_vlm}"
        other_key = "vlm_a" if args.append_vlm == "b" else "vlm_b"
        append_config = config_a if args.append_vlm == "a" else config_b
        is_local_append = append_config.get("backend") == "local"

        # Load all existing records
        existing_records = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))

        # Find records that need the appended VLM
        needs_append = [r for r in existing_records
                        if r.get(other_key) and not r.get(vlm_key)]
        print(f"Append mode: adding VLM-{args.append_vlm.upper()} to {len(needs_append)} records")

        if args.limit > 0:
            needs_append = needs_append[:args.limit]

        if not needs_append:
            print("All records already have both VLMs.")
            sys.exit(0)

        # Build clip_id -> record index mapping
        record_index = {r["clip_id"]: i for i, r in enumerate(existing_records)}

        success_count = 0
        fail_count = 0
        agreement_scores = []

        for i, record in enumerate(needs_append):
            clip_id = record["clip_id"]
            # Find frames for this clip
            clip_entry = next((c for c in clips if c.get("clip_id") == clip_id), None)
            if not clip_entry:
                continue

            frames_dir = clip_entry.get("frames_dir", "")
            abs_frames_dir = os.path.join(OUTPUT_DIR, frames_dir)
            frame_paths = sorted(glob.glob(os.path.join(abs_frames_dir, "frame_*.jpg")))
            existing_frames = [p for p in frame_paths if os.path.exists(p)]
            if len(existing_frames) < 2:
                continue

            print(f"[{i+1}/{len(needs_append)}] {clip_id}...")
            print(f"  VLM {args.append_vlm.upper()} ({append_config['model']})...", end=" ", flush=True)

            if is_local_append:
                parsed, raw, success = call_vlm_local(
                    append_config["model_path"], existing_frames)
            else:
                parsed, raw, success = call_vlm(
                    client_a if args.append_vlm == "a" else client_b,
                    append_config["model"], existing_frames,
                    use_openai_params=(append_config.get("base_url") is None),
                    response_format=append_config.get("response_format"),
                    disable_thinking=append_config.get("disable_thinking", False),
                )
            print("OK" if success else "FAIL")

            norm = normalize_annotation(parsed) if parsed else None

            # Update the record
            idx = record_index[clip_id]
            existing_records[idx][vlm_key] = {
                "model": append_config["model"],
                "success": success,
                "parsed": parsed,
                "normalized": norm,
                "raw_response": raw[:5000] if isinstance(raw, str) else "",
            }

            # Compute agreement now that both VLMs exist
            norm_other = existing_records[idx].get(other_key, {}).get("normalized")
            norm_this = norm
            if norm_other and norm_this:
                if args.append_vlm == "a":
                    agreement = compute_clip_agreement(norm_this, norm_other)
                else:
                    agreement = compute_clip_agreement(norm_other, norm_this)
                existing_records[idx]["agreement"] = agreement
                existing_records[idx]["vlm"]["success"] = True
                agreement_scores.append(agreement["composite"])
                print(f"  Agreement: composite={agreement['composite']:.3f}")
            else:
                pass  # single VLM - no agreement needed

            if success:
                success_count += 1
            else:
                fail_count += 1

            if not is_local_append:
                time.sleep(args.delay)

        # Rewrite the entire JSONL with updated records
        with open(results_path, "w") as f:
            for r in existing_records:
                f.write(json.dumps(r) + "\n")

        print(f"\nAppend complete: {success_count} OK, {fail_count} failed")
        if agreement_scores:
            import numpy as np
            scores = np.array(agreement_scores)
            print(f"Agreement: mean={scores.mean():.3f}, median={np.median(scores):.3f}")
        print(f"Updated: {results_path}")
        sys.exit(0)

    # Normal mode: process new clips
    processed = load_existing_results(output_path)
    remaining = [c for c in clips if c.get("clip_id", "") not in processed]
    print(f"Total clips: {len(clips)}, Already processed: {len(processed)}, "
          f"Remaining: {len(remaining)}")

    if not remaining:
        print("All clips already processed.")
        sys.exit(0)

    # Load safety rules for safety-enhanced mode
    safety_rules_config = None
    camera_zones = None
    if args.safety_enhanced:
        from annotation_tool.safety_rules import (
            load_safety_rules, load_camera_zones, build_safety_prompt_block)
        safety_rules_config = load_safety_rules()
        camera_zones = load_camera_zones()
        print(f"Safety-enhanced mode: loaded {len(camera_zones)} camera zone mappings")
        if not camera_zones:
            print("⚠ No camera_zones.csv found — using general rules only for all clips")

    # Health check tracking
    health_window_success = 0
    health_window_fail = 0
    health_window_empty_fields = 0
    health_interval = args.health_check_interval

    def run_health_check(window_num, success, fail, empty_fields):
        """Print health check report every N clips."""
        total = success + fail
        if total == 0:
            return
        fail_rate = fail / total
        empty_rate = empty_fields / max(total, 1)
        status = "✓ OK" if fail_rate < 0.05 else "⚠ HIGH FAILURE RATE"
        print(f"\n--- Health Check (window {window_num}, clips {(window_num-1)*health_interval+1}-"
              f"{window_num*health_interval}) ---")
        print(f"  Success: {success}/{total} ({100*(1-fail_rate):.1f}%)")
        print(f"  Parse failures: {fail}/{total} ({100*fail_rate:.1f}%)")
        print(f"  Empty required fields: {empty_fields} ({100*empty_rate:.1f}%)")
        print(f"  Status: {status}")
        if fail_rate >= 0.05:
            print(f"  ⚠ PAUSING RECOMMENDED: failure rate {100*fail_rate:.1f}% exceeds 5% threshold")
        print()

    # Process clips
    success_count = 0
    fail_count = 0
    agreement_scores = []

    for i, clip in enumerate(remaining):
        clip_id = clip.get("clip_id", "")
        frames_dir = clip.get("frames_dir", "")
        abs_frames_dir = os.path.join(OUTPUT_DIR, frames_dir)

        frame_paths = sorted(glob.glob(os.path.join(abs_frames_dir, "frame_*.jpg")))
        if len(frame_paths) < 2:
            print(f"[{i+1}/{len(remaining)}] {clip_id}: Skipping ({len(frame_paths)} frames)")
            continue

        print(f"[{i+1}/{len(remaining)}] {clip_id} ({len(frame_paths)} frames)...")

        # Build per-clip safety prompt block (if safety-enhanced mode)
        clip_safety_block = None
        if args.safety_enhanced:
            site = clip.get("site", "Unknown")
            work_area = clip.get("work_area", "Unknown")
            camera_id = clip.get("camera_id", "")
            clip_safety_block = build_safety_prompt_block(
                camera_id, site, work_area,
                rules_config=safety_rules_config,
                camera_zones=camera_zones,
            )

        parsed_a, raw_a, success_a = None, "", False
        parsed_b, raw_b, success_b = None, "", False

        # Call VLM A
        if run_a:
            is_local_a = config_a.get("backend") == "local"
            print(f"  VLM A ({config_a['model']})...", end=" ", flush=True)
            if is_local_a:
                parsed_a, raw_a, success_a = call_vlm_local(
                    config_a["model_path"], frame_paths)
            else:
                parsed_a, raw_a, success_a = call_vlm(
                    client_a, config_a["model"], frame_paths,
                    use_openai_params=(config_a.get("base_url") is None),
                    response_format=config_a.get("response_format"),
                    disable_thinking=config_a.get("disable_thinking", False),
                    safety_block=clip_safety_block,
                )
            print("OK" if success_a else "FAIL")

        if run_a and run_b:
            time.sleep(args.delay)

        # Call VLM B
        if run_b:
            is_local_b = config_b.get("backend") == "local"
            print(f"  VLM B ({config_b['model']})...", end=" ", flush=True)
            if is_local_b:
                parsed_b, raw_b, success_b = call_vlm_local(
                    config_b["model_path"], frame_paths)
            else:
                parsed_b, raw_b, success_b = call_vlm(
                    client_b, config_b["model"], frame_paths,
                    use_openai_params=(config_b.get("base_url") is None),
                    response_format=config_b.get("response_format"),
                    disable_thinking=config_b.get("disable_thinking", False),
                    safety_block=clip_safety_block,
                )
            print("OK" if success_b else "FAIL")

        # Normalize annotations
        norm_a = normalize_annotation(parsed_a) if parsed_a else None
        norm_b = normalize_annotation(parsed_b) if parsed_b else None

        # Compute agreement (only when both VLMs ran)
        agreement = {}
        if norm_a and norm_b:
            agreement = compute_clip_agreement(norm_a, norm_b)
            agreement_scores.append(agreement["composite"])
            print(f"  Agreement: composite={agreement['composite']}, "
                  f"action={agreement['a1_action']}, "
                  f"spatial={agreement['a2_spatial']}, "
                  f"ppe={agreement['a3_ppe']}")
        elif args.single_vlm:
            which = "A" if run_a else "B"
            status = success_a if run_a else success_b
            print(f"  Single-VLM mode ({which}): {'OK' if status else 'FAIL'}")
        else:
            print(f"  Agreement: N/A (parse failure)")

        # Save result — single VLM format
        # Use whichever VLM ran (prefer B if both ran)
        if run_b and success_b:
            vlm_model = config_b["model"]
            vlm_parsed = parsed_b
            vlm_norm = norm_b
            vlm_raw = raw_b
            vlm_success = success_b
        elif run_a and success_a:
            vlm_model = config_a["model"]
            vlm_parsed = parsed_a
            vlm_norm = norm_a
            vlm_raw = raw_a
            vlm_success = success_a
        else:
            vlm_model = config_b["model"] if run_b else config_a["model"]
            vlm_parsed = parsed_b if run_b else parsed_a
            vlm_norm = norm_b if run_b else norm_a
            vlm_raw = raw_b if run_b else raw_a
            vlm_success = False

        result = {
            "clip_id": clip_id,
            "timestamp": datetime.now().isoformat(),
            "vlm": {
                "model": vlm_model,
                "success": vlm_success,
                "parsed": vlm_parsed,
                "normalized": vlm_norm,
                "raw_response": vlm_raw[:5000] if isinstance(vlm_raw, str) else "",
            },
        }
        save_result(result, output_path)

        if vlm_success:
            success_count += 1
            health_window_success += 1
            # Check for empty required fields in successful parses
            if vlm_norm:
                persons = vlm_norm.get("persons", [])
                for p in persons:
                    if not p.get("action_code") or not p.get("position"):
                        health_window_empty_fields += 1
                        break
        else:
            fail_count += 1
            health_window_fail += 1

        # Health check every N clips
        clips_processed = success_count + fail_count
        if clips_processed > 0 and clips_processed % health_interval == 0:
            window_num = clips_processed // health_interval
            run_health_check(window_num, health_window_success,
                             health_window_fail, health_window_empty_fields)
            health_window_success = 0
            health_window_fail = 0
            health_window_empty_fields = 0

        # Rate limiting
        time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 60)
    print("PILOT ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Clips processed:  {success_count + fail_count}")
    print(f"Success:          {success_count}")
    print(f"Parse failures:   {fail_count}")

    if agreement_scores:
        import numpy as np
        scores = np.array(agreement_scores)
        print(f"\nAgreement scores (n={len(scores)}):")
        print(f"  Mean:   {scores.mean():.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std:    {scores.std():.4f}")
        print(f"  Min:    {scores.min():.4f}")
        print(f"  Max:    {scores.max():.4f}")

        # Routing distribution at default thresholds
        high = (scores >= 0.80).sum()
        medium = ((scores >= 0.50) & (scores < 0.80)).sum()
        low = (scores < 0.50).sum()
        print(f"\nRouting distribution:")
        print(f"  High confidence (>=0.80):  {high} ({100*high/len(scores):.1f}%)")
        print(f"  Medium (0.50-0.80):        {medium} ({100*medium/len(scores):.1f}%)")
        print(f"  Low (<0.50):               {low} ({100*low/len(scores):.1f}%)")

    out_display = output_path or os.path.join(VLM_RESULTS_DIR, "vlm_annotations.jsonl")
    print(f"\nResults saved to: {out_display}")


if __name__ == "__main__":
    main()
