#!/usr/bin/env python3
"""Evaluation VLM inference runner for SteelBench.

Runs evaluation models on GT clips and stores predictions for metrics computation.
Reuses the exact same prompt and parsing pipeline as vlm_annotator.py to ensure
identical evaluation conditions.

Usage:
    # Test 10 clips first (validation before batch)
    python eval_inference.py --model gpt4o --test 10
    python eval_inference.py --model qwen36_35b --test 10

    # Full run (real-time with parallel workers)
    python eval_inference.py --model qwen36_35b --workers 4
    python eval_inference.py --model gpt4o --workers 5

    # Bedrock models (real-time, for test or small runs)
    python eval_inference.py --model claude_sonnet --test 10
    python eval_inference.py --model llama4_maverick --workers 4

    # Run all real-time models
    python eval_inference.py --model all --workers 4
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from annotation_tool.vlm_annotator import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    build_messages,
    call_vlm,
    encode_image_base64,
    normalize_annotation,
    parse_json_response,
    validate_and_fix_response,
    validate_response_completeness,
)
from annotation_tool.safety_rules import build_safety_prompt_block

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
EVAL_MODELS = {
    # --- Tier 2: Medium open (DeepInfra API) ---
    "qwen36_35b": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "api_key_env": "DeepInfra_API_KEY_1",
        "disable_thinking": True,
    },
    "gemma4_31b": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "google/gemma-4-31B-it",
        "api_key_env": "DeepInfra_API_KEY_2",
    },
    # --- Tier 3: Large open (DeepInfra API) ---
    "qwen35_122b": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen3.5-122B-A10B",
        "api_key_env": "DeepInfra_API_KEY_3",
        "disable_thinking": True,
    },
    # --- Tier 3: Large open (Bedrock) ---
    "llama4_maverick": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_key_env": "DeepInfra_API_KEY_4",
    },
    "pixtral_large": {
        "provider": "bedrock",
        "model": "us.mistral.pixtral-large-2502-v1:0",
    },
    # --- Tier 4: Frontier ---
    "gpt4o": {
        "provider": "openai_compat",
        "base_url": None,
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "response_format": {"type": "json_object"},
        "use_openai_params": True,
    },
    "gpt54": {
        "provider": "openai_compat",
        "base_url": None,
        "model": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
        "response_format": {"type": "json_object"},
        "use_openai_params": True,
    },
    "claude_opus": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini25_pro": {
        "provider": "google",
        "model": "gemini-2.5-pro",
        "gcp_project": "atomic-amulet-493017-r1",
        "gcp_location": "us-central1",
    },
    "nemotron_12b": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL",
        "api_key_env": "DeepInfra_API_KEY_2",
    },
    "gemini25_flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "gcp_project": "atomic-amulet-493017-r1",
        "gcp_location": "us-central1",
    },
    # --- New models (OpenRouter + DeepInfra) ---
    "nemotron_omni": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
        "api_key_env": "DeepInfra_API_KEY_1",
    },
    "kimi_k26": {
        "provider": "openai_compat",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "moonshotai/Kimi-K2.6",
        "api_key_env": "DeepInfra_API_KEY_1",
    },
    "qwen35_plus": {
        "provider": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3.5-plus-20260420",
        "api_key_env": "OpenRouter_API_KEY",
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent / "eval_data"
FRAMES_DIR = EVAL_DIR / "frames"  # Override with --frames-dir
RESULTS_DIR = EVAL_DIR / "results"
GT_CLIPS_FILE = EVAL_DIR / "gt_clips.json"
CONFIG_DIR = Path(__file__).parent / "eval_data" / "config"

# ---------------------------------------------------------------------------
# Manifest cache (avoid re-reading CSV for every clip)
# ---------------------------------------------------------------------------
_MANIFEST_CACHE = None


def load_manifest():
    """Load and cache the batch manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    _MANIFEST_CACHE = {}
    manifest_path = CONFIG_DIR / "batch_manifest.csv"
    if not manifest_path.exists():
        return _MANIFEST_CACHE

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row.get("clip_id", "")
            if clip_id:
                _MANIFEST_CACHE[clip_id] = row
    return _MANIFEST_CACHE


def load_gt_clip_ids():
    """Load the list of GT clip IDs to evaluate."""
    with open(GT_CLIPS_FILE) as f:
        return json.load(f)


def get_frame_paths(clip_id):
    """Get sorted frame paths for a clip."""
    frame_dir = FRAMES_DIR / clip_id
    if not frame_dir.exists():
        return []
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    return [str(f) for f in frames]


def load_existing_results(output_path):
    """Load successfully-processed clip IDs to enable resume.

    Only skips clips with at least one successful record.
    Failed clips are retried.
    """
    processed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    clip_id = record.get("clip_id", "")
                    if clip_id and record.get("success"):
                        processed.add(clip_id)
                except json.JSONDecodeError:
                    continue
    return processed


# ---------------------------------------------------------------------------
# Bedrock Converse API caller
# ---------------------------------------------------------------------------
def call_bedrock(bedrock_client, model_id, frame_paths, safety_block=None):
    """Call a Bedrock model via the Converse API with images.

    Returns (parsed_json, raw_text, success_bool)
    """
    prompt_text = USER_PROMPT
    if safety_block:
        prompt_text = USER_PROMPT + "\n\n" + safety_block

    # Build content blocks: text + images
    content = [{"text": prompt_text}]
    for path in frame_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                img_bytes = f.read()
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": img_bytes},
                }
            })

    messages = [{"role": "user", "content": content}]

    for attempt in range(4):
        try:
            response = bedrock_client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": 6000,
                    "temperature": 0.1,
                },
            )

            raw_text = ""
            for block in response.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    raw_text += block["text"]

            parsed = parse_json_response(raw_text)
            if parsed:
                parsed = validate_and_fix_response(parsed) or parsed
                parsed, _ = validate_response_completeness(parsed)
                return parsed, raw_text, True
            elif attempt < 3:
                time.sleep(2)
                continue
            else:
                return None, raw_text, False

        except Exception as e:
            err_str = str(e)
            is_throttle = "throttl" in err_str.lower() or "rate" in err_str.lower()
            if attempt < 3:
                wait = (attempt + 1) * 15 if is_throttle else (attempt + 1) * 5
                print(f"    Bedrock error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None, err_str, False


# ---------------------------------------------------------------------------
# Anthropic Messages API caller
# ---------------------------------------------------------------------------
def call_anthropic(anthropic_client, model_id, frame_paths, safety_block=None):
    """Call Anthropic Claude via the Messages API with images.

    Returns (parsed_json, raw_text, success_bool)
    """
    prompt_text = USER_PROMPT
    if safety_block:
        prompt_text = USER_PROMPT + "\n\n" + safety_block

    # Build content blocks: images first, then text
    content = []
    for path in frame_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
    content.append({"type": "text", "text": prompt_text})

    for attempt in range(4):
        try:
            response = anthropic_client.messages.create(
                model=model_id,
                max_tokens=6000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )

            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text += block.text

            parsed = parse_json_response(raw_text)
            if parsed:
                parsed = validate_and_fix_response(parsed) or parsed
                parsed, _ = validate_response_completeness(parsed)
                return parsed, raw_text, True
            elif attempt < 3:
                time.sleep(2)
                continue
            else:
                return None, raw_text, False

        except Exception as e:
            err_str = str(e)
            is_rate = "rate" in err_str.lower() or "429" in err_str
            if attempt < 3:
                wait = (attempt + 1) * 15 if is_rate else (attempt + 1) * 5
                print(f"    Anthropic error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None, err_str, False


# ---------------------------------------------------------------------------
# Google Gemini API caller
# ---------------------------------------------------------------------------
def call_gemini(gemini_client, model_id, frame_paths, safety_block=None, compact=False):
    """Call Google Gemini via Vertex AI with images.

    Args:
        compact: If True, add instruction to keep per-worker output concise.
                 Used for Flash models that truncate on large scenes.

    Returns (parsed_json, raw_text, success_bool)
    """
    from google.genai import types

    prompt_text = USER_PROMPT
    if safety_block:
        prompt_text = USER_PROMPT + "\n\n" + safety_block
    if compact:
        prompt_text += ("\n\nIMPORTANT: Keep your response concise. For each worker, "
                        "provide ONLY the required fields (id, position, action_class, "
                        "confidence, ppe, spatial_context, unsafe_act, group_flag, "
                        "occlusion_level). Omit free_text_description, physical_description, "
                        "and tool_or_equipment to stay within output limits.")

    # Build parts: system + text + images
    parts = [types.Part(text=prompt_text)]
    for path in frame_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                img_bytes = f.read()
            parts.append(types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)
            ))

    content = types.Content(parts=parts, role="user")

    for attempt in range(4):
        try:
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=16000,
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text
            parsed = parse_json_response(raw_text)
            if parsed:
                parsed = validate_and_fix_response(parsed) or parsed
                parsed, _ = validate_response_completeness(parsed)
                return parsed, raw_text, True
            elif attempt < 3:
                time.sleep(2)
                continue
            else:
                return None, raw_text, False

        except Exception as e:
            err_str = str(e)
            if attempt < 3:
                wait = (attempt + 1) * 10
                print(f"    Gemini error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None, err_str, False


# ---------------------------------------------------------------------------
# Unified clip processor
# ---------------------------------------------------------------------------
def process_single_clip(clip_id, model_config, client, rules_config, camera_zones):
    """Process a single clip with any provider.

    Returns a result dict.
    """
    frame_paths = get_frame_paths(clip_id)
    if not frame_paths:
        return {"clip_id": clip_id, "error": "no_frames", "success": False}

    # Build safety prompt block for this clip
    manifest = load_manifest()
    row = manifest.get(clip_id, {})
    camera_id = row.get("camera_id", "")
    site = row.get("site", "")
    work_area = row.get("work_area", "")

    safety_block = None
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

    provider = model_config.get("provider", "openai_compat")

    # Call appropriate provider
    if provider == "bedrock":
        parsed, raw_text, success = call_bedrock(
            client, model_config["model"], frame_paths, safety_block
        )
    elif provider == "anthropic":
        parsed, raw_text, success = call_anthropic(
            client, model_config["model"], frame_paths, safety_block
        )
    elif provider == "google":
        compact = model_config.get("compact_output", False)
        parsed, raw_text, success = call_gemini(
            client, model_config["model"], frame_paths, safety_block,
            compact=compact,
        )
    else:
        # OpenAI-compatible (OpenAI, DeepInfra, OpenRouter)
        from openai import OpenAI as _OpenAI
        parsed, raw_text, success = call_vlm(
            client=client,
            model=model_config["model"],
            frame_paths=frame_paths,
            max_retries=3,
            use_openai_params=model_config.get("use_openai_params", False),
            response_format=model_config.get("response_format"),
            disable_thinking=model_config.get("disable_thinking", False),
            safety_block=safety_block,
        )

    # Normalize
    normalized = None
    if success and parsed:
        try:
            normalized = normalize_annotation(parsed)
        except Exception:
            normalized = None

    return {
        "clip_id": clip_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_config["model"],
        "success": success,
        "parsed": parsed,
        "normalized": normalized,
        "raw_response": (raw_text or "")[:5000],
    }


# ---------------------------------------------------------------------------
# Validation report for test mode
# ---------------------------------------------------------------------------
def print_validation_report(results, model_name):
    """Print detailed validation report for test clips."""
    print(f"\n{'='*70}")
    print(f"  VALIDATION REPORT: {model_name}")
    print(f"{'='*70}")

    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    failed = total - success

    print(f"\n  Success: {success}/{total}")
    if failed:
        print(f"  FAILED:  {failed}/{total}")
        for r in results:
            if not r.get("success"):
                print(f"    {r['clip_id']}: {r.get('error', 'parse_fail')}")

    # Check parsing quality
    print(f"\n  --- Parsing Quality ---")
    for r in results:
        if not r.get("success"):
            continue
        norm = r.get("normalized", {})
        clip = r["clip_id"][:50]

        issues = []
        # Check required fields
        if not norm.get("scene_type"):
            issues.append("missing scene_type")
        if not norm.get("persons") and norm.get("annotation_layer") == 2:
            issues.append("Layer 2 but no persons")
        if norm.get("num_workers", 0) == 0:
            issues.append("num_workers=0")

        # Check persons
        for i, p in enumerate(norm.get("persons", [])):
            ac = p.get("action_code", "")
            if not ac:
                issues.append(f"person[{i}] missing action_code")
            elif ac not in ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",
                           "B6","B7","B8","B9","C1","C2","C3","C4","D1","D2",
                           "D3","E1","E2","F1","F2","X1"]:
                issues.append(f"person[{i}] invalid action_code: {ac}")

            conf = p.get("confidence", -1)
            if not (0 <= conf <= 1):
                issues.append(f"person[{i}] confidence out of range: {conf}")

            ppe = p.get("ppe", {})
            for item in ["helmet", "high_vis_vest", "safety_shoes"]:
                val = ppe.get(item, "")
                if val not in ["worn", "not_worn", "cannot_determine", "not_applicable"]:
                    issues.append(f"person[{i}] bad PPE {item}={val}")

        status = "OK" if not issues else f"ISSUES: {'; '.join(issues)}"
        workers = norm.get("num_workers", "?")
        scene = norm.get("scene_type", "?")
        actions = [p.get("action_code", "?") for p in norm.get("persons", [])][:3]
        print(f"    {clip:<50} w={workers} {scene} {actions} {status}")

    # Summary stats
    successful = [r for r in results if r.get("success")]
    if successful:
        confs = []
        for r in successful:
            for p in r.get("normalized", {}).get("persons", []):
                c = p.get("confidence", 0)
                if isinstance(c, (int, float)):
                    confs.append(c)

        if confs:
            avg_conf = sum(confs) / len(confs)
            print(f"\n  Avg confidence: {avg_conf:.3f}")
            print(f"  Confidence range: [{min(confs):.3f}, {max(confs):.3f}]")

        # Raw response length
        raw_lens = [len(r.get("raw_response", "")) for r in successful]
        print(f"  Avg response length: {sum(raw_lens)/len(raw_lens):.0f} chars")

    print(f"\n  Verdict: {'PASS - ready for full run' if failed == 0 else 'NEEDS FIXING'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def create_client(config):
    """Create the appropriate API client for a model config."""
    provider = config.get("provider", "openai_compat")

    if provider == "bedrock":
        import boto3
        from botocore.config import Config as BotoConfig
        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        return session.client(
            "bedrock-runtime",
            config=BotoConfig(
                read_timeout=300,      # 5 min read timeout for large vision requests
                connect_timeout=30,
                retries={"max_attempts": 2},
            ),
        )

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            print(f"ERROR: {config['api_key_env']} not found in .env")
            return None
        return anthropic.Anthropic(api_key=api_key)

    elif provider == "google":
        from google import genai
        project = config.get("gcp_project", "atomic-amulet-493017-r1")
        location = config.get("gcp_location", "us-central1")
        return genai.Client(vertexai=True, project=project, location=location)

    else:
        # OpenAI-compatible
        from openai import OpenAI
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            print(f"ERROR: {config['api_key_env']} not found in .env")
            return None
        client_kwargs = {"api_key": api_key}
        if config.get("base_url"):
            client_kwargs["base_url"] = config["base_url"]
        return OpenAI(**client_kwargs)


def run_inference(model_name, workers=4, test_n=0, rpm_limit=0):
    """Run inference for a single model."""
    load_dotenv()

    if model_name not in EVAL_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(EVAL_MODELS.keys())}")
        return

    config = EVAL_MODELS[model_name]
    client = create_client(config)
    if client is None:
        return

    # Load clips
    clip_ids = load_gt_clip_ids()
    print(f"Loaded {len(clip_ids)} eval clips")

    # Test mode: use pre-selected diverse test clips
    if test_n > 0:
        test_file = EVAL_DIR / "test_clips.json"
        if test_file.exists():
            clip_ids = json.load(open(test_file))[:test_n]
            print(f"TEST MODE: using {len(clip_ids)} pre-selected diverse clips")
        else:
            test_clips = []
            for cid in clip_ids:
                if get_frame_paths(cid):
                    test_clips.append(cid)
                if len(test_clips) >= test_n:
                    break
            clip_ids = test_clips
            print(f"TEST MODE: running on {len(clip_ids)} clips")
        workers = min(workers, 2)  # Slow down for test

    # Load safety rules
    rules_config = None
    camera_zones = None
    try:
        from annotation_tool.safety_rules import load_safety_rules, load_camera_zones
        rules_config = load_safety_rules(str(CONFIG_DIR))
        camera_zones = load_camera_zones(str(CONFIG_DIR))
        print(f"Loaded safety rules from {CONFIG_DIR}")
    except Exception as e:
        print(f"Warning: Could not load safety rules: {e}")

    # Setup output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_test" if test_n > 0 else ""
    output_path = RESULTS_DIR / f"{model_name}{suffix}.jsonl"

    # Resume support (not for test mode)
    if test_n > 0:
        # Clear test file
        if output_path.exists():
            output_path.unlink()
        remaining = clip_ids
    else:
        already_done = load_existing_results(output_path)
        remaining = [c for c in clip_ids if c not in already_done]
        if already_done:
            print(f"Resuming: {len(already_done)} done, {len(remaining)} remaining")

    if not remaining:
        print("All clips already processed!")
        return

    print(f"\nRunning {config['model']} on {len(remaining)} clips with {workers} workers...")
    print(f"Output: {output_path}\n")

    success_count = 0
    fail_count = 0
    all_results = []
    start_time = time.time()
    min_interval = 60.0 / rpm_limit if rpm_limit > 0 else 0
    last_request_time = 0

    if min_interval > 0:
        print(f"Rate limit: {rpm_limit} RPM ({min_interval:.1f}s between requests)")

    with open(output_path, "a") as out_f:
        if workers == 1 or min_interval > 0:
            # Serial processing — simpler, avoids ThreadPoolExecutor issues
            # Required for rate-limited Bedrock calls
            with tqdm(total=len(remaining), desc=model_name) as pbar:
                for clip_id in remaining:
                    # Rate limiting
                    if min_interval > 0:
                        elapsed_since_last = time.time() - last_request_time
                        if elapsed_since_last < min_interval:
                            time.sleep(min_interval - elapsed_since_last)
                        last_request_time = time.time()

                    try:
                        result = process_single_clip(
                            clip_id, config, client, rules_config, camera_zones
                        )
                        out_f.write(json.dumps(result) + "\n")
                        out_f.flush()
                        all_results.append(result)

                        if result.get("success"):
                            success_count += 1
                        else:
                            fail_count += 1
                            err = result.get("error", "parse_fail")
                            tqdm.write(f"  FAIL {clip_id}: {err}")
                    except Exception as e:
                        fail_count += 1
                        tqdm.write(f"  ERROR {clip_id}: {e}")
                        error_record = {
                            "clip_id": clip_id,
                            "timestamp": datetime.now().isoformat(),
                            "model": config["model"],
                            "success": False,
                            "error": str(e),
                        }
                        out_f.write(json.dumps(error_record) + "\n")
                        out_f.flush()
                        all_results.append(error_record)

                    pbar.update(1)
                    pbar.set_postfix(ok=success_count, fail=fail_count)
        else:
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_single_clip, clip_id, config, client,
                        rules_config, camera_zones
                    ): clip_id
                    for clip_id in remaining
                }

                with tqdm(total=len(remaining), desc=model_name) as pbar:
                    for future in as_completed(futures):
                        clip_id = futures[future]
                        try:
                            result = future.result()
                            out_f.write(json.dumps(result) + "\n")
                            out_f.flush()
                            all_results.append(result)

                            if result.get("success"):
                                success_count += 1
                            else:
                                fail_count += 1
                                err = result.get("error", "parse_fail")
                                tqdm.write(f"  FAIL {clip_id}: {err}")
                        except Exception as e:
                            fail_count += 1
                            tqdm.write(f"  ERROR {clip_id}: {e}")
                            error_record = {
                                "clip_id": clip_id,
                                "timestamp": datetime.now().isoformat(),
                                "model": config["model"],
                                "success": False,
                                "error": str(e),
                            }
                            out_f.write(json.dumps(error_record) + "\n")
                            out_f.flush()
                            all_results.append(error_record)

                        pbar.update(1)
                        pbar.set_postfix(ok=success_count, fail=fail_count)

    elapsed = time.time() - start_time
    total = success_count + fail_count
    print(f"\n{'='*60}")
    print(f"Model: {config['model']}")
    print(f"Total: {total} clips in {elapsed:.1f}s ({elapsed/max(total,1):.1f}s/clip)")
    print(f"Success: {success_count}/{total} ({100*success_count/max(total,1):.1f}%)")
    print(f"Failed: {fail_count}/{total}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    # Print validation report in test mode
    if test_n > 0:
        print_validation_report(all_results, model_name)


def main():
    parser = argparse.ArgumentParser(description="SteelBench evaluation inference")
    parser.add_argument("--model", required=True,
                        choices=list(EVAL_MODELS.keys()) + ["all"],
                        help="Model to run (or 'all')")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: run on N clips with validation report")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Override frames directory (e.g., /opt/steelbench/active_batch/frames)")
    parser.add_argument("--rpm-limit", type=int, default=0,
                        help="Rate limit: max requests per minute (0=no limit)")
    args = parser.parse_args()

    # Override frames directory if specified
    global FRAMES_DIR
    if args.frames_dir:
        FRAMES_DIR = Path(args.frames_dir)
        print(f"Using frames directory: {FRAMES_DIR}")

    if args.model == "all":
        for model_name in EVAL_MODELS:
            config = EVAL_MODELS[model_name]
            # Skip models that need special setup
            if config.get("provider") == "google":
                api_key = os.getenv(config.get("api_key_env", ""))
                if not api_key or api_key == "your_gemini_api_key_here":
                    print(f"Skipping {model_name} (Gemini API key not set)")
                    continue
            print(f"\n{'#'*60}")
            print(f"# Running {model_name}")
            print(f"{'#'*60}\n")
            run_inference(model_name, args.workers, args.test, args.rpm_limit)
    else:
        run_inference(args.model, args.workers, args.test, args.rpm_limit)


if __name__ == "__main__":
    main()
