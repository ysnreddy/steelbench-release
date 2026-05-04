#!/usr/bin/env python3
"""A/B test: does the hardcoded '8 frames' string in USER_PROMPT bias inference
when we actually send fewer (or more) frames?

Runs Gemma 4-31B on 30 clips at 4 frames each:
  A) with ORIGINAL prompt ("These are 8 frames...")
  B) with CORRECTED prompt ("These are 4 frames...")

Computes paired differences on key normalized fields. One-off; no permanent
changes to USER_PROMPT.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from annotation_tool import vlm_annotator as VA
from annotation_tool.vlm_annotator import (
    SYSTEM_PROMPT,
    USER_PROMPT as ORIGINAL_USER_PROMPT,
    parse_json_response,
    validate_and_fix_response,
    validate_response_completeness,
    normalize_annotation,
)
from scripts.frame_density_run import (
    get_base64_images,
    load_manifest,
    get_safety_block,
)


CORRECTED_USER_PROMPT = ORIGINAL_USER_PROMPT.replace("8 frames", "4 frames")
N_CLIPS = 30
NUM_FRAMES = 4
MODEL = "google/gemma-4-31B-it"


def call_gemma(client, system_prompt, user_text, b64_images):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in b64_images]
    content.append({"type": "text", "text": user_text})
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=8000,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_json_response(raw)
    if parsed:
        parsed = validate_and_fix_response(parsed) or parsed
        parsed, _ = validate_response_completeness(parsed)
        try:
            normalized = normalize_annotation(parsed)
        except Exception:
            normalized = None
        return {"success": True, "parsed": parsed, "normalized": normalized}
    return {"success": False, "raw": raw[:500]}


def run_one(clip_id, manifest, client):
    safety = get_safety_block(clip_id, manifest)
    b64 = get_base64_images(clip_id, NUM_FRAMES,
                            frames_dir=Path("active_batch/frames"))
    if not b64 or len(b64) < NUM_FRAMES:
        return clip_id, None, None, "missing frames"

    user_a = ORIGINAL_USER_PROMPT + ("\n\n" + safety if safety else "")
    user_b = CORRECTED_USER_PROMPT + ("\n\n" + safety if safety else "")

    try:
        out_a = call_gemma(client, SYSTEM_PROMPT, user_a, b64)
        out_b = call_gemma(client, SYSTEM_PROMPT, user_b, b64)
        return clip_id, out_a, out_b, None
    except Exception as e:
        return clip_id, None, None, str(e)[:200]


def extract_fields(rec):
    if not rec or not rec.get("normalized"):
        return None
    n = rec["normalized"]
    persons = n.get("persons") or []
    return {
        "scene_type": n.get("scene_type"),
        "num_workers": n.get("num_workers"),
        "annotation_layer": n.get("annotation_layer"),
        "dominant_action_0": (n.get("dominant_actions") or [None])[0],
        "overall_ppe_compliance": n.get("overall_ppe_compliance"),
        "action_transition_detected": (n.get("action_transition") or {}).get("detected"),
        "annotator_confidence": n.get("annotator_confidence"),
        "n_persons": len(persons),
        "primary_person_action": (persons[0].get("action_code") if persons else None),
        "primary_person_unsafe": (persons[0].get("unsafe_act") if persons else None),
    }


def main():
    load_dotenv()
    api_key = os.getenv("DeepInfra_API_KEY_2")
    from openai import OpenAI

    clips = json.load(open(ROOT / "eval_data/prompt_sensitivity_clips.json"))[:N_CLIPS]
    manifest = load_manifest()
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    out_dir = ROOT / "eval_data/results/ab_prompt_frame_count"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"A/B test: {N_CLIPS} clips at {NUM_FRAMES} frames")
    print(f"  A = original prompt ('These are 8 frames...')")
    print(f"  B = corrected prompt ('These are 4 frames...')")
    print()
    print(f"  prompt diffs:")
    print(f"    original 'These are 8 frames' occurrences: {ORIGINAL_USER_PROMPT.count('8 frames')}")
    print(f"    corrected 'These are 4 frames' occurrences: {CORRECTED_USER_PROMPT.count('4 frames')}")
    print()

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_one, c, manifest, client): c for c in clips}
        for k, fut in enumerate(as_completed(futures), 1):
            cid, a, b, err = fut.result()
            results.append({"clip_id": cid, "a": a, "b": b, "error": err})
            if err:
                print(f"  {k}/{len(clips)} ERR {cid}: {err}")
            else:
                print(f"  {k}/{len(clips)} ok  {cid}")

    print(f"\n  total time: {time.time()-t0:.0f} s")

    # Save raw results
    with open(out_dir / "ab_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Pair-wise comparison
    print("\n=== Pairwise field-level comparison ===")
    fields = [
        "scene_type", "num_workers", "annotation_layer",
        "dominant_action_0", "overall_ppe_compliance",
        "action_transition_detected", "annotator_confidence",
        "n_persons", "primary_person_action", "primary_person_unsafe",
    ]
    n_paired = 0
    counts = {f: {"agree": 0, "disagree": 0, "missing": 0, "deltas": []} for f in fields}
    for r in results:
        a, b = extract_fields(r["a"]), extract_fields(r["b"])
        if a is None or b is None:
            for f in fields: counts[f]["missing"] += 1
            continue
        n_paired += 1
        for f in fields:
            va, vb = a.get(f), b.get(f)
            if va == vb:
                counts[f]["agree"] += 1
            else:
                counts[f]["disagree"] += 1
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    counts[f]["deltas"].append((va, vb, vb - va))

    print(f"  paired clips: {n_paired}/{len(results)}\n")
    print(f"  {'field':30s}  agree  disagree  agree%   notable_deltas")
    for f in fields:
        c = counts[f]
        tot = c["agree"] + c["disagree"]
        pct = 100.0 * c["agree"] / tot if tot > 0 else 0.0
        deltas_str = ""
        if c["deltas"]:
            mean_delta = sum(d[2] for d in c["deltas"]) / len(c["deltas"])
            max_abs = max(abs(d[2]) for d in c["deltas"])
            deltas_str = f"mean(B-A)={mean_delta:+.3f}  max|d|={max_abs:.2f}"
        print(f"  {f:30s}  {c['agree']:5d}  {c['disagree']:8d}  {pct:5.1f}%   {deltas_str}")

    print()
    print("Interpretation:")
    print("  - 'agree%' = fraction of paired clips where A and B produced identical normalized field.")
    print("  - high agree% (≥95%) on perception fields → prompt mismatch is benign.")
    print("  - significant disagreement on dominant_action_0 / scene_type / primary_person_action")
    print("    → prompt influenced perception; full re-run with density-aware prompt is justified.")


if __name__ == "__main__":
    main()
