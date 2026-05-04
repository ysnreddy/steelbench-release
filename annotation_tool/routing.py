"""Clip routing based on VLM agreement scores.

Routes clips to high/medium/low confidence verification paths
based on composite agreement thresholds.
"""

# Default thresholds (can be overridden after pilot calibration)
HIGH_AGREEMENT_THRESHOLD = 0.80
LOW_AGREEMENT_THRESHOLD = 0.50


def route_clip(composite_score, high_threshold=None, low_threshold=None):
    """Route a single clip based on its composite agreement score.

    Returns:
        dict with route, human_effort, and description
    """
    high_t = high_threshold or HIGH_AGREEMENT_THRESHOLD
    low_t = low_threshold or LOW_AGREEMENT_THRESHOLD

    if composite_score >= high_t:
        return {
            "route": "high_confidence",
            "human_effort": "sample_verify",
            "description": "Accept VLM consensus, verify via stratified random sample (~8%)",
        }
    elif composite_score >= low_t:
        return {
            "route": "medium_confidence",
            "human_effort": "full_review",
            "description": "100% human review with VLM suggestions pre-filled",
        }
    else:
        return {
            "route": "low_confidence",
            "human_effort": "fresh_annotation",
            "description": "100% fresh human annotation from scratch",
        }


def route_batch(clip_agreements, high_threshold=None, low_threshold=None):
    """Route a batch of clips and return summary statistics.

    Args:
        clip_agreements: list of dicts, each with clip_id and either composite
            score (dual-VLM) or vlm_confidence (single-VLM)

    Returns:
        dict with routed clips and distribution summary
    """
    high_t = high_threshold or HIGH_AGREEMENT_THRESHOLD
    low_t = low_threshold or LOW_AGREEMENT_THRESHOLD

    routed = {"high_confidence": [], "medium_confidence": [], "low_confidence": []}

    for clip in clip_agreements:
        # Single-VLM mode: use vlm_confidence (composite is always 0).
        # Dual-VLM mode: use composite agreement score.
        score = clip.get("composite", 0.0)
        if score == 0.0 and "vlm_confidence" in clip:
            score = clip["vlm_confidence"]
        result = route_clip(score, high_t, low_t)
        clip_entry = {**clip, **result}
        routed[result["route"]].append(clip_entry)

    total = len(clip_agreements)
    summary = {
        "total_clips": total,
        "high_confidence": {
            "count": len(routed["high_confidence"]),
            "percentage": round(100 * len(routed["high_confidence"]) / total, 1) if total else 0,
            "human_clips_needed": max(1, int(len(routed["high_confidence"]) * 0.08)),
        },
        "medium_confidence": {
            "count": len(routed["medium_confidence"]),
            "percentage": round(100 * len(routed["medium_confidence"]) / total, 1) if total else 0,
            "human_clips_needed": len(routed["medium_confidence"]),
        },
        "low_confidence": {
            "count": len(routed["low_confidence"]),
            "percentage": round(100 * len(routed["low_confidence"]) / total, 1) if total else 0,
            "human_clips_needed": len(routed["low_confidence"]),
        },
    }

    summary["total_human_clips"] = (
        summary["high_confidence"]["human_clips_needed"]
        + summary["medium_confidence"]["human_clips_needed"]
        + summary["low_confidence"]["human_clips_needed"]
    )
    summary["human_effort_percentage"] = round(
        100 * summary["total_human_clips"] / total, 1
    ) if total else 0

    return {"routed_clips": routed, "summary": summary}


def select_verification_sample(high_confidence_clips, sample_size=None,
                               stratify_by=None):
    """Select stratified random sample from high-confidence clips for verification.

    Args:
        high_confidence_clips: list of clip dicts in high-confidence path
        sample_size: number of clips to sample (default: ~8% or 350-400)
        stratify_by: list of keys to stratify on (e.g., ["camera_id", "visual_condition_auto"])

    Returns:
        list of selected clip_ids for verification
    """
    import random

    if not high_confidence_clips:
        return []

    if sample_size is None:
        sample_size = max(1, min(400, int(len(high_confidence_clips) * 0.08)))

    if not stratify_by or sample_size >= len(high_confidence_clips):
        return random.sample(
            high_confidence_clips,
            min(sample_size, len(high_confidence_clips))
        )

    # Stratified sampling
    strata = {}
    for clip in high_confidence_clips:
        key = tuple(str(clip.get(k, "unknown")) for k in stratify_by)
        strata.setdefault(key, []).append(clip)

    # Proportional allocation with minimum 1 per stratum
    selected = []
    remaining_budget = sample_size

    for key, clips in strata.items():
        proportion = len(clips) / len(high_confidence_clips)
        n = max(1, round(proportion * sample_size))
        n = min(n, len(clips), remaining_budget)
        selected.extend(random.sample(clips, n))
        remaining_budget -= n
        if remaining_budget <= 0:
            break

    return selected
