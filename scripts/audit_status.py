#!/usr/bin/env python3
"""Check audit data readiness for paper Section 4.

Computes:
1. Expert GT (proper-chain vs VLM-sourced)
2. Double-annotated pairs (blind vs anchored)
3. Blind clips for anchoring bias
4. Blind + expert GT overlap
5. Direction analysis pairs (tier_1 vs expert proper-chain)
6. Safety officer reviews
"""
import json, glob, os
from collections import defaultdict

BASE = os.environ.get("STEELBENCH_BASE", "/opt/steelbench")

print("=" * 70)
print("  AUDIT DATA STATUS")
print("=" * 70)

# Load all annotations indexed by role and clip_id
annotations = {}  # {role: {clip_id: annotation_dict}}
for role in ["annotator_1", "annotator_2", "annotator_3", "annotator_4",
             "expert_1", "expert_2", "safety_officer"]:
    annotations[role] = {}
    for f in glob.glob(os.path.join(BASE, "active_batch/annotations/{}/*.json".format(role))):
        try:
            d = json.load(open(f))
            cid = d.get("clip_id", "")
            if cid:
                annotations[role][cid] = d
        except:
            pass

tier1_roles = ["annotator_1", "annotator_2", "annotator_3", "annotator_4"]
expert_roles = ["expert_1", "expert_2"]

# Submitted sets
tier1_submitted = {}  # {role: set of clip_ids}
for a in tier1_roles:
    tier1_submitted[a] = set(cid for cid, d in annotations[a].items() if d.get("status") == "submitted")

all_tier1_clips = set()
for a in tier1_roles:
    all_tier1_clips |= tier1_submitted[a]

# ============================================================
# 1. Expert GT
# ============================================================
print("\n1. EXPERT GT CLIPS")
print("-" * 50)
all_proper = set()
all_vlm_sourced = set()

for e in expert_roles:
    proper = vlm = other = total = 0
    for cid, d in annotations[e].items():
        if d.get("status") != "submitted":
            continue
        total += 1
        st = d.get("edit_tracking", {}).get("source_type", "")
        if st == "tier_1":
            proper += 1
            all_proper.add(cid)
        elif st == "vlm":
            vlm += 1
            all_vlm_sourced.add(cid)
        else:
            other += 1
    print("  {}: {} total (proper-chain={}, VLM-sourced={}, other={})".format(
        e, total, proper, vlm, other))

print("  TOTAL proper-chain: {}".format(len(all_proper)))
print("  TOTAL VLM-sourced: {}".format(len(all_vlm_sourced)))

# ============================================================
# 2. Double-annotated pairs
# ============================================================
print("\n2. DOUBLE-ANNOTATED PAIRS (for IAA)")
print("-" * 50)

# Find clips submitted by 2+ tier_1 annotators
clip_annotators = defaultdict(list)  # clip_id -> [(role, annotation)]
for a in tier1_roles:
    for cid, d in annotations[a].items():
        if d.get("status") == "submitted":
            clip_annotators[cid].append((a, d))

da_clips = {cid: anns for cid, anns in clip_annotators.items() if len(anns) >= 2}

# Classify by calibration condition (check ALL annotators' records)
da_blind = da_anchored = da_contaminated = da_null = 0
for cid, anns in da_clips.items():
    conditions = set(d.get("calibration_condition", "") for _, d in anns)
    if "blind" in conditions:
        da_blind += 1
    elif "anchored" in conditions:
        da_anchored += 1
    elif "contaminated" in conditions:
        da_contaminated += 1
    else:
        da_null += 1

print("  Total DA pairs: {}".format(len(da_clips)))
print("    Blind: {}".format(da_blind))
print("    Anchored: {}".format(da_anchored))
print("    Contaminated (excluded): {}".format(da_contaminated))
print("    No condition: {}".format(da_null))

# ============================================================
# 3. Blind clips
# ============================================================
print("\n3. BLIND CLIPS (for anchoring bias)")
print("-" * 50)
blind_clips = set()
anchored_clips = set()
for a in tier1_roles:
    for cid, d in annotations[a].items():
        if d.get("status") != "submitted":
            continue
        cc = d.get("calibration_condition", "")
        if cc == "blind":
            blind_clips.add(cid)
        elif cc == "anchored":
            anchored_clips.add(cid)

print("  Blind unique clips: {}".format(len(blind_clips)))
print("  Anchored unique clips: {}".format(len(anchored_clips)))
print("  Total calibration clips: {}".format(len(blind_clips | anchored_clips)))

# ============================================================
# 4. Blind clips with expert GT
# ============================================================
print("\n4. BLIND + EXPERT GT")
print("-" * 50)
expert_submitted_clips = set()
for e in expert_roles:
    for cid, d in annotations[e].items():
        if d.get("status") == "submitted":
            expert_submitted_clips.add(cid)

blind_with_expert = blind_clips & expert_submitted_clips
print("  Blind clips reviewed by expert: {}".format(len(blind_with_expert)))
# Also check proper-chain specifically
blind_with_proper = blind_clips & all_proper
print("  Blind clips with proper-chain expert: {}".format(len(blind_with_proper)))

# ============================================================
# 5. Direction analysis (tier_1 annotations for proper-chain clips)
# ============================================================
print("\n5. DIRECTION ANALYSIS")
print("-" * 50)
# Direction analysis = for each proper-chain expert clip, count matching
# tier_1 PERSON-LEVEL pairs (not just clip-level)
direction_clip_pairs = 0
direction_person_pairs = 0

for cid in all_proper:
    expert_ann = None
    for e in expert_roles:
        if cid in annotations[e] and annotations[e][cid].get("status") == "submitted":
            if annotations[e][cid].get("edit_tracking", {}).get("source_type") == "tier_1":
                expert_ann = annotations[e][cid]
                break

    if not expert_ann:
        continue

    # Find matching tier_1 annotation
    for a in tier1_roles:
        if cid in annotations[a] and annotations[a][cid].get("status") == "submitted":
            direction_clip_pairs += 1
            # Count person-level pairs
            t1_persons = annotations[a][cid].get("persons", [])
            exp_persons = expert_ann.get("persons", [])
            if t1_persons and exp_persons:
                direction_person_pairs += min(len(t1_persons), len(exp_persons))
            elif not t1_persons and not exp_persons:
                # Layer 1 — dominant actions comparison
                direction_person_pairs += 1
            break  # Only count first matching tier_1

print("  Clip-level pairs: {}".format(direction_clip_pairs))
print("  Person-level pairs: {}".format(direction_person_pairs))

# ============================================================
# 6. Safety officer
# ============================================================
print("\n6. SAFETY OFFICER")
print("-" * 50)
so_submitted = sum(1 for cid, d in annotations["safety_officer"].items()
                   if d.get("status") == "submitted")
print("  Submitted: {}".format(so_submitted))

# Tier3 queue
try:
    with open(os.path.join(BASE, "annotation_tool/data/assignments/tier3_queue.json")) as f:
        t3 = json.load(f)
    if isinstance(t3, dict):
        t3 = t3.get("queue", [])
    print("  Tier3 queue available: {}".format(len(t3)))
except:
    pass

# ============================================================
# 7. Unique clips summary
# ============================================================
print("\n7. DATASET SIZE")
print("-" * 50)
expert_unique = set()
for e in expert_roles:
    for cid, d in annotations[e].items():
        if d.get("status") == "submitted":
            expert_unique.add(cid)

print("  Tier_1 unique: {}".format(len(all_tier1_clips)))
print("  Expert unique: {}".format(len(expert_unique)))
print("  Eval set (union): {}".format(len(all_tier1_clips | expert_unique)))

# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print("  PAPER READINESS")
print("=" * 70)
checks = [
    ("Expert proper-chain GT", len(all_proper), 100),
    ("Blind DA pairs", da_blind, 50),
    ("Blind unique clips", len(blind_clips), 50),
    ("Blind + expert GT", len(blind_with_expert), 30),
    ("Direction analysis (clips)", direction_clip_pairs, 100),
    ("Direction analysis (persons)", direction_person_pairs, 150),
    ("Safety officer reviews", so_submitted, 50),
]
for name, have, need in checks:
    status = "PASS" if have >= need else "FAIL"
    print("  {:<35} {:>4}/{:<4}  {}".format(name, have, need, status))
