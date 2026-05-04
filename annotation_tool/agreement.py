"""Inter-model agreement computation for dual-VLM annotation pipeline.

Aligned with the SteelBench Annotation Policy document.
Computes per-axis agreement scores and weighted composite for routing.
"""

# ---------- AXIS 1: ACTION TAXONOMY (27 classes) ----------

ACTION_TAXONOMY = {
    "A1": "locomotion.walking",
    "A2": "locomotion.climbing_up",
    "A3": "locomotion.climbing_down",
    "A4": "locomotion.carrying_walking",
    "A5": "locomotion.pushing_pulling",
    "B1": "work.standing_operation_tool_use",
    "B2": "work.standing_panel_operation",
    "B3": "work.hot_work_welding",
    "B4": "work.hot_work_non_welding",
    "B5": "work.crouching_floor_work",
    "B6": "work.overhead_reaching",
    "B7": "work.inspection_stationary",
    "B8": "work.lifting_placing",
    "B9": "work.sitting_operation",
    "C1": "crane.signalling",
    "C2": "crane.hook_interaction",
    "C3": "crane.guiding_load",
    "C4": "vehicle.operating",
    "D1": "material.manual_team_carry",
    "D2": "material.loading_unloading",
    "D3": "material.cylinder_handling",
    "E1": "social.communicating",
    "E2": "social.supervising",
    "F1": "idle.standing",
    "F2": "idle.sitting",
    "X1": "other.unlisted_action",
}

# Group mapping
ACTION_GROUPS = {
    "A": ["A1", "A2", "A3", "A4", "A5"],
    "B": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
    "C": ["C1", "C2", "C3", "C4"],
    "D": ["D1", "D2", "D3"],
    "E": ["E1", "E2"],
    "F": ["F1", "F2"],
    "X": ["X1"],
}

# Reverse lookups
CODE_TO_GROUP = {}
NAME_TO_CODE = {}
for code, name in ACTION_TAXONOMY.items():
    group = code[0]
    CODE_TO_GROUP[code] = group
    NAME_TO_CODE[name] = code

# All action codes for dropdowns
ALL_ACTION_CODES = list(ACTION_TAXONOMY.keys())
ALL_ACTION_LABELS = {code: f"[{code}] {name}" for code, name in ACTION_TAXONOMY.items()}

# ---------- AXIS 2: SPATIAL CONTEXT (multi-label) ----------

SPATIAL_TAGS = [
    "ground_level", "elevated", "vehicle_area", "crane_zone",
    "furnace_zone", "confined_space", "restricted", "near_machinery",
    "walkway", "storage_area", "locomotive_area", "hot_area",
]

# ---------- AXIS 3: PPE (5 items) ----------

PPE_ITEMS = ["helmet", "high_vis_vest", "welding_protection", "harness", "safety_shoes"]
PPE_VALUES = ["worn", "not_worn", "cannot_determine"]
PPE_VALUES_EXTENDED = ["worn", "not_worn", "not_applicable", "cannot_determine"]
# welding_protection and harness use PPE_VALUES_EXTENDED (have not_applicable)
PPE_ITEMS_WITH_NA = ["welding_protection", "harness"]

# ---------- AXIS 4: COMPOSITE SAFETY REASONING + INTERACTION ----------

INTERACTION_TYPES = [
    "fixed_equipment",
    "portable_object",
    "crane_system",
    "tool",
    "vehicle",
    "another_person",
    "ground_surface",
    "none_visible",
]

# ---------- CLIP-LEVEL ----------

VISIBILITY_LEVELS = ["clear", "moderate", "harsh"]
VISIBILITY_CONDITIONS = [
    "clear", "steam", "dust", "smoke", "rain", "glare", "low_light",
]
OCCLUSION_LEVELS = [
    "fully_visible", "partially_occluded", "heavily_occluded",
]
OCCLUSION_SOURCES = [
    "equipment", "structure", "other_person", "atmospheric", "frame_edge",
]
GROUP_FLAGS = ["solo", "coordinated", "concurrent"]

# ---------- SCENE TYPE CLASSIFICATION (addendum v2) ----------

SCENE_TYPES = ["SA", "MAI", "MAC", "SEQ", "SA-X", "MAI-X", "MAC-X"]
SCENE_TYPE_LABELS = {
    "SA": "Single-Actor",
    "MAI": "Multi-Actor Independent",
    "MAC": "Multi-Actor Coordinated",
    "SEQ": "Sequential",
    "SA-X": "Single-Actor Unlisted",
    "MAI-X": "Multi-Actor Independent (Unlisted)",
    "MAC-X": "Multi-Actor Coordinated (Unlisted)",
}

# ---------- DUAL-LAYER TAXONOMY (addendum v2) ----------

UNLISTED_ACTION_FIELDS = {
    "closest_existing_class": {"type": "str", "required": True},
    "why_not_existing": {"type": "str", "required": True},
    "free_text_description": {"type": "str", "required": True},
    "tool_or_equipment": {"type": "str", "required": False},
    "suggested_new_class": {"type": "str", "required": False},
    "industry_specific": {"type": "bool", "required": True},
}

# ---------- ANNOTATOR ROLES ----------

ANNOTATOR_ROLES = {
    "tier_1": "General annotation (action, spatial, PPE, interaction)",
    "tier_2": "Senior adjudication (flagged clips, disagreements, QC audit)",
    "tier_3": "Safety officer (safety-critical clips, rule compliance verification)",
}

# ---------- COMPOSITE AGREEMENT WEIGHTS ----------

# Spatial excluded from composite: VLMs produce inconsistent spatial tags
# (GPT-4o returns empty, Qwen returns free-text instead of taxonomy tags).
# Spatial context is evaluated through human annotation instead.
WEIGHTS = {"action": 0.67, "spatial": 0.0, "ppe": 0.33}


# ---------- AGREEMENT FUNCTIONS ----------

def normalize_action_code(action_str):
    """Normalize action input to a code (e.g., 'A1').

    Accepts: 'A1', 'locomotion.walking', 'A1 locomotion.walking', etc.
    """
    if not action_str:
        return ""
    s = action_str.strip()

    # Handle X1 / unlisted variants
    if s.upper() == "X1" or "unlisted" in s.lower():
        return "X1"

    # Extract code from parentheses: 'walking (A1)' -> 'A1'
    import re as _re
    paren_match = _re.search(r'\(([A-FX]\d)\)', s.upper())
    if paren_match and paren_match.group(1) in ACTION_TAXONOMY:
        return paren_match.group(1)

    # Already a code like 'A1'
    if s.upper() in ACTION_TAXONOMY:
        return s.upper()

    # Full name like 'locomotion.walking'
    if s.lower() in NAME_TO_CODE:
        return NAME_TO_CODE[s.lower()]

    # VLM often returns underscore variants like 'idle_standing' instead of 'idle.standing'
    s_dotted = s.lower().replace("_", ".")
    if s_dotted in NAME_TO_CODE:
        return NAME_TO_CODE[s_dotted]
    s_underscored = s.lower().replace(".", "_")
    for name, code in NAME_TO_CODE.items():
        name_underscored = name.replace(".", "_")
        suffix = name.split(".", 1)[-1] if "." in name else name
        suffix_underscored = suffix.replace(".", "_")
        if s_underscored == name_underscored or s.lower() == suffix_underscored:
            return code

    # Label like '[A1] locomotion.walking' or 'A1 locomotion.walking'
    for code in ACTION_TAXONOMY:
        if s.upper().startswith(code):
            return code

    # Try partial match on name
    for name, code in NAME_TO_CODE.items():
        if s.lower() in name or name in s.lower():
            return code

    return s


def action_agreement(vlm_a_action, vlm_b_action, hierarchical=True):
    """Compute agreement for Axis 1: Action Class.

    Returns: 1.0 (exact match), 0.5 (same group), or 0.0
    """
    code_a = normalize_action_code(vlm_a_action)
    code_b = normalize_action_code(vlm_b_action)

    if not code_a or not code_b:
        return 0.0

    if code_a == code_b:
        return 1.0

    if hierarchical:
        group_a = CODE_TO_GROUP.get(code_a, "")
        group_b = CODE_TO_GROUP.get(code_b, "")
        if group_a and group_b and group_a == group_b:
            return 0.5

    return 0.0


def spatial_agreement(vlm_a_tags, vlm_b_tags):
    """Compute agreement for Axis 2: Spatial Context using Jaccard similarity."""
    set_a = set(vlm_a_tags) if vlm_a_tags else set()
    set_b = set(vlm_b_tags) if vlm_b_tags else set()

    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b
    if not union:
        return 1.0

    intersection = set_a & set_b
    return len(intersection) / len(union)


def ppe_agreement(vlm_a_ppe, vlm_b_ppe):
    """Compute agreement for Axis 3: PPE.

    PPE values: worn / not_worn / cannot_determine / not_applicable
    Agreement = proportion of matching items across 5 PPE fields.
    """
    if not vlm_a_ppe or not vlm_b_ppe:
        return 0.0

    matches = 0
    total = 0

    for item in PPE_ITEMS:
        a_val = vlm_a_ppe.get(item, "").lower().strip()
        b_val = vlm_b_ppe.get(item, "").lower().strip()
        if a_val and b_val:
            total += 1
            if a_val == b_val:
                matches += 1

    return matches / total if total > 0 else 0.0


def interaction_agreement(vlm_a_interaction, vlm_b_interaction):
    """Compute agreement for interaction type within Axis 4."""
    if not vlm_a_interaction or not vlm_b_interaction:
        return 0.0

    a = vlm_a_interaction.lower().strip() if isinstance(vlm_a_interaction, str) else ""
    b = vlm_b_interaction.lower().strip() if isinstance(vlm_b_interaction, str) else ""

    if a == b:
        return 1.0
    return 0.0


def safety_reasoning_agreement(vlm_a_unsafe, vlm_b_unsafe):
    """Compute agreement on unsafe act identification within Axis 4.

    Both VLMs either identify an unsafe act or both say safe.
    """
    a_has_unsafe = bool(vlm_a_unsafe and vlm_a_unsafe.strip())
    b_has_unsafe = bool(vlm_b_unsafe and vlm_b_unsafe.strip())

    if a_has_unsafe == b_has_unsafe:
        return 1.0
    return 0.0


def scene_type_agreement(vlm_a_scene_type, vlm_b_scene_type):
    """Compute agreement on scene type classification.

    Returns: 1.0 (exact match), 0.5 (base type matches, e.g. SA vs SA-X), 0.0
    """
    a = (vlm_a_scene_type or "").strip().upper()
    b = (vlm_b_scene_type or "").strip().upper()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a.split("-")[0] == b.split("-")[0]:
        return 0.5
    return 0.0


def coordination_agreement(vlm_a_coord, vlm_b_coord):
    """Compute agreement on coordinated_with worker ID lists (Jaccard)."""
    set_a = set(vlm_a_coord) if vlm_a_coord else set()
    set_b = set(vlm_b_coord) if vlm_b_coord else set()
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def composite_agreement(a1, a2, a3, weights=None):
    """Compute weighted composite: A_composite = w1*A1 + w2*A2 + w3*A3"""
    w = weights or WEIGHTS
    return w["action"] * a1 + w["spatial"] * a2 + w["ppe"] * a3


def compute_person_agreement(person_a, person_b):
    """Compute agreement for a single matched person (P1 vs P1, etc.)."""
    a1 = action_agreement(
        person_a.get("action_code", person_a.get("action_class", "")),
        person_b.get("action_code", person_b.get("action_class", "")),
    )
    a2 = spatial_agreement(
        person_a.get("spatial_context", []),
        person_b.get("spatial_context", []),
    )
    a3 = ppe_agreement(
        person_a.get("ppe", {}),
        person_b.get("ppe", {}),
    )
    a4_interaction = interaction_agreement(
        person_a.get("interaction", ""),
        person_b.get("interaction", ""),
    )
    a4_safety = safety_reasoning_agreement(
        person_a.get("unsafe_act", ""),
        person_b.get("unsafe_act", ""),
    )
    a5_coordination = coordination_agreement(
        person_a.get("coordinated_with", []),
        person_b.get("coordinated_with", []),
    )

    comp = composite_agreement(a1, a2, a3)

    return {
        "a1_action": round(a1, 4),
        "a2_spatial": round(a2, 4),
        "a3_ppe": round(a3, 4),
        "a4_interaction": round(a4_interaction, 4),
        "a4_safety_reasoning": round(a4_safety, 4),
        "a5_coordination": round(a5_coordination, 4),
        "composite": round(comp, 4),
    }


def compute_clip_agreement(vlm_a_annotation, vlm_b_annotation):
    """Compute agreement across all persons in a clip.

    For Layer 2 clips (≤5 workers): matches persons by position, computes
    per-person agreement, averages.
    For Layer 1 clips (6+ workers): computes scene-level agreement only
    (scene_type + worker count).
    """
    persons_a = vlm_a_annotation.get("persons", [])
    persons_b = vlm_b_annotation.get("persons", [])

    # Determine if this is a Layer 1 clip (both VLMs see 6+ workers)
    is_layer1 = len(persons_a) > 5 or len(persons_b) > 5

    if is_layer1:
        # Scene-level agreement only — no per-person matching
        st_agree = scene_type_agreement(
            vlm_a_annotation.get("scene_type", ""),
            vlm_b_annotation.get("scene_type", ""),
        )
        count_a, count_b = len(persons_a), len(persons_b)
        max_count = max(count_a, count_b)
        count_agree = 1.0 - abs(count_a - count_b) / max_count if max_count > 0 else 1.0

        # Composite for Layer 1: scene_type agreement only
        composite = round(st_agree, 4)

        return {
            "a1_action": 0.0, "a2_spatial": 0.0, "a3_ppe": 0.0,
            "a4_interaction": 0.0, "a4_safety_reasoning": 0.0,
            "a5_coordination": 0.0,
            "composite": composite,
            "scene_type_agreement": round(st_agree, 4),
            "worker_count_agreement": round(count_agree, 4),
            "num_persons_a": count_a,
            "num_persons_b": count_b,
            "num_matched": 0,
            "annotation_layer": 1,
        }

    # Layer 2: per-person agreement
    if not persons_a:
        persons_a = [{
            "action_class": vlm_a_annotation.get("action_class", ""),
            "spatial_context": vlm_a_annotation.get("spatial_context", []),
            "ppe": vlm_a_annotation.get("ppe", {}),
            "interaction": vlm_a_annotation.get("interaction", ""),
            "unsafe_act": vlm_a_annotation.get("unsafe_act", ""),
        }]
    if not persons_b:
        persons_b = [{
            "action_class": vlm_b_annotation.get("action_class", ""),
            "spatial_context": vlm_b_annotation.get("spatial_context", []),
            "ppe": vlm_b_annotation.get("ppe", {}),
            "interaction": vlm_b_annotation.get("interaction", ""),
            "unsafe_act": vlm_b_annotation.get("unsafe_act", ""),
        }]

    # Worker count agreement
    count_a, count_b = len(persons_a), len(persons_b)
    max_count = max(count_a, count_b)
    count_agreement = 1.0 - abs(count_a - count_b) / max_count if max_count > 0 else 1.0

    # Match persons by position (greedy)
    used_b = set()
    per_person_scores = []

    for pa in persons_a:
        best_score = -1
        best_idx = -1
        pos_a = pa.get("position", "")

        for i, pb in enumerate(persons_b):
            if i in used_b:
                continue
            pos_b = pb.get("position", "")
            pos_match = 1.0 if pos_a == pos_b else 0.0
            act_match = action_agreement(
                pa.get("action_code", pa.get("action_class", "")),
                pb.get("action_code", pb.get("action_class", "")),
            )
            score = pos_match * 0.3 + act_match * 0.7
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            used_b.add(best_idx)
            person_scores = compute_person_agreement(pa, persons_b[best_idx])
            per_person_scores.append(person_scores)

    # Average across matched persons
    if per_person_scores:
        avg = {}
        for key in per_person_scores[0]:
            avg[key] = round(
                sum(s[key] for s in per_person_scores) / len(per_person_scores), 4
            )
    else:
        avg = {
            "a1_action": 0.0, "a2_spatial": 0.0, "a3_ppe": 0.0,
            "a4_interaction": 0.0, "a4_safety_reasoning": 0.0,
            "a5_coordination": 0.0, "composite": 0.0,
        }

    # Scene type agreement (clip-level)
    st_agree = scene_type_agreement(
        vlm_a_annotation.get("scene_type", ""),
        vlm_b_annotation.get("scene_type", ""),
    )

    avg["scene_type_agreement"] = round(st_agree, 4)
    avg["worker_count_agreement"] = round(count_agreement, 4)
    avg["num_persons_a"] = count_a
    avg["num_persons_b"] = count_b
    avg["num_matched"] = len(per_person_scores)
    avg["annotation_layer"] = 2

    return avg
