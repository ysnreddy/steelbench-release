"""Quality control metrics: Cohen's Kappa, confusion matrix, anchoring bias,
decoy catch rate, VLM override rate, and inter-annotator agreement."""

from collections import Counter

import numpy as np


def cohens_kappa(labels_a, labels_b):
    """Compute Cohen's Kappa between two annotators.

    Kappa = (p_o - p_e) / (1 - p_e)
    """
    assert len(labels_a) == len(labels_b), "Label lists must have same length"

    n = len(labels_a)
    if n == 0:
        return 0.0

    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    all_labels = set(labels_a) | set(labels_b)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    p_e = sum((count_a[label] / n) * (count_b[label] / n) for label in all_labels)

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def weighted_cohens_kappa(labels_a, labels_b, weight_func=None):
    """Compute weighted Cohen's Kappa with hierarchical partial credit."""
    from .agreement import CODE_TO_GROUP

    if weight_func is None:
        def weight_func(a, b):
            if a == b:
                return 1.0
            ga = CODE_TO_GROUP.get(a, "")
            gb = CODE_TO_GROUP.get(b, "")
            if ga and gb and ga == gb:
                return 0.5
            return 0.0

    n = len(labels_a)
    if n == 0:
        return 0.0

    all_labels = sorted(set(labels_a) | set(labels_b))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)

    w = np.zeros((k, k))
    for i, li in enumerate(all_labels):
        for j, lj in enumerate(all_labels):
            w[i, j] = 1.0 - weight_func(li, lj)

    observed = np.zeros((k, k))
    for a, b in zip(labels_a, labels_b):
        observed[label_to_idx[a], label_to_idx[b]] += 1
    observed /= n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = np.zeros((k, k))
    for i, li in enumerate(all_labels):
        for j, lj in enumerate(all_labels):
            expected[i, j] = (count_a[li] / n) * (count_b[lj] / n)

    w_observed = np.sum(w * observed)
    w_expected = np.sum(w * expected)

    if w_expected == 0:
        return 1.0

    return 1.0 - w_observed / w_expected


def multi_label_kappa(tags_a_list, tags_b_list, all_possible_tags):
    """Compute per-tag Cohen's Kappa for multi-label annotations (Axis 2)."""
    per_tag_kappas = {}

    for tag in all_possible_tags:
        binary_a = [1 if tag in (tags or []) else 0 for tags in tags_a_list]
        binary_b = [1 if tag in (tags or []) else 0 for tags in tags_b_list]

        if len(set(binary_a)) <= 1 and len(set(binary_b)) <= 1:
            per_tag_kappas[tag] = 1.0 if binary_a == binary_b else 0.0
        else:
            per_tag_kappas[tag] = cohens_kappa(binary_a, binary_b)

    mean_kappa = np.mean(list(per_tag_kappas.values())) if per_tag_kappas else 0.0
    return {"per_tag_kappa": per_tag_kappas, "mean_kappa": round(float(mean_kappa), 4)}


def ppe_kappa(ppe_a_list, ppe_b_list):
    """Compute per-item Cohen's Kappa for PPE annotations (Axis 3)."""
    from .agreement import PPE_ITEMS
    per_item_kappas = {}

    for item in PPE_ITEMS:
        labels_a = [p.get(item, "cannot_determine") for p in ppe_a_list]
        labels_b = [p.get(item, "cannot_determine") for p in ppe_b_list]

        if len(set(labels_a)) <= 1 and len(set(labels_b)) <= 1:
            per_item_kappas[item] = 1.0 if labels_a == labels_b else 0.0
        else:
            per_item_kappas[item] = cohens_kappa(labels_a, labels_b)

    mean_kappa = np.mean(list(per_item_kappas.values())) if per_item_kappas else 0.0
    return {"per_item_kappa": per_item_kappas, "mean_kappa": round(float(mean_kappa), 4)}


def confusion_matrix(labels_true, labels_pred, label_order=None):
    """Build confusion matrix with per-class precision/recall/F1.

    Args:
        labels_true: list of ground truth labels
        labels_pred: list of predicted labels
        label_order: optional list defining row/column order.
            Labels not in label_order are dropped with a warning.

    Returns:
        dict with 'matrix' (2D list), 'labels', 'per_class' metrics, 'n_dropped'
    """
    import warnings
    all_labels = label_order or sorted(set(labels_true) | set(labels_pred))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    n_dropped = 0
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(labels_true, labels_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
        else:
            n_dropped += 1

    if n_dropped > 0 and label_order is not None:
        warnings.warn(
            f"confusion_matrix: {n_dropped} pairs dropped because labels "
            f"were not in label_order ({len(label_order)} labels specified)"
        )

    per_class = {}
    for i, label in enumerate(all_labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[i]),
        }

    return {
        "matrix": matrix,
        "labels": all_labels,
        "per_class": per_class,
    }


def _wilson_ci(overrides_count, total_count):
    if total_count == 0:
        return (0.0, 0.0)
    p = overrides_count / total_count
    z = 1.96
    denom = 1 + z * z / total_count
    center = p + z * z / (2 * total_count)
    spread = z * ((p * (1 - p) / total_count + z * z / (4 * total_count * total_count)) ** 0.5)
    return ((center - spread) / denom, (center + spread) / denom)


def vlm_override_rate(human_annotations, vlm_annotations):
    """Compute rate at which humans changed VLM labels, broken out by layer.

    Layer 2: per-person comparison on persons[0] (action, spatial, ppe).
    Layer 1: scene-level comparison on scene_type, dominant_actions (set
             similarity), scene_unsafe_act (NEW field).

    Layer 1 records have persons=[] so per-person fields are not measured —
    instead the scene-level fields fill the same role. Reporting both layers
    separately closes the silent-exclusion bug where Layer 1 records were
    previously falling back to the legacy `action_class` flat field (which
    is empty in the new schema).

    Args:
        human_annotations: list of annotation dicts (with clip_id, annotation_layer, persons)
        vlm_annotations: dict {clip_id: normalized VLM annotation}

    Returns:
        dict with per-layer override rates + Wilson 95% CIs + a combined view
    """
    # Layer 2 axes: action, spatial, ppe (per-person)
    l2_overrides = {"action": 0, "spatial": 0, "ppe": 0}
    l2_totals = {"action": 0, "spatial": 0, "ppe": 0}
    # Layer 1 axes: scene_type, dominant_actions, scene_unsafe_act
    l1_overrides = {"scene_type": 0, "dominant_actions": 0, "scene_unsafe_act": 0}
    l1_totals = {"scene_type": 0, "dominant_actions": 0, "scene_unsafe_act": 0}
    n_l1_clips = 0
    n_l2_clips = 0

    for ann in human_annotations:
        cid = ann.get("clip_id", "")
        vlm = vlm_annotations.get(cid)
        if not vlm:
            continue
        ann_layer = ann.get("annotation_layer")

        if ann_layer == 2:
            n_l2_clips += 1
            h_persons = ann.get("persons") or []
            v_persons = vlm.get("persons") or []
            if not h_persons or not v_persons:
                continue
            hp = h_persons[0]
            vp = v_persons[0]

            # Action
            h_action = hp.get("action_code", "")
            v_action = vp.get("action_code", "")
            if h_action and v_action:
                l2_totals["action"] += 1
                if h_action != v_action:
                    l2_overrides["action"] += 1

            # Spatial (set comparison)
            h_spatial = set(hp.get("spatial_context", []) or [])
            v_spatial = set(vp.get("spatial_context", []) or [])
            if h_spatial or v_spatial:
                l2_totals["spatial"] += 1
                if h_spatial != v_spatial:
                    l2_overrides["spatial"] += 1

            # PPE (dict comparison)
            h_ppe = hp.get("ppe", {}) or {}
            v_ppe = vp.get("ppe", {}) or {}
            if h_ppe and v_ppe:
                l2_totals["ppe"] += 1
                if h_ppe != v_ppe:
                    l2_overrides["ppe"] += 1

        elif ann_layer == 1:
            n_l1_clips += 1
            # scene_type
            h_st = (ann.get("scene_type") or "").strip()
            v_st = (vlm.get("scene_type") or "").strip()
            if h_st and v_st:
                l1_totals["scene_type"] += 1
                if h_st != v_st:
                    l1_overrides["scene_type"] += 1

            # dominant_actions: derive a VLM "dominant_actions" set from the
            # VLM persons array (the union of all persons[*].action_code)
            h_da = set(ann.get("dominant_actions") or [])
            v_da = set()
            for p in vlm.get("persons") or []:
                ac = p.get("action_code", "")
                if ac:
                    v_da.add(ac)
            if h_da or v_da:
                l1_totals["dominant_actions"] += 1
                if h_da != v_da:
                    l1_overrides["dominant_actions"] += 1

            # scene_unsafe_act: VLM has no scene-level field, but if any of
            # the VLM persons reported an unsafe act, treat that as VLM
            # signaling "unsafe present". Compare presence/absence of the
            # human's scene_unsafe_act vs VLM's any-person-unsafe.
            h_sua = bool((ann.get("scene_unsafe_act") or "").strip())
            v_sua = any(
                (p.get("unsafe_act") or "").strip().lower() not in ("", "none")
                for p in vlm.get("persons") or []
            )
            l1_totals["scene_unsafe_act"] += 1
            if h_sua != v_sua:
                l1_overrides["scene_unsafe_act"] += 1

    def _layer_summary(overrides, totals, n_clips):
        total_comparisons = sum(totals.values())
        total_overrides = sum(overrides.values())
        overall = total_overrides / total_comparisons if total_comparisons > 0 else 0
        ci_low, ci_high = _wilson_ci(total_overrides, total_comparisons)
        return {
            "n_clips": n_clips,
            "overall_override_rate": round(overall, 4),
            "per_axis_rates": {
                k: round((overrides[k] / totals[k]) if totals[k] > 0 else 0, 4)
                for k in overrides
            },
            "axis_totals": dict(totals),
            "axis_overrides": dict(overrides),
            "ci_95": (round(ci_low, 4), round(ci_high, 4)),
        }

    layer_2 = _layer_summary(l2_overrides, l2_totals, n_l2_clips)
    layer_1 = _layer_summary(l1_overrides, l1_totals, n_l1_clips)

    # Combined: weighted by total comparisons across both layers
    n_l2_total = sum(l2_totals.values())
    n_l1_total = sum(l1_totals.values())
    n_total = n_l1_total + n_l2_total
    n_overrides_total = sum(l1_overrides.values()) + sum(l2_overrides.values())
    combined_rate = n_overrides_total / n_total if n_total > 0 else 0
    ci_low, ci_high = _wilson_ci(n_overrides_total, n_total)

    return {
        "total_clips": n_l1_clips + n_l2_clips,
        "n_layer_1": n_l1_clips,
        "n_layer_2": n_l2_clips,
        "layer_1": layer_1,
        "layer_2": layer_2,
        "combined_override_rate": round(combined_rate, 4),
        "combined_ci_95": (round(ci_low, 4), round(ci_high, 4)),
    }


def anchoring_bias_analysis(anchored_results, blind_results):
    """Measure anchoring bias by comparing accuracy in anchored vs blind conditions."""
    anchored_correct = sum(1 for a, gt in anchored_results if a == gt)
    blind_correct = sum(1 for a, gt in blind_results if a == gt)

    anchored_acc = anchored_correct / len(anchored_results) if anchored_results else 0
    blind_acc = blind_correct / len(blind_results) if blind_results else 0

    bias = anchored_acc - blind_acc
    significant = abs(bias) > 0.05

    return {
        "anchored_accuracy": round(anchored_acc, 4),
        "blind_accuracy": round(blind_acc, 4),
        "bias_magnitude": round(bias, 4),
        "significant": significant,
        "n_anchored": len(anchored_results),
        "n_blind": len(blind_results),
        "recommendation": (
            "Anchoring bias detected (>5%). Report in paper."
            if significant
            else "No significant anchoring bias detected."
        ),
    }


def decoy_catch_rate(annotator_decisions, decoy_clip_ids):
    """Compute the rate at which annotators catch deliberately wrong VLM labels."""
    total_decoys = 0
    caught = 0

    for clip_id in decoy_clip_ids:
        if clip_id in annotator_decisions:
            total_decoys += 1
            if annotator_decisions[clip_id].get("changed", False):
                caught += 1

    catch_rate = caught / total_decoys if total_decoys > 0 else 0

    return {
        "total_decoys": total_decoys,
        "caught": caught,
        "catch_rate": round(catch_rate, 4),
        "acceptable": catch_rate >= 0.70,
        "recommendation": (
            "Annotator is vigilant -- catching decoys at acceptable rate."
            if catch_rate >= 0.70
            else "Annotator may need retraining -- low decoy catch rate (<70%)."
        ),
    }


def tier2_adjudication_stats(tier2_annotations, tier1_annotations):
    """Compute statistics on tier_2 adjudication outcomes.

    Args:
        tier2_annotations: list of tier_2 annotation dicts
        tier1_annotations: dict {clip_id: [tier_1 annotations]}

    Returns:
        dict with override patterns and disagreement categories
    """
    total_reviewed = len(tier2_annotations)
    overridden = 0
    override_reasons = Counter()

    for ann in tier2_annotations:
        cid = ann.get("clip_id", "")
        t1_anns = tier1_annotations.get(cid, [])
        if not t1_anns:
            continue

        t2_action = ""
        if ann.get("persons"):
            t2_action = ann["persons"][0].get("action_code", "")

        # Check if tier_2 changed the action from tier_1
        for t1 in t1_anns:
            t1_action = ""
            if t1.get("persons"):
                t1_action = t1["persons"][0].get("action_code", "")
            if t1_action and t2_action and t1_action != t2_action:
                overridden += 1
                override_reasons[f"{t1_action}->{t2_action}"] += 1
                break

    return {
        "total_reviewed": total_reviewed,
        "overridden": overridden,
        "override_rate": round(overridden / total_reviewed, 4) if total_reviewed > 0 else 0,
        "common_overrides": dict(override_reasons.most_common(10)),
    }


def compute_all_iaa(annotations_a, annotations_b):
    """Compute complete inter-annotator agreement across all axes."""
    from .agreement import SPATIAL_TAGS

    def get_primary(ann):
        persons = ann.get("persons", [])
        if persons:
            return persons[0]
        return ann

    primaries_a = [get_primary(a) for a in annotations_a]
    primaries_b = [get_primary(a) for a in annotations_b]

    # A1: Action Class
    actions_a = [p.get("action_code", "") or p.get("action_class", "") for p in primaries_a]
    actions_b = [p.get("action_code", "") or p.get("action_class", "") for p in primaries_b]
    a1_kappa = weighted_cohens_kappa(actions_a, actions_b)

    # A2: Spatial Context
    spatial_a = [p.get("spatial_context", []) for p in primaries_a]
    spatial_b = [p.get("spatial_context", []) for p in primaries_b]
    a2_result = multi_label_kappa(spatial_a, spatial_b, SPATIAL_TAGS)

    # A3: PPE
    ppe_a = [p.get("ppe", {}) for p in primaries_a]
    ppe_b = [p.get("ppe", {}) for p in primaries_b]
    a3_result = ppe_kappa(ppe_a, ppe_b)

    # Scene Type (new)
    scene_a = [a.get("scene_type", "") for a in annotations_a]
    scene_b = [a.get("scene_type", "") for a in annotations_b]
    valid_scene = [(a, b) for a, b in zip(scene_a, scene_b) if a and b]
    scene_kappa = None
    if len(valid_scene) >= 5:
        scene_kappa = cohens_kappa(
            [v[0] for v in valid_scene], [v[1] for v in valid_scene]
        )

    targets = {
        "a1_action": 0.75,
        "a2_spatial": 0.70,
        "a3_ppe": 0.65,
        "safety_judgment": 0.80,
    }

    results = {
        "a1_action_kappa": round(a1_kappa, 4),
        "a1_meets_target": a1_kappa >= targets["a1_action"],
        "a2_spatial_kappa": a2_result["mean_kappa"],
        "a2_meets_target": a2_result["mean_kappa"] >= targets["a2_spatial"],
        "a2_per_tag": a2_result["per_tag_kappa"],
        "a3_ppe_kappa": a3_result["mean_kappa"],
        "a3_meets_target": a3_result["mean_kappa"] >= targets["a3_ppe"],
        "a3_per_item": a3_result["per_item_kappa"],
        "scene_type_kappa": round(scene_kappa, 4) if scene_kappa is not None else None,
        "targets": targets,
        "n_clips": len(annotations_a),
    }

    return results
