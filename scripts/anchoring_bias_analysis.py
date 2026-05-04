#!/usr/bin/env python3
"""
Anchoring Bias Deep Analysis — 4 analyses for defending transparent VLM pre-fill approach.

Analysis A: Anchoring bias deep dive (per-class, per-layer, direction)
Analysis B: Blind-only finding validation (7 findings on 125 blind clips)
Analysis C: Override rate by field (granular per-field, per-class)
Analysis D: Expert debiasing effect (VLM → tier_1 → tier_2 chain)

Usage:
    python scripts/anchoring_bias_analysis.py \
        --annotations-dir active_batch/annotations \
        --vlm-file annotation_tool/data/vlm_results/vlm_unified.jsonl \
        --calibration-file active_batch/data/calibration_set.json \
        --output-dir paper_reference/anchoring_analysis
"""

import argparse
import json
import glob
import os
import math
from collections import defaultdict, Counter
from pathlib import Path


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for a mean."""
    import random
    if len(values) < 2:
        return (0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return (lo, hi)


# ── Action group mapping ──
ACTION_GROUPS = {
    'A': ['A1', 'A2', 'A3', 'A4', 'A5'],
    'B': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'],
    'C': ['C1', 'C2', 'C3', 'C4'],
    'D': ['D1', 'D2', 'D3'],
    'E': ['E1', 'E2', 'E3'],
    'F': ['F1', 'F2'],
    'X': ['X1'],
}
CODE_TO_GROUP = {}
for g, codes in ACTION_GROUPS.items():
    for c in codes:
        CODE_TO_GROUP[c] = g


def load_annotations(annotations_dir, active_annotators=None):
    """Load all submitted annotations from active annotators."""
    if active_annotators is None:
        active_annotators = [
            'annotator_1', 'annotator_2', 'annotator_3', 'annotator_4',
            'annotator_10', 'expert_1', 'expert_2', 'safety_officer'
        ]

    annotations = {}  # {annotator_id: {clip_id: annotation}}
    for annotator in active_annotators:
        ann_dir = os.path.join(annotations_dir, annotator)
        if not os.path.isdir(ann_dir):
            continue
        annotations[annotator] = {}
        for f in glob.glob(os.path.join(ann_dir, '*.json')):
            with open(f) as fh:
                ann = json.load(fh)
            if ann.get('status') == 'submitted':
                annotations[annotator][ann['clip_id']] = ann
    return annotations


def load_vlm(vlm_file):
    """Load VLM predictions indexed by clip_id."""
    vlm = {}
    with open(vlm_file) as f:
        for line in f:
            rec = json.loads(line.strip())
            clip_id = rec.get('clip_id')
            if clip_id and rec.get('vlm', {}).get('success'):
                vlm[clip_id] = rec['vlm']['normalized']
    return vlm


def load_calibration(calibration_file):
    """Load calibration set — returns anchored_ids and blind_ids."""
    with open(calibration_file) as f:
        cal = json.load(f)
    anchored = set(cal.get('anchored_clips', {}).keys())
    blind = set(cal.get('blind_clips', {}).keys())
    return anchored, blind


def get_action_code(ann):
    """Extract action code(s) from annotation depending on layer."""
    if ann.get('annotation_layer') == 1:
        return ann.get('dominant_actions', [])
    elif ann.get('persons'):
        return [p.get('action_code') for p in ann['persons'] if p.get('action_code')]
    return []


def get_vlm_action_code(vlm_norm):
    """Extract action code(s) from VLM normalized output."""
    if vlm_norm.get('annotation_layer') == 1 or (vlm_norm.get('num_workers', 0) > 5):
        return vlm_norm.get('dominant_actions', [])
    elif vlm_norm.get('persons'):
        return [p.get('action_code') for p in vlm_norm['persons'] if p.get('action_code')]
    return []


# ═══════════════════════════════════════════════════════════════
# ANALYSIS A: Anchoring Bias Deep Dive
# ═══════════════════════════════════════════════════════════════

def analysis_a(annotations, vlm, anchored_ids, blind_ids):
    """Anchoring bias deep dive: per-class, per-layer, direction analysis."""
    results = {
        'overall': {},
        'per_group': {},
        'per_layer': {},
        'direction': {},
        'per_class': {},
    }

    tier1_annotators = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_10']

    # Collect per-person comparisons for anchored vs blind
    anchored_matches = []  # (human_action, vlm_action, clip_id)
    blind_matches = []

    for ann_id in tier1_annotators:
        if ann_id not in annotations:
            continue
        for clip_id, ann in annotations[ann_id].items():
            if clip_id not in vlm:
                continue
            vlm_norm = vlm[clip_id]

            # Determine condition
            if clip_id in anchored_ids:
                condition = 'anchored'
            elif clip_id in blind_ids:
                condition = 'blind'
            else:
                # Not in calibration set — skip for A/B but include for C/D
                continue

            # Compare per-person (Layer 2) or scene-level (Layer 1)
            if ann.get('annotation_layer') == 2 and ann.get('persons') and vlm_norm.get('persons'):
                # Match persons by index (same as eval pipeline)
                for i, person in enumerate(ann['persons']):
                    h_action = person.get('action_code')
                    if i < len(vlm_norm['persons']):
                        v_action = vlm_norm['persons'][i].get('action_code')
                    else:
                        v_action = None
                    if h_action and v_action:
                        entry = (h_action, v_action, clip_id, ann.get('annotation_layer'))
                        if condition == 'anchored':
                            anchored_matches.append(entry)
                        else:
                            blind_matches.append(entry)
            elif ann.get('annotation_layer') == 1:
                h_actions = set(ann.get('dominant_actions', []))
                v_actions = set(vlm_norm.get('dominant_actions', []))
                if h_actions and v_actions:
                    # For Layer 1, compute Jaccard overlap
                    overlap = len(h_actions & v_actions) / len(h_actions | v_actions) if (h_actions | v_actions) else 0
                    entry = (h_actions, v_actions, clip_id, 1, overlap)
                    if condition == 'anchored':
                        anchored_matches.append(entry)
                    else:
                        blind_matches.append(entry)

    # ── Overall anchoring ──
    # For Layer 2: agreement = human action == VLM action
    # Layer 2 entries are (h_action, v_action, clip_id, layer)
    # Layer 1 entries are (h_actions_set, v_actions_set, clip_id, layer, jaccard)
    anch_l2 = [(e[0], e[1]) for e in anchored_matches if e[3] == 2]
    blind_l2 = [(e[0], e[1]) for e in blind_matches if e[3] == 2]

    anch_agree_l2 = sum(1 for h, v in anch_l2 if h == v)
    blind_agree_l2 = sum(1 for h, v in blind_l2 if h == v)

    anch_rate_l2 = anch_agree_l2 / len(anch_l2) if anch_l2 else 0
    blind_rate_l2 = blind_agree_l2 / len(blind_l2) if blind_l2 else 0

    results['overall'] = {
        'layer_2': {
            'anchored_agreement': anch_rate_l2,
            'anchored_n': len(anch_l2),
            'blind_agreement': blind_rate_l2,
            'blind_n': len(blind_l2),
            'bias_magnitude_pp': (anch_rate_l2 - blind_rate_l2) * 100,
            'bias_ci': wilson_ci(anch_agree_l2, len(anch_l2)) if anch_l2 else (0, 0),
        },
        'total_calibration_comparisons': len(anchored_matches) + len(blind_matches),
    }

    # ── Per action group ──
    for group_name, codes in ACTION_GROUPS.items():
        code_set = set(codes)
        a_group = [(h, v) for h, v in anch_l2 if h in code_set or v in code_set]
        b_group = [(h, v) for h, v in blind_l2 if h in code_set or v in code_set]

        if not a_group and not b_group:
            continue

        a_agree = sum(1 for h, v in a_group if h == v)
        b_agree = sum(1 for h, v in b_group if h == v)

        results['per_group'][group_name] = {
            'anchored_agreement': a_agree / len(a_group) if a_group else 0,
            'anchored_n': len(a_group),
            'blind_agreement': b_agree / len(b_group) if b_group else 0,
            'blind_n': len(b_group),
            'bias_pp': ((a_agree / len(a_group) if a_group else 0) -
                       (b_agree / len(b_group) if b_group else 0)) * 100,
        }

    # ── Per action class ──
    for code in sorted(set(h for h, v in anch_l2) | set(h for h, v in blind_l2)):
        a_class = [(h, v) for h, v in anch_l2 if h == code or v == code]
        b_class = [(h, v) for h, v in blind_l2 if h == code or v == code]
        if len(a_class) + len(b_class) < 3:
            continue
        a_ag = sum(1 for h, v in a_class if h == v)
        b_ag = sum(1 for h, v in b_class if h == v)
        results['per_class'][code] = {
            'anchored_agreement': a_ag / len(a_class) if a_class else 0,
            'anchored_n': len(a_class),
            'blind_agreement': b_ag / len(b_class) if b_class else 0,
            'blind_n': len(b_class),
            'bias_pp': ((a_ag / len(a_class) if a_class else 0) -
                       (b_ag / len(b_class) if b_class else 0)) * 100,
        }

    # ── Direction analysis ──
    # When anchored annotator agrees with VLM, was VLM actually correct?
    # We use expert annotations as ground truth
    expert_gt = {}
    for exp_id in ['expert_1', 'expert_2']:
        if exp_id in annotations:
            for clip_id, ann in annotations[exp_id].items():
                if ann.get('annotation_layer') == 2 and ann.get('persons'):
                    expert_gt[clip_id] = ann

    productive_anchor = 0  # VLM correct, annotator accepts (good)
    harmful_anchor = 0     # VLM wrong, annotator accepts (bad)
    productive_override = 0  # VLM wrong, annotator corrects (good)
    harmful_override = 0   # VLM correct, annotator overrides (bad)
    direction_total = 0

    for ann_id in tier1_annotators:
        if ann_id not in annotations:
            continue
        for clip_id, ann in annotations[ann_id].items():
            if clip_id not in vlm or clip_id not in expert_gt:
                continue
            if ann.get('annotation_layer') != 2 or not ann.get('persons'):
                continue

            vlm_norm = vlm[clip_id]
            exp = expert_gt[clip_id]

            for i, person in enumerate(ann['persons']):
                h_action = person.get('action_code')
                v_action = vlm_norm['persons'][i].get('action_code') if i < len(vlm_norm.get('persons', [])) else None
                e_action = exp['persons'][i].get('action_code') if i < len(exp.get('persons', [])) else None

                if not all([h_action, v_action, e_action]):
                    continue

                direction_total += 1
                vlm_correct = (v_action == e_action)
                annotator_agrees_vlm = (h_action == v_action)

                if annotator_agrees_vlm and vlm_correct:
                    productive_anchor += 1
                elif annotator_agrees_vlm and not vlm_correct:
                    harmful_anchor += 1
                elif not annotator_agrees_vlm and not vlm_correct:
                    productive_override += 1
                elif not annotator_agrees_vlm and vlm_correct:
                    harmful_override += 1

    results['direction'] = {
        'total_with_expert_gt': direction_total,
        'productive_anchor': productive_anchor,
        'productive_anchor_pct': productive_anchor / direction_total * 100 if direction_total else 0,
        'harmful_anchor': harmful_anchor,
        'harmful_anchor_pct': harmful_anchor / direction_total * 100 if direction_total else 0,
        'productive_override': productive_override,
        'productive_override_pct': productive_override / direction_total * 100 if direction_total else 0,
        'harmful_override': harmful_override,
        'harmful_override_pct': harmful_override / direction_total * 100 if direction_total else 0,
        'note': 'productive_anchor = VLM correct + annotator accepts; harmful_anchor = VLM wrong + annotator accepts',
    }

    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS B: Blind-Only Finding Validation
# ═══════════════════════════════════════════════════════════════

def analysis_b(annotations, vlm, blind_ids):
    """Re-run key findings using ONLY blind subset clips as ground truth."""
    results = {}

    tier1_annotators = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_10']

    # Collect blind-only human annotations as GT
    blind_gt = {}  # clip_id -> annotation (prefer expert if available)

    # First, experts on blind clips
    for exp_id in ['expert_1', 'expert_2']:
        if exp_id in annotations:
            for clip_id, ann in annotations[exp_id].items():
                if clip_id in blind_ids:
                    blind_gt[clip_id] = ann

    # Then tier_1 on blind clips (if no expert)
    for ann_id in tier1_annotators:
        if ann_id not in annotations:
            continue
        for clip_id, ann in annotations[ann_id].items():
            if clip_id in blind_ids and clip_id not in blind_gt:
                blind_gt[clip_id] = ann

    results['n_blind_clips_with_gt'] = len(blind_gt)

    # Now compare VLM predictions against blind GT
    # ── F1-magnet check ──
    action_correct = 0
    action_total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    vlm_predictions = []
    confusion = defaultdict(lambda: defaultdict(int))
    f1_as_wrong_pred = 0
    total_wrong = 0

    for clip_id, gt in blind_gt.items():
        if clip_id not in vlm:
            continue
        vlm_norm = vlm[clip_id]

        if gt.get('annotation_layer') == 2 and gt.get('persons') and vlm_norm.get('persons'):
            for i, person in enumerate(gt['persons']):
                gt_action = person.get('action_code')
                if i < len(vlm_norm['persons']):
                    pred_action = vlm_norm['persons'][i].get('action_code')
                else:
                    pred_action = None

                if gt_action and pred_action:
                    action_total += 1
                    correct = (gt_action == pred_action)
                    if correct:
                        action_correct += 1
                    else:
                        total_wrong += 1
                        if pred_action == 'F1':
                            f1_as_wrong_pred += 1
                    per_class_correct[gt_action] += int(correct)
                    per_class_total[gt_action] += 1
                    confusion[gt_action][pred_action] += 1

        elif gt.get('annotation_layer') == 1:
            gt_actions = set(gt.get('dominant_actions', []))
            vlm_actions = set(vlm_norm.get('dominant_actions', []))
            if gt_actions and vlm_actions:
                overlap = len(gt_actions & vlm_actions)
                total_in = len(gt_actions | vlm_actions)
                action_total += 1
                if overlap / total_in >= 0.5:
                    action_correct += 1

    blind_accuracy = action_correct / action_total if action_total else 0
    blind_ci = wilson_ci(action_correct, action_total)

    results['finding_1_f1_magnet'] = {
        'blind_accuracy': blind_accuracy,
        'blind_accuracy_pct': blind_accuracy * 100,
        'blind_ci_95': [blind_ci[0] * 100, blind_ci[1] * 100],
        'n_comparisons': action_total,
        'f1_as_wrong_prediction_pct': f1_as_wrong_pred / total_wrong * 100 if total_wrong else 0,
        'total_wrong': total_wrong,
        'f1_wrong_count': f1_as_wrong_pred,
    }

    # ── Per-class gradient ──
    per_class_acc = {}
    for code in sorted(per_class_total.keys()):
        n = per_class_total[code]
        c = per_class_correct[code]
        acc = c / n if n else 0
        ci = wilson_ci(c, n)
        per_class_acc[code] = {
            'accuracy_pct': acc * 100,
            'ci_95': [ci[0] * 100, ci[1] * 100],
            'n': n,
            'correct': c,
        }

    accs = [v['accuracy_pct'] for v in per_class_acc.values()]
    results['finding_2_gradient'] = {
        'per_class': per_class_acc,
        'spread_pp': max(accs) - min(accs) if accs else 0,
        'macro_accuracy_pct': sum(accs) / len(accs) if accs else 0,
    }

    # ── Degradation × complexity (if visibility data available) ──
    vis_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    for clip_id, gt in blind_gt.items():
        if clip_id not in vlm:
            continue
        vlm_norm = vlm[clip_id]
        vis_conds = gt.get('visibility_conditions', ['clear'])
        vis = vis_conds[0] if vis_conds else 'clear'

        if gt.get('annotation_layer') == 2 and gt.get('persons') and vlm_norm.get('persons'):
            for i, person in enumerate(gt['persons']):
                gt_action = person.get('action_code')
                pred_action = vlm_norm['persons'][i].get('action_code') if i < len(vlm_norm.get('persons', [])) else None
                if gt_action and pred_action:
                    vis_accuracy[vis]['total'] += 1
                    if gt_action == pred_action:
                        vis_accuracy[vis]['correct'] += 1

    results['finding_3_degradation'] = {}
    for vis, counts in sorted(vis_accuracy.items()):
        acc = counts['correct'] / counts['total'] if counts['total'] else 0
        results['finding_3_degradation'][vis] = {
            'accuracy_pct': acc * 100,
            'n': counts['total'],
        }

    # ── Safety (if violations in blind set) ──
    safety_clips = 0
    has_violation_gt = 0
    vlm_detected_violation = 0
    false_safe = 0
    false_alarm = 0

    for clip_id, gt in blind_gt.items():
        if clip_id not in vlm:
            continue
        vlm_norm = vlm[clip_id]

        # Check if GT has violations
        gt_has_violation = False
        vlm_has_violation = False

        if gt.get('annotation_layer') == 2 and gt.get('persons'):
            for p in gt['persons']:
                ua = p.get('unsafe_act', '')
                if ua and ua.lower() not in ('none', 'n/a', '', 'safe'):
                    gt_has_violation = True
                    break
        if gt.get('scene_unsafe_act', ''):
            ua = gt['scene_unsafe_act']
            if ua.lower() not in ('none', 'n/a', '', 'safe'):
                gt_has_violation = True

        if vlm_norm.get('persons'):
            for p in vlm_norm['persons']:
                ua = p.get('unsafe_act', '')
                if ua and ua.lower() not in ('none', 'n/a', '', 'safe'):
                    vlm_has_violation = True
                    break

        if gt_has_violation or vlm_has_violation:
            safety_clips += 1
        if gt_has_violation:
            has_violation_gt += 1
            if vlm_has_violation:
                vlm_detected_violation += 1
            else:
                false_safe += 1
        elif vlm_has_violation:
            false_alarm += 1

    recall = vlm_detected_violation / has_violation_gt if has_violation_gt else 0
    results['finding_4_safety'] = {
        'n_clips_with_violations': has_violation_gt,
        'vlm_detected': vlm_detected_violation,
        'recall_pct': recall * 100,
        'false_safe': false_safe,
        'false_safe_rate_pct': false_safe / has_violation_gt * 100 if has_violation_gt else 0,
        'false_alarm': false_alarm,
        'note': 'Limited by N in blind subset',
    }

    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS C: Override Rate by Field
# ═══════════════════════════════════════════════════════════════

def analysis_c(annotations, vlm):
    """Granular override rate analysis by field, class, and condition."""
    results = {
        'per_field': {},
        'per_class_action_override': {},
        'per_ppe_item': {},
        'summary': {},
    }

    tier1_annotators = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_10']

    # Field-level tracking
    field_overrides = defaultdict(lambda: {'changed': 0, 'total': 0})
    per_class_override = defaultdict(lambda: {'changed': 0, 'total': 0})
    per_ppe_override = defaultdict(lambda: {'changed': 0, 'total': 0})
    edit_counts = []

    total_clips = 0
    any_override_clips = 0

    for ann_id in tier1_annotators:
        if ann_id not in annotations:
            continue
        for clip_id, ann in annotations[ann_id].items():
            if clip_id not in vlm:
                continue
            vlm_norm = vlm[clip_id]
            total_clips += 1

            clip_had_override = False

            # Scene-level fields
            # scene_type
            field_overrides['scene_type']['total'] += 1
            if ann.get('scene_type') != vlm_norm.get('scene_type'):
                field_overrides['scene_type']['changed'] += 1
                clip_had_override = True

            # num_workers
            field_overrides['num_workers']['total'] += 1
            if ann.get('num_workers') != vlm_norm.get('num_workers'):
                field_overrides['num_workers']['changed'] += 1
                clip_had_override = True

            # visibility_conditions
            field_overrides['visibility_conditions']['total'] += 1
            h_vis = set(ann.get('visibility_conditions', []))
            v_vis = set(vlm_norm.get('visibility_conditions', []))
            if h_vis != v_vis:
                field_overrides['visibility_conditions']['changed'] += 1
                clip_had_override = True

            # dominant_actions
            field_overrides['dominant_actions']['total'] += 1
            h_da = set(ann.get('dominant_actions', []))
            v_da = set(vlm_norm.get('dominant_actions', []))
            if h_da != v_da:
                field_overrides['dominant_actions']['changed'] += 1
                clip_had_override = True

            # overall_ppe_compliance
            field_overrides['overall_ppe_compliance']['total'] += 1
            if ann.get('overall_ppe_compliance') != vlm_norm.get('overall_ppe_compliance'):
                field_overrides['overall_ppe_compliance']['changed'] += 1
                clip_had_override = True

            # Per-person fields (Layer 2)
            if ann.get('annotation_layer') == 2 and ann.get('persons') and vlm_norm.get('persons'):
                for i, person in enumerate(ann['persons']):
                    if i >= len(vlm_norm['persons']):
                        break
                    v_person = vlm_norm['persons'][i]

                    # action_code
                    h_action = person.get('action_code')
                    v_action = v_person.get('action_code')
                    if h_action and v_action:
                        field_overrides['action_code']['total'] += 1
                        per_class_override[v_action]['total'] += 1
                        if h_action != v_action:
                            field_overrides['action_code']['changed'] += 1
                            per_class_override[v_action]['changed'] += 1
                            clip_had_override = True

                    # spatial_context
                    h_sp = set(person.get('spatial_context', []))
                    v_sp = set(v_person.get('spatial_context', []))
                    if h_sp or v_sp:
                        field_overrides['spatial_context']['total'] += 1
                        if h_sp != v_sp:
                            field_overrides['spatial_context']['changed'] += 1
                            clip_had_override = True

                    # PPE items
                    h_ppe = person.get('ppe', {})
                    v_ppe = v_person.get('ppe', {})
                    for item in ['helmet', 'high_vis_vest', 'welding_protection', 'harness', 'safety_shoes']:
                        h_val = h_ppe.get(item)
                        v_val = v_ppe.get(item)
                        if h_val and v_val:
                            field_overrides[f'ppe_{item}']['total'] += 1
                            per_ppe_override[item]['total'] += 1
                            if h_val != v_val:
                                field_overrides[f'ppe_{item}']['changed'] += 1
                                per_ppe_override[item]['changed'] += 1
                                clip_had_override = True

                    # occlusion_level
                    if person.get('occlusion_level') and v_person.get('occlusion_level'):
                        field_overrides['occlusion_level']['total'] += 1
                        if person['occlusion_level'] != v_person['occlusion_level']:
                            field_overrides['occlusion_level']['changed'] += 1
                            clip_had_override = True

            if clip_had_override:
                any_override_clips += 1

            # Track edit counts from edit_tracking
            et = ann.get('edit_tracking', {})
            mc = et.get('modification_count', 0)
            edit_counts.append(mc)

    # Compile per-field results
    for field, counts in sorted(field_overrides.items()):
        rate = counts['changed'] / counts['total'] if counts['total'] else 0
        ci = wilson_ci(counts['changed'], counts['total'])
        results['per_field'][field] = {
            'override_rate_pct': rate * 100,
            'ci_95': [ci[0] * 100, ci[1] * 100],
            'changed': counts['changed'],
            'total': counts['total'],
        }

    # Per-class action override
    for code in sorted(per_class_override.keys()):
        counts = per_class_override[code]
        rate = counts['changed'] / counts['total'] if counts['total'] else 0
        results['per_class_action_override'][code] = {
            'override_rate_pct': rate * 100,
            'changed': counts['changed'],
            'total': counts['total'],
        }

    # Per-PPE item
    for item in sorted(per_ppe_override.keys()):
        counts = per_ppe_override[item]
        rate = counts['changed'] / counts['total'] if counts['total'] else 0
        ci = wilson_ci(counts['changed'], counts['total'])
        results['per_ppe_item'][item] = {
            'override_rate_pct': rate * 100,
            'ci_95': [ci[0] * 100, ci[1] * 100],
            'changed': counts['changed'],
            'total': counts['total'],
        }

    # Summary
    results['summary'] = {
        'total_clips_compared': total_clips,
        'clips_with_any_override': any_override_clips,
        'clips_with_any_override_pct': any_override_clips / total_clips * 100 if total_clips else 0,
        'mean_edits_per_clip': sum(edit_counts) / len(edit_counts) if edit_counts else 0,
        'median_edits_per_clip': sorted(edit_counts)[len(edit_counts)//2] if edit_counts else 0,
        'max_edits': max(edit_counts) if edit_counts else 0,
        'min_edits': min(edit_counts) if edit_counts else 0,
    }

    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS D: Expert Debiasing Effect
# ═══════════════════════════════════════════════════════════════

def analysis_d(annotations, vlm):
    """Expert debiasing: does tier_2 review reduce VLM anchoring?"""
    results = {
        'three_way_chain': {},
        'expert_correction_on_anchored': {},
        'debiasing_effect': {},
    }

    tier1_annotators = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_10']

    # Find clips where we have all three: VLM + tier_1 + tier_2
    # Expert annotations are keyed by clip_id
    expert_anns = {}
    for exp_id in ['expert_1', 'expert_2']:
        if exp_id in annotations:
            for clip_id, ann in annotations[exp_id].items():
                expert_anns[clip_id] = (exp_id, ann)

    # Find tier_1 annotations for same clips
    tier1_for_clip = {}
    for ann_id in tier1_annotators:
        if ann_id not in annotations:
            continue
        for clip_id, ann in annotations[ann_id].items():
            if clip_id in expert_anns:
                tier1_for_clip[clip_id] = (ann_id, ann)

    # Three-way comparison: VLM → tier_1 → tier_2
    vlm_t1_agree = 0
    vlm_t2_agree = 0
    t1_t2_agree = 0
    total_three_way = 0

    # When tier_1 accepted VLM (no override), does tier_2 correct?
    t1_accepted_vlm = 0
    t2_corrected_after_t1_accept = 0

    # When tier_1 overrode VLM, does tier_2 agree with tier_1 or revert to VLM?
    t1_overrode_vlm = 0
    t2_agrees_t1_override = 0
    t2_reverts_to_vlm = 0

    # Direction: does expert move label AWAY from VLM?
    expert_moves_away = 0
    expert_moves_toward = 0
    expert_stays = 0

    for clip_id in tier1_for_clip:
        if clip_id not in vlm:
            continue

        _, t1_ann = tier1_for_clip[clip_id]
        _, t2_ann = expert_anns[clip_id]
        vlm_norm = vlm[clip_id]

        if t1_ann.get('annotation_layer') != 2 or t2_ann.get('annotation_layer') != 2:
            continue
        if not t1_ann.get('persons') or not t2_ann.get('persons') or not vlm_norm.get('persons'):
            continue

        for i in range(min(len(t1_ann['persons']), len(t2_ann['persons']), len(vlm_norm['persons']))):
            vlm_action = vlm_norm['persons'][i].get('action_code')
            t1_action = t1_ann['persons'][i].get('action_code')
            t2_action = t2_ann['persons'][i].get('action_code')

            if not all([vlm_action, t1_action, t2_action]):
                continue

            total_three_way += 1

            if vlm_action == t1_action:
                vlm_t1_agree += 1
            if vlm_action == t2_action:
                vlm_t2_agree += 1
            if t1_action == t2_action:
                t1_t2_agree += 1

            # Did tier_1 accept VLM?
            if t1_action == vlm_action:
                t1_accepted_vlm += 1
                if t2_action != vlm_action:
                    t2_corrected_after_t1_accept += 1
            else:
                t1_overrode_vlm += 1
                if t2_action == t1_action:
                    t2_agrees_t1_override += 1
                elif t2_action == vlm_action:
                    t2_reverts_to_vlm += 1

            # Direction relative to VLM
            t1_matches_vlm = (t1_action == vlm_action)
            t2_matches_vlm = (t2_action == vlm_action)

            if t1_matches_vlm and not t2_matches_vlm:
                expert_moves_away += 1
            elif not t1_matches_vlm and t2_matches_vlm:
                expert_moves_toward += 1
            else:
                expert_stays += 1

    results['three_way_chain'] = {
        'total_comparisons': total_three_way,
        'vlm_tier1_agreement_pct': vlm_t1_agree / total_three_way * 100 if total_three_way else 0,
        'vlm_tier2_agreement_pct': vlm_t2_agree / total_three_way * 100 if total_three_way else 0,
        'tier1_tier2_agreement_pct': t1_t2_agree / total_three_way * 100 if total_three_way else 0,
    }

    results['expert_correction_on_anchored'] = {
        'tier1_accepted_vlm': t1_accepted_vlm,
        'tier2_corrected_after_accept': t2_corrected_after_t1_accept,
        'correction_rate_pct': t2_corrected_after_t1_accept / t1_accepted_vlm * 100 if t1_accepted_vlm else 0,
        'note': 'When tier_1 accepted VLM prediction, how often did tier_2 expert override?',
    }

    results['expert_override_on_tier1_overrides'] = {
        'tier1_overrode_vlm': t1_overrode_vlm,
        'tier2_agrees_tier1': t2_agrees_t1_override,
        'tier2_reverts_to_vlm': t2_reverts_to_vlm,
        'tier2_agrees_tier1_pct': t2_agrees_t1_override / t1_overrode_vlm * 100 if t1_overrode_vlm else 0,
        'tier2_reverts_pct': t2_reverts_to_vlm / t1_overrode_vlm * 100 if t1_overrode_vlm else 0,
    }

    results['debiasing_effect'] = {
        'expert_moves_away_from_vlm': expert_moves_away,
        'expert_moves_toward_vlm': expert_moves_toward,
        'expert_no_change_relative': expert_stays,
        'net_debiasing': expert_moves_away - expert_moves_toward,
        'debiasing_direction': 'away_from_vlm' if expert_moves_away > expert_moves_toward else 'toward_vlm',
        'note': 'Positive net_debiasing means experts systematically correct VLM-anchored labels',
    }

    # PPE override chain
    ppe_vlm_t1_agree = 0
    ppe_vlm_t2_agree = 0
    ppe_t1_t2_agree = 0
    ppe_total = 0

    for clip_id in tier1_for_clip:
        if clip_id not in vlm:
            continue
        _, t1_ann = tier1_for_clip[clip_id]
        _, t2_ann = expert_anns[clip_id]
        vlm_norm = vlm[clip_id]

        if t1_ann.get('annotation_layer') != 2:
            continue

        for i in range(min(len(t1_ann.get('persons', [])), len(t2_ann.get('persons', [])), len(vlm_norm.get('persons', [])))):
            for item in ['helmet', 'high_vis_vest', 'safety_shoes']:
                v_val = vlm_norm['persons'][i].get('ppe', {}).get(item)
                t1_val = t1_ann['persons'][i].get('ppe', {}).get(item)
                t2_val = t2_ann['persons'][i].get('ppe', {}).get(item)
                if all([v_val, t1_val, t2_val]):
                    ppe_total += 1
                    if v_val == t1_val:
                        ppe_vlm_t1_agree += 1
                    if v_val == t2_val:
                        ppe_vlm_t2_agree += 1
                    if t1_val == t2_val:
                        ppe_t1_t2_agree += 1

    results['ppe_chain'] = {
        'total_comparisons': ppe_total,
        'vlm_tier1_agree_pct': ppe_vlm_t1_agree / ppe_total * 100 if ppe_total else 0,
        'vlm_tier2_agree_pct': ppe_vlm_t2_agree / ppe_total * 100 if ppe_total else 0,
        'tier1_tier2_agree_pct': ppe_t1_t2_agree / ppe_total * 100 if ppe_total else 0,
    }

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Anchoring Bias Deep Analysis')
    parser.add_argument('--annotations-dir', default='active_batch/annotations')
    parser.add_argument('--vlm-file', default='annotation_tool/data/vlm_results/vlm_unified.jsonl')
    parser.add_argument('--calibration-file', default='active_batch/data/calibration_set.json')
    parser.add_argument('--output-dir', default='paper_reference/anchoring_analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    annotations = load_annotations(args.annotations_dir)
    vlm = load_vlm(args.vlm_file)
    anchored_ids, blind_ids = load_calibration(args.calibration_file)

    print(f"  Annotations: {sum(len(v) for v in annotations.values())} total across {len(annotations)} annotators")
    print(f"  VLM records: {len(vlm)}")
    print(f"  Calibration: {len(anchored_ids)} anchored, {len(blind_ids)} blind")

    # Run all 4 analyses
    print("\n" + "="*60)
    print("ANALYSIS A: Anchoring Bias Deep Dive")
    print("="*60)
    res_a = analysis_a(annotations, vlm, anchored_ids, blind_ids)
    with open(os.path.join(args.output_dir, 'analysis_a_anchoring.json'), 'w') as f:
        json.dump(res_a, f, indent=2, default=str)

    print(f"\n  Overall L2 anchored agreement: {res_a['overall'].get('layer_2', {}).get('anchored_agreement', 0)*100:.1f}% (n={res_a['overall'].get('layer_2', {}).get('anchored_n', 0)})")
    print(f"  Overall L2 blind agreement: {res_a['overall'].get('layer_2', {}).get('blind_agreement', 0)*100:.1f}% (n={res_a['overall'].get('layer_2', {}).get('blind_n', 0)})")
    print(f"  Bias magnitude: {res_a['overall'].get('layer_2', {}).get('bias_magnitude_pp', 0):.1f}pp")

    print("\n  Per-group bias:")
    for g, data in sorted(res_a['per_group'].items()):
        print(f"    Group {g}: {data['bias_pp']:+.1f}pp (anch n={data['anchored_n']}, blind n={data['blind_n']})")

    print(f"\n  Direction analysis (n={res_a['direction'].get('total_with_expert_gt', 0)}):")
    print(f"    Productive anchor (VLM correct, accepted): {res_a['direction'].get('productive_anchor_pct', 0):.1f}%")
    print(f"    Harmful anchor (VLM wrong, accepted): {res_a['direction'].get('harmful_anchor_pct', 0):.1f}%")
    print(f"    Productive override (VLM wrong, corrected): {res_a['direction'].get('productive_override_pct', 0):.1f}%")
    print(f"    Harmful override (VLM correct, overridden): {res_a['direction'].get('harmful_override_pct', 0):.1f}%")

    print("\n" + "="*60)
    print("ANALYSIS B: Blind-Only Finding Validation")
    print("="*60)
    res_b = analysis_b(annotations, vlm, blind_ids)
    with open(os.path.join(args.output_dir, 'analysis_b_blind_validation.json'), 'w') as f:
        json.dump(res_b, f, indent=2, default=str)

    print(f"\n  Blind clips with GT: {res_b['n_blind_clips_with_gt']}")
    f1 = res_b.get('finding_1_f1_magnet', {})
    print(f"  Finding 1 (F1-magnet):")
    print(f"    Blind accuracy: {f1.get('blind_accuracy_pct', 0):.1f}% CI [{f1.get('blind_ci_95', [0,0])[0]:.1f}%, {f1.get('blind_ci_95', [0,0])[1]:.1f}%]")
    print(f"    F1 as wrong prediction: {f1.get('f1_as_wrong_prediction_pct', 0):.1f}% of errors")

    f2 = res_b.get('finding_2_gradient', {})
    print(f"  Finding 2 (Gradient): spread={f2.get('spread_pp', 0):.1f}pp, macro={f2.get('macro_accuracy_pct', 0):.1f}%")

    f3 = res_b.get('finding_3_degradation', {})
    print(f"  Finding 3 (Degradation):")
    for vis, data in sorted(f3.items()):
        print(f"    {vis}: {data['accuracy_pct']:.1f}% (n={data['n']})")

    f4 = res_b.get('finding_4_safety', {})
    print(f"  Finding 4 (Safety): {f4.get('n_clips_with_violations', 0)} violation clips, recall={f4.get('recall_pct', 0):.1f}%")

    print("\n" + "="*60)
    print("ANALYSIS C: Override Rate by Field")
    print("="*60)
    res_c = analysis_c(annotations, vlm)
    with open(os.path.join(args.output_dir, 'analysis_c_override.json'), 'w') as f:
        json.dump(res_c, f, indent=2, default=str)

    print(f"\n  Total clips compared: {res_c['summary']['total_clips_compared']}")
    print(f"  Clips with ANY override: {res_c['summary']['clips_with_any_override_pct']:.1f}%")
    print(f"  Mean edits per clip: {res_c['summary']['mean_edits_per_clip']:.1f}")

    print("\n  Per-field override rates:")
    for field, data in sorted(res_c['per_field'].items(), key=lambda x: -x[1]['override_rate_pct']):
        print(f"    {field:30s}: {data['override_rate_pct']:5.1f}% [{data['ci_95'][0]:.1f}%, {data['ci_95'][1]:.1f}%] (n={data['total']})")

    print("\n  Per-PPE item override:")
    for item, data in sorted(res_c['per_ppe_item'].items()):
        print(f"    {item:20s}: {data['override_rate_pct']:5.1f}%  (n={data['total']})")

    print("\n  Per-class action override (top overridden):")
    sorted_classes = sorted(res_c['per_class_action_override'].items(), key=lambda x: -x[1]['override_rate_pct'])
    for code, data in sorted_classes[:10]:
        print(f"    {code}: {data['override_rate_pct']:.1f}% ({data['changed']}/{data['total']})")

    print("\n" + "="*60)
    print("ANALYSIS D: Expert Debiasing Effect")
    print("="*60)
    res_d = analysis_d(annotations, vlm)
    with open(os.path.join(args.output_dir, 'analysis_d_debiasing.json'), 'w') as f:
        json.dump(res_d, f, indent=2, default=str)

    tw = res_d.get('three_way_chain', {})
    print(f"\n  Three-way comparisons: {tw.get('total_comparisons', 0)}")
    print(f"  VLM-Tier1 agreement: {tw.get('vlm_tier1_agreement_pct', 0):.1f}%")
    print(f"  VLM-Tier2 agreement: {tw.get('vlm_tier2_agreement_pct', 0):.1f}%")
    print(f"  Tier1-Tier2 agreement: {tw.get('tier1_tier2_agreement_pct', 0):.1f}%")

    ec = res_d.get('expert_correction_on_anchored', {})
    print(f"\n  When tier_1 accepted VLM (n={ec.get('tier1_accepted_vlm', 0)}):")
    print(f"    Tier_2 corrected: {ec.get('correction_rate_pct', 0):.1f}%")

    eo = res_d.get('expert_override_on_tier1_overrides', {})
    print(f"  When tier_1 overrode VLM (n={eo.get('tier1_overrode_vlm', 0)}):")
    print(f"    Tier_2 agrees with tier_1: {eo.get('tier2_agrees_tier1_pct', 0):.1f}%")
    print(f"    Tier_2 reverts to VLM: {eo.get('tier2_reverts_pct', 0):.1f}%")

    de = res_d.get('debiasing_effect', {})
    print(f"\n  Debiasing direction: {de.get('debiasing_direction', 'unknown')}")
    print(f"    Moves away from VLM: {de.get('expert_moves_away_from_vlm', 0)}")
    print(f"    Moves toward VLM: {de.get('expert_moves_toward_vlm', 0)}")
    print(f"    Net debiasing: {de.get('net_debiasing', 0)}")

    pp = res_d.get('ppe_chain', {})
    print(f"\n  PPE chain (n={pp.get('total_comparisons', 0)}):")
    print(f"    VLM-T1: {pp.get('vlm_tier1_agree_pct', 0):.1f}%")
    print(f"    VLM-T2: {pp.get('vlm_tier2_agree_pct', 0):.1f}%")
    print(f"    T1-T2: {pp.get('tier1_tier2_agree_pct', 0):.1f}%")

    # Save combined summary
    summary = {
        'analysis_a_anchoring': res_a,
        'analysis_b_blind_validation': res_b,
        'analysis_c_override': res_c,
        'analysis_d_debiasing': res_d,
    }
    with open(os.path.join(args.output_dir, 'full_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nResults saved to {args.output_dir}/")
    print("Files: analysis_a_anchoring.json, analysis_b_blind_validation.json,")
    print("       analysis_c_override.json, analysis_d_debiasing.json,")
    print("       full_analysis_summary.json")


if __name__ == '__main__':
    main()
