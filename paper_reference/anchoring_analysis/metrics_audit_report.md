# Full Metrics Audit Report

**Date**: 2026-04-20  
**Scope**: All QC metrics, evaluation metrics, and human baselines  
**Data**: 1,030 submitted annotations, 3,744 VLM records, 156 expert clips

---

## Summary of Issues Found

| # | Metric | Severity | Issue | Impact |
|---|---|---|---|---|
| 1 | IAA | **HIGH** | 96/98 double-annotated pairs have shared VLM anchoring | IAA inflated — both annotators edited same VLM pre-fill |
| 2 | Expert GT | **HIGH** | 118/168 expert annotations directly edited VLM output | ECE, human baseline, direction analysis all use anchored GT |
| 3 | Human Baseline | **HIGH** | Reported 79.5% mixes anchored and proper-chain clips | True proper-chain accuracy is 81.3% (n=75), but "both VLM" clips are 34.1% |
| 4 | ECE | **MEDIUM** | Gold-tier GT is 70% VLM-sourced (expert edited VLM) | ECE may be biased toward VLM confidence patterns |
| 5 | Override Rate | **LOW** | Person count mismatch silently drops extra persons | 135/522 L2 clips have mismatched person counts — understates override |
| 6 | Safety GT | **MEDIUM** | Expert safety violations from VLM-anchored annotations | False-safe rate may be underestimated if experts accepted VLM "no violation" |
| 7 | Missing fields | **LOW** | 22 annotations with no scene_type/dominant_actions | All safety_officer — SO has different schema requirements |
| 8 | Code bugs | **LOW** | macro_f1 KeyError risk, confusion_matrix support calc | See details below |

---

## Issue 1: IAA Inflation from Shared Anchoring (HIGH)

**Finding**: 98 double-annotated clips exist between tier_1 annotators. Of these, 96 pairs have at least one annotation with `calibration_condition=None` (pre-deployment, both saw VLM pre-fill). Only 2 pairs are both truly blind.

**Impact**: The reported IAA (κ=0.568 action, κ=0.697 group) measures "how similarly two annotators edit the same VLM pre-fill" rather than "how independently two humans annotate the same clip." True independent IAA is likely lower.

**What this means for the paper**: IAA reported in the paper is an upper bound. With anchoring, annotators converge toward VLM predictions, artificially inflating agreement.

**Fix**: After April 25, ensure some double-annotated clips are in the blind condition (both annotators don't see VLM). This requires assigning the same blind clip to two different annotators.

---

## Issue 2: Expert GT Independence (HIGH)

**Finding**: Of 168 expert annotations:
- 118 have `source_type: "vlm"` (expert directly edited VLM output)
- 48 have `source_type: "tier_1"` (expert reviewed tier_1 work)

The 118 VLM-sourced break down as:
- 92 are GT batch clips (no tier_1 existed yet — expert was first human annotator)
- 26 are clips where expert annotated before tier_1 (race condition — expert started first)

**Impact**: Expert annotations used as "gold standard" ground truth are themselves VLM-anchored for 70% of clips. This affects:
- **ECE**: Computed against expert GT that was VLM-anchored → ECE may be artificially low
- **Direction analysis**: Used expert as "true" GT, but expert was also VLM-influenced
- **Human baseline**: See Issue 3

**What this means**: The expert GT is not truly independent. However, experts are domain experts with safety training — they are expected to catch and correct VLM errors even when starting from VLM output. The 48 clips with `source_type: "tier_1"` are more trustworthy as GT.

**Fix**: For the paper, report ECE and baselines on the `source_type: "tier_1"` subset (n=48) alongside the full set, and note the difference.

---

## Issue 3: Human Baseline Disaggregation (HIGH)

**Finding**: The reported 79.5% "tier_1 vs expert" accuracy (previously computed) is actually 64.5% when recomputed. More importantly, it mixes fundamentally different comparisons:

| Source Combination | Clips | Accuracy | Interpretation |
|---|---|---|---|
| Both VLM-sourced | 26 | 34.1% | Two people editing same VLM — low agreement shows humans diverge even with same starting point |
| Proper chain (T1=VLM, Exp=tier_1) | 37 | **81.3%** | Expert reviews tier_1 work — correct methodology |
| Other | 9 | 68.2% | Mixed sources |
| **Combined** | **72** | **64.5%** | **Misleading average** |

**Impact**: The 79.5% number previously reported cannot be reproduced on current data (gets 64.5%). The proper-chain subset (81.3%, n=75) is the defensible human baseline.

**What this means for the paper**:
- Report "Tier_1 annotators achieve 81.3% agreement with domain experts on action classification" (n=75)
- The VLM achieves 57.7% → human-VLM gap is +23.6pp (stronger than the +21.2pp previously reported)

---

## Issue 4: ECE Ground Truth Anchoring (MEDIUM)

**Finding**: ECE gold (0.155, n=155) was computed against expert GT that is 70% VLM-sourced. If experts accepted VLM predictions without change, the GT labels match VLM predictions, making the VLM appear well-calibrated.

**Impact**: ECE may be artificially low. The true ECE against independent GT is likely higher (worse calibration).

**Fix**: Recompute ECE on the `source_type: "tier_1"` expert subset (n=48). If ECE increases, report both numbers with explanation.

---

## Issue 5: Override Rate Person Mismatch (LOW)

**Finding**: 135/522 Layer 2 clips have different person counts between human and VLM annotations. The override rate computation uses `min(human_persons, vlm_persons)`, silently dropping extra workers.

**Impact**: Minor — understates override rate slightly. The num_workers field itself is tracked separately (32% override rate).

**Fix**: Not critical. Document in supplementary that override rate for action_code is computed on matched persons only.

---

## Issue 6: Safety GT Anchoring (MEDIUM)

**Finding**: Expert safety violation annotations are 70% VLM-sourced. If the VLM said "no violation" and the expert accepted without checking, real violations could be missed in GT.

**Impact**: False-safe rate (23.6%) may be underestimated. The VLM might be missing more violations than we think.

**Mitigating factor**: Safety violations are high-stakes — domain experts are trained to look for them regardless of pre-fill. Also, safety officer reviews provide a third tier of checking.

**Fix**: Cross-check: for clips where SO reviewed, does SO find violations that expert missed?

---

## Issue 7: Missing Fields in Safety Officer Annotations (LOW)

**Finding**: 22 safety officer annotations have `scene_type=None` and `dominant_actions=None`. All 22 are from the safety_officer role.

**Impact**: These clips would be skipped by any metric that requires scene_type. No impact on tier_1/expert metrics.

**Fix**: Safety officer has different requirements (focused on unsafe_act, not full annotation). These clips should be excluded from scene_type and action accuracy metrics but included in safety metrics. Current code likely handles this correctly via None checks.

---

## Issue 8: Code Bugs (LOW)

From code review:

1. **macro_f1 KeyError risk** (`eval_metrics.py` ~line 395): Assumes all 27 action codes exist in per_class dict. If a code never appears in predictions or GT, `per_class[code]` will KeyError. Fix: use `.get(code, {"f1": 0.0})`.

2. **confusion_matrix support** (`qc_metrics.py` ~line 160): Uses row sum for support, which is the total predictions for that class, not the GT count. Standard convention is support = GT count. Minor impact.

3. **nAUDC edge case** (`novel_metrics.py` ~line 103): If all bins are empty, uses 1.0 as best_bin_acc, which could inflate nAUDC. Unlikely in practice.

---

## Recommended Actions

### Immediate (before paper):
1. **Report human baseline as 81.3%** (proper-chain only, n=75), not 79.5% or 64.5%
2. **Recompute ECE** on tier_1-sourced expert subset (n≈48)
3. **Note IAA caveat**: "IAA reflects agreement between annotators who both had access to VLM pre-fill; true independent IAA is likely lower"
4. **Fix macro_f1 KeyError** in eval_metrics.py

### After April 25 (full data):
5. **Create blind IAA pairs**: Assign same blind clips to two different annotators
6. **Recompute all metrics** with larger sample sizes
7. **Cross-check safety GT** against safety officer findings

### For the paper:
8. Report metrics with source-type disaggregation where possible
9. Acknowledge VLM anchoring as a systematic limitation of all GT-dependent metrics
10. Frame as: "Our reported metrics represent upper bounds on VLM performance and lower bounds on VLM failure severity"
