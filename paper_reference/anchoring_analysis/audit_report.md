# Anchoring Bias Audit — Critical Bugs Found

**Date**: 2026-04-20  
**Auditor**: Claude (requested by <Author>)

---

## 4 Bugs Found

### Bug 1: Blind Clip Contamination (CRITICAL)

**21 of 36 blind clips (58.3%) were shown VLM pre-fill.**

Cause: Calibration subset was created during Phase 2 deployment (~Apr 13). But annotators 1-4 had already been annotating since Apr 8. Clips assigned in the pilot phase did NOT have `calibration_condition` set in their assignment records. When these annotators annotated clips that later became part of the calibration set, the VLM pre-fill was shown on ALL of them (including blind clips) because the calibration condition stripping in `app.py` only works if `calibration_condition` is set in the assignment.

**Evidence**:
- All 21 contaminated blind annotations were made Apr 8-12 (pre-deployment)
- All 15 clean blind annotations were made Apr 14-19 (post-deployment)
- Contaminated clips have `edit_tracking.prefilled_from = "vlm"` and `calibration_condition = None`
- Clean clips have `edit_tracking.prefilled_from = None` and `calibration_condition = "blind"`

**Impact**: The "blind" group in all anchoring analyses contains a mix of truly blind and VLM-pre-filled annotations, artificially inflating blind agreement and making the bias appear smaller than it actually is.

### Bug 2: calibration_condition Propagation Failure (CRITICAL)

**42 of 77 calibration-set annotations have `calibration_condition = None`.**

Cause: Same root cause as Bug 1. Annotations created before Phase 2 deployment don't have the field set because the assignment didn't include it.

**Impact**: Any analysis that uses the `calibration_condition` field in the annotation JSON to filter (rather than looking up the calibration_set.json) will miss these clips.

### Bug 3: ALL-persons Dilution (METHODOLOGICAL)

**Analysis A used all persons[i] instead of persons[0], diluting the bias from +19.2pp to +4.9pp.**

Cause: `scripts/anchoring_bias_analysis.py` matched all persons by index, creating 62 anchored and 60 blind pairs from 26+26 clips. The additional persons (workers 2, 3, 4, 5) add noise because:
- Later persons are more ambiguous (further from camera, more occluded)
- VLM predictions for persons[2+] are less reliable
- The signal-to-noise ratio drops for non-primary workers

**Correct approach**: Either use persons[0] only (matching the dashboard) or weight by person index.

### Bug 4: Analysis B (Blind Validation) Used Contaminated GT

**18 of 30 blind clips used as "unanchored GT" were actually VLM-pre-filled.**

Cause: Analysis B filtered by `clip_id in blind_ids` but didn't check whether the annotation was truly blind. The "blind accuracy = 38.8%" is unreliable — it's a weighted average of truly blind and contaminated annotations.

---

## Corrected Numbers

### Anchoring Bias (persons[0], clean blind only)

| Condition | Agreement with VLM | 95% CI | N |
|---|---|---|---|
| Anchored | 57.7% | [38.9%, 74.5%] | 26 |
| Blind (clean) | 26.7% | [10.9%, 52.0%] | 15 |
| **Bias** | **+31.0pp** | | |

**WARNING**: n=15 for blind is very small. CIs are extremely wide. This number will change significantly with more data.

### Comparison of All Methods

| Method | Anchored | Blind | Bias | Issue |
|---|---|---|---|---|
| Dashboard (persons[0], all annotators, contaminated) | 60.0% (n=30) | 41.4% (n=29) | +18.6pp | Includes contaminated blind + experts |
| Analysis A (ALL persons, tier_1, contaminated) | 53.2% (n=62) | 48.3% (n=60) | +4.9pp | Diluted by multi-person + contaminated |
| Tier_1 only (persons[0], contaminated) | 57.7% (n=26) | 38.5% (n=26) | +19.2pp | Still includes contaminated blind |
| **CORRECT (persons[0], tier_1, clean only)** | **57.7% (n=26)** | **26.7% (n=15)** | **+31.0pp** | Small N, wide CIs |

### What the +21.3pp Was

The +21.3pp reported in `experiment_analysis_report.md` was likely from an earlier computation that used a similar methodology to the dashboard (persons[0], mixed contamination). The true bias on clean data is potentially **higher** (+31.0pp), but this is unreliable due to n=15.

---

## Root Cause

The calibration infrastructure was deployed AFTER annotators had already started work. The system design assumes calibration conditions are set at assignment time, but pilot-phase assignments didn't have this field. When the calibration_set.json was later created, it retroactively assigned clips to anchored/blind groups, but the annotations for those clips had already been completed WITH VLM pre-fill shown on ALL of them.

**The fix path is**:
1. For already-annotated clips: mark as contaminated, exclude from anchoring analysis
2. For future clips: ensure calibration_condition is in the assignment BEFORE the annotator sees the clip
3. Wait for more clean blind annotations (currently only 15 — need at least 50+ for reliable estimates)

---

## Impact on Paper Claims

### Previous claim: "Anchoring bias = +21.3pp, bounded by calibration subset"
- **Status**: The 21.3pp is not reliably computed. True bias could be anywhere from +15pp to +40pp given current CIs.

### Analysis A claim: "Anchoring bias = +4.9pp, below significance"
- **Status**: WRONG. This was an artifact of contamination + multi-person dilution.

### Direction analysis claim: "Harmful anchoring = 8.5%"
- **Status**: This used expert GT, not calibration conditions. Still valid but should be verified.

### Override rate claim: "97.3% of clips have at least one override"
- **Status**: Still valid — this doesn't depend on calibration conditions.

### Expert debiasing claim: "T1-T2 agreement 77.1%"
- **Status**: Still valid — this doesn't depend on calibration conditions.

---

## Recommendations

1. **Do NOT report +4.9pp in the paper** — it's an artifact
2. **Do NOT report +31.0pp yet** — n=15 is too small
3. **Wait for more clean blind annotations** (target: 50+ blind clips annotated post-deployment)
4. **Report the methodology honestly**: describe the calibration design, note that early annotations were contaminated, report the clean-only estimate with wide CIs
5. **The override rate (97.3%) and direction analysis (8.5% harmful) are the stronger arguments** — these don't depend on the calibration subset integrity
6. **Fix the propagation**: backfill `calibration_condition` in existing annotations where possible; ensure all future annotations have it set
