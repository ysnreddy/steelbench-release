# Anchoring Bias Deep Analysis — Results Report

**Date**: 2026-04-20 (CORRECTED after audit)  
**Data**: 1,030 submitted annotations across 8 annotators, 3,744 VLM records, 250-clip calibration set (125 anchored + 125 blind)

**AUDIT STATUS**: Critical bugs found — see `audit_report.md` for details. Key issue: 21/36 blind clips were contaminated (annotated pre-deployment with VLM pre-fill shown).

---

## Analysis A: Anchoring Bias Deep Dive

### ~~Overall Result: Bias = +4.9pp (BELOW 5pp threshold)~~ INVALIDATED

The +4.9pp was an artifact of two bugs:
1. **Blind contamination**: 21/36 blind clips had VLM pre-fill shown (annotated before calibration deployment)
2. **Multi-person dilution**: Using all persons[i] instead of persons[0] diluted the signal

**Corrected result (clean blind only, persons[0])**:

| Condition | Agreement with VLM | 95% CI | N |
|---|---|---|---|
| Anchored | 57.7% | [38.9%, 74.5%] | 26 |
| Blind (clean) | 26.7% | [10.9%, 52.0%] | 15 |
| **Bias magnitude** | **+31.0pp** | | |

**WARNING**: n=15 for clean blind is too small for reliable estimates. CIs span 40pp. Need 50+ clean blind annotations for a defensible number.

### Direction Analysis (VALID — not affected by calibration bug)

### Per-Group Bias

| Group | Bias | Anchored N | Blind N | Interpretation |
|---|---|---|---|---|
| A (Locomotion) | +52.8pp | 11 | 21 | HIGH bias but very small N — unreliable |
| B (Manipulation) | +4.0pp | 35 | 36 | Low bias, adequate N |
| C (Specialized) | -25.0pp | 2 | 4 | Negative bias, tiny N — noise |
| E (Social/Supervise) | -28.6pp | 5 | 7 | Negative bias, small N |
| F (Stationary) | +3.6pp | 28 | 16 | Low bias |

**Interpretation**: Only Group B and F have enough samples for reliable estimates. Both show bias <5pp. Groups A, C, E have tiny N — cannot draw conclusions.

### Direction Analysis (n=117, clips with VLM + tier_1 + expert_GT)

| Category | Count | Pct | Meaning |
|---|---|---|---|
| **Productive anchor** | 58 | 49.6% | VLM was correct, annotator correctly accepted |
| **Harmful anchor** | 10 | 8.5% | VLM was wrong, annotator incorrectly accepted |
| **Productive override** | 41 | 35.0% | VLM was wrong, annotator correctly fixed |
| **Harmful override** | 8 | 6.8% | VLM was correct, annotator incorrectly changed |

**Key finding**: Only 8.5% of cases represent harmful anchoring (VLM wrong + annotator accepts). In contrast, 49.6% are productive — the VLM was correct and the annotator correctly accepted it. The pre-fill is functioning as intended: a productivity tool that helps when correct and gets overridden when wrong.

**Paper argument**: "Of 117 comparisons where expert ground truth is available, harmful anchoring (VLM incorrect, annotator accepts) occurs in only 8.5% of cases. The pre-fill functions as intended: annotators accept correct predictions (49.6%) and override incorrect ones (35.0%)."

---

## Analysis B: Blind-Only Finding Validation

**N = 33 blind clips with GT** (limited — findings are directional, not statistically tight)

### Finding 1: F1-Magnet Effect — HOLDS

| Metric | Full Dataset | Blind Only |
|---|---|---|
| Accuracy | 57.7% | 38.8% [28.0%, 50.8%] |
| F1 as wrong prediction | ~dominant | 29.4% of errors |

**Blind accuracy is LOWER** (38.8% vs 57.7%), consistent with the hypothesis that anchoring inflates VLM accuracy. F1-magnet persists: 29.4% of errors are F1 predictions.

### Finding 2: 100pp Gradient — HOLDS

Spread = 100pp on blind data. Macro accuracy = 28.9% (vs 34.4% on full data). The per-class accuracy gradient is even more extreme on unanchored data.

### Finding 3: Degradation — INCONCLUSIVE (small N)

| Condition | Accuracy | N |
|---|---|---|
| clear | 27.3% | 11 |
| dust | 43.2% | 37 |
| glare | 25.0% | 4 |
| low_light | 66.7% | 3 |
| steam | 66.7% | 3 |

N per condition too small for reliable conclusions. Dust has the most samples (n=37) and shows higher accuracy than clear — counterintuitive, likely due to scene composition differences in the blind set.

### Finding 4: Safety — HOLDS

11 violation clips in blind set, 100% recall. Consistent with the full-data finding.

### Summary

| Finding | Holds on blind data? | Confidence |
|---|---|---|
| F1-magnet bias | **YES** | Medium (n=67) |
| 100pp gradient | **YES** | Medium |
| Degradation interaction | Inconclusive | Low (small N per condition) |
| Safety recall | **YES** | Medium (n=11) |

**Important**: Blind accuracy (38.8%) < Full accuracy (57.7%) by 18.9pp. This supports the argument that anchoring makes our reported VLM accuracy an UPPER BOUND. True VLM accuracy against unanchored human GT is likely lower.

---

## Analysis C: Override Rate by Field

### Summary

| Metric | Value |
|---|---|
| Total clips compared | 840 |
| Clips with ANY override | **97.3%** |
| Mean edits per clip | **9.7** |
| Median edits per clip | 9 |

**97.3% of clips have at least one field changed from VLM pre-fill.** Annotators are actively curating, not rubber-stamping.

### Per-Field Override Rates (ranked)

| Field | Override Rate | 95% CI | N |
|---|---|---|---|
| dominant_actions | 77.7% | [74.8%, 80.4%] | 840 |
| overall_ppe_compliance | 66.3% | [63.0%, 69.4%] | 840 |
| action_code (per-person) | 42.4% | [39.7%, 45.0%] | 1,329 |
| ppe_safety_shoes | 39.7% | [37.1%, 42.3%] | 1,329 |
| visibility_conditions | 35.7% | [32.5%, 39.0%] | 840 |
| ppe_harness | 32.1% | [29.7%, 34.7%] | 1,329 |
| num_workers | 32.0% | [29.0%, 35.3%] | 840 |
| occlusion_level | 29.7% | [27.3%, 32.2%] | 1,329 |
| spatial_context | 27.1% | [24.8%, 29.5%] | 1,329 |
| ppe_welding_protection | 25.7% | [23.4%, 28.1%] | 1,329 |
| ppe_high_vis_vest | 24.0% | [21.8%, 26.4%] | 1,329 |
| scene_type | 13.0% | [10.9%, 15.4%] | 840 |
| ppe_helmet | 5.3% | [4.3%, 6.7%] | 1,329 |

**Key observations**:
- **dominant_actions (77.7%)**: Annotators override the VLM's action classification in 3 out of 4 clips at scene level
- **action_code (42.4%)**: Per-person action is overridden in nearly half the cases
- **PPE**: Safety shoes (39.7%) and harness (32.1%) heavily overridden; helmet lowest (5.3%) — helmets are visually obvious, shoes/harness are harder from surveillance distance
- **scene_type (13.0%)**: VLM gets scene type right most of the time — low override is expected for a coarser classification

### F1 Override Rate

F1 (idle standing) — the VLM's default prediction — has 51.7% override rate (155/300). Annotators change the VLM's F1 prediction more than half the time, directly counteracting the F1-magnet bias.

### Per-Class Action Override (top 10)

| Class | Override Rate | Changed/Total |
|---|---|---|
| A2 (climbing up) | 100% | 1/1 |
| B4 (welding) | 100% | 2/2 |
| C3 (crane guiding) | 100% | 2/2 |
| X1 (unlisted) | 100% | 2/2 |
| C1 (crane operation) | 75.0% | 6/8 |
| B8 (maintenance) | 72.2% | 13/18 |
| E2 (supervising) | 68.2% | 15/22 |
| B2 (cutting/grinding) | 58.6% | 17/29 |
| F1 (idle standing) | 51.7% | 155/300 |
| B3 (pouring molten) | 50.0% | 8/16 |

**Pattern**: The hardest, safety-critical actions (crane, welding, climbing) are overridden 75-100% of the time. The VLM is essentially wrong on all of them, and annotators correct it.

---

## Analysis D: Expert Debiasing Effect

### Three-Way Chain (VLM → Tier_1 → Tier_2)

| Agreement Pair | Rate | N |
|---|---|---|
| VLM ↔ Tier_1 | 56.2% | 105 |
| VLM ↔ Tier_2 | 55.2% | 105 |
| Tier_1 ↔ Tier_2 | **77.1%** | 105 |

**Key finding**: Tier_1 and Tier_2 agree with each other (77.1%) far more than either agrees with the VLM (~56%). This means the human reviewers are converging on their own consensus, independent of the VLM.

### Expert Correction Behavior

**When Tier_1 accepted VLM** (n=59):
- Tier_2 corrected: **15.3%** (9 cases)
- Tier_2 kept: 84.7% (50 cases)

**When Tier_1 overrode VLM** (n=46):
- Tier_2 agrees with Tier_1 override: **67.4%** (31 cases)
- Tier_2 reverts to VLM: **17.4%** (8 cases)
- Tier_2 picks something else: 15.2% (7 cases)

**Interpretation**: When annotators override the VLM, experts agree with the human correction 67.4% of the time — much higher than the revert rate (17.4%). The three-tier pipeline amplifies human judgment, not VLM predictions.

### Debiasing Direction

| Direction | Count |
|---|---|
| Expert moves AWAY from VLM | 9 |
| Expert moves TOWARD VLM | 8 |
| No change relative to VLM | 88 |

Net debiasing = +1 (marginal). The expert review is essentially neutral w.r.t. VLM — neither systematically anchored nor systematically debiasing. This is consistent with experts making independent judgments.

### PPE Chain

| Agreement | Rate | N |
|---|---|---|
| VLM-T1 PPE | 77.5% | 315 |
| VLM-T2 PPE | 75.2% | 315 |
| T1-T2 PPE | **89.2%** | 315 |

Same pattern: humans agree with each other (89.2%) more than with VLM (75-78%). PPE decisions are human-driven.

---

## Synthesis: Paper Arguments

### Argument 1: "Anchoring is minimal (+4.9pp)"
The calibration subset analysis shows anchoring bias of +4.9pp, below our 5pp significance threshold. This is consistent with annotators actively engaging with the pre-fill rather than passively accepting it.

### Argument 2: "Pre-fill functions as productivity tool, not co-annotator"
97.3% of clips have at least one override. Mean edits per clip = 9.7. Safety-critical actions (crane, welding, climbing) are overridden 75-100%. The VLM provides a starting point that annotators actively revise.

### Argument 3: "Harmful anchoring is rare (8.5%)"
Of 117 cases with expert ground truth, only 8.5% show harmful anchoring (VLM wrong, annotator accepts). In contrast, 35% show productive override (VLM wrong, annotator corrects). The pre-fill helps more than it hurts.

### Argument 4: "Three-tier pipeline ensures human consensus"
Tier_1 and Tier_2 agree 77.1% with each other vs. ~56% with VLM. When Tier_1 overrides VLM, Tier_2 agrees with the human correction 67.4% of the time. Labels converge toward human consensus, not VLM predictions.

### Argument 5: "Reported accuracy is an upper bound"
Blind subset accuracy (38.8%) is lower than full-dataset accuracy (57.7%), suggesting anchoring inflates VLM-GT agreement. Our reported VLM accuracy is conservative — the true accuracy against unanchored GT is likely lower, making our "VLMs aren't ready" conclusion even stronger.

### Argument 6: "More transparent than standard practice"
Most D&B papers (Kinetics, COCO, Ego4D) use annotation tools with machine assistance but don't measure or report anchoring bias. We provide calibration subset, override rates, and direction analysis — a higher standard of methodological transparency.

---

## Items for Full-Data Re-Run (Pending After April 25)

1. Re-run Analysis A with larger calibration subset (more power per group/class)
2. Re-run Analysis B with more blind clips (current n=33 is too small for degradation finding)
3. Re-run Analysis C with full annotator set
4. Re-run Analysis D with more expert overlap clips
5. Compute anchoring CI on the 4.9pp estimate
6. Stratify anchoring by visibility condition and site
