# SteelBench Dataset Distribution Report
**Date**: 2026-04-24
**Total clips**: 997 (tier_1 a1-a4 OR expert e1/e2, excluding a10)

## 1. Site Distribution

| Site | Clips | % |
|---|---|---|
| CRM 1&2 | 154 | 15.4% |
| Tippler 3&4 | 130 | 13.0% |
| ASP | 116 | 11.6% |
| CRM | 110 | 11.0% |
| Tippler 1&2 | 101 | 10.1% |
| RERS | 87 | 8.7% |
| Sinter Plant | 76 | 7.6% |
| BF Conveyor | 39 | 3.9% |
| HME | 34 | 3.4% |
| RED | 29 | 2.9% |
| RMHP | 28 | 2.8% |
| SMS 1 | 28 | 2.8% |
| BF CDI | 26 | 2.6% |
| RERS 2 | 21 | 2.1% |
| CO - ASF | 17 | 1.7% |
| TAR Plant | 1 | 0.1% |

## 2. Camera Distribution (top 10)

| Camera | Clips |
|---|---|
| Tippler_1&2_WT1_Load_Side | 97 |
| RERS_RERS-1 | 87 |
| CRM_Work_Shop | 82 |
| ASP_Go_Down_South_Side | 73 |
| Sinter_Plant_Flux_Screening | 64 |
| CRM_1&2_WELD_Mc | 63 |
| Tippler_3&4_19.1-19.2_Tend | 44 |
| BF_Conveyor_JN_21 | 39 |
| Tippler_3&4_wT3LOAD-SIDE | 39 |
| HME_Work_Shop | 34 |

44 total cameras

## 3. Visibility / Severity

**Visibility conditions** (multi-label):
| Condition | Clips |
|---|---|
| dust | 461 |
| clear | 461 |
| glare | 334 |
| low_light | 181 |
| steam | 42 |
| smoke | 12 |

**Severity levels**:
| Level | Clips |
|---|---|
| S0 (clean) | 651 |
| S1 (mild) | 149 |
| S2 (degraded) | 197 |

## 4. Action Class Distribution (clip-level, primary action)

| Code | Name | Clips | Persons | Status |
|---|---|---|---|---|
| F1 | Idle standing | 109 | 575 | |
| A1 | Walking | 306 | 569 | |
| B1 | Operating machinery | 213 | 540 | |
| B5 | Inspection | 59 | 387 | |
| B7 | Supervision | 82 | 387 | |
| E1 | Communicating | 25 | 277 | |
| F2 | Sitting/resting | 33 | 221 | |
| E2 | Supervising | 22 | 185 | |
| B9 | Cleaning | 14 | 127 | BORDERLINE |
| A4 | Carrying | 30 | 105 | |
| B2 | Panel operation | 32 | 73 | |
| A5 | Pushing/pulling | 12 | 53 | BELOW 15 |
| B8 | Maintenance | 10 | 53 | BELOW 15 |
| D2 | Loading/unloading | 8 | 55 | BELOW 15 |
| D1 | Team carry | 4 | 43 | CRITICAL |
| B3 | Welding (arc) | 12 | 41 | BELOW 15 |
| C3 | Crane guiding | 3 | 27 | CRITICAL |
| C2 | Crane operation | 2 | 26 | CRITICAL |
| A2 | Climbing up | 1 | 25 | CRITICAL |
| C1 | Crane signalling | 4 | 22 | CRITICAL |
| B6 | Overhead reaching | 9 | 20 | BELOW 15 |
| A3 | Climbing down | 2 | 18 | CRITICAL |
| B4 | Grinding/cutting | 2 | 11 | CRITICAL |
| C4 | Vehicle operating | 3 | 6 | CRITICAL |
| D3 | Cylinder handling | 0 | 4 | CRITICAL |
| X1 | Unlisted | 0 | 4 | CRITICAL |

## 5. Scene Type Distribution

| Type | Clips | % |
|---|---|---|
| MAI (Multi-Actor Independent) | 698 | 70.0% |
| MAC (Multi-Actor Coordinated) | 184 | 18.5% |
| SA (Single Actor) | 114 | 11.4% |
| SEQ (Sequential) | 1 | 0.1% |

## 6. Worker Count Distribution

| Workers | Clips | Layer |
|---|---|---|
| 1 | 107 | L2 |
| 2 | 179 | L2 |
| 3 | 146 | L2 |
| 4 | 91 | L2 |
| 5 | 81 | L2 |
| 6-10 | 243 | L1 |
| 11-15 | 103 | L1 |
| 16-21 | 47 | L1 |

- Layer 1 (>5 workers): 393 (39.4%)
- Layer 2 (≤5 workers): 604 (60.6%)

## 7. PPE Compliance

| Compliance | Clips |
|---|---|
| Compliant | 637 |
| Partial | 225 |
| Non-compliant | 122 |
| Cannot determine | 13 |

**Per-item** (person-level):
| Item | Worn | Not worn | Cannot det. | N/A |
|---|---|---|---|---|
| Helmet | 1492 | 135 | 64 | 0 |
| High-vis vest | 247 | 1192 | 252 | 0 |
| Welding protection | 15 | 60 | 27 | 1589 |
| Harness | 15 | 123 | 56 | 1497 |
| Safety shoes | 740 | 16 | 935 | 0 |

## 8. Safety — Unsafe Acts

- Clips with violations: 419 (42.0%)
- Clips without: 578 (58.0%)

**Top violation types**:
| Violation | Count |
|---|---|
| no_helmet | 56 |
| no_high_vis_vest | 52 |
| no high vis vest | 30 |
| UA-G-26: No safety goggles | 15 |
| UA-SP-01: No dust mask | 15 |
| UA-G-16: No safety helmet | 15 |
| working at height without harness | 11 |

## 9. Underrepresented Classes — Curation Needed

| Code | Name | Have | Need | Available in VLM pool | Can reach 15? |
|---|---|---|---|---|---|
| D3 | Cylinder handling | 0 | 15 | 1 | NO (short 14) |
| A3 | Climbing down | 2 | 13 | 3 | NO (short 10) |
| C2 | Crane operation | 2 | 13 | 4 | NO (short 9) |
| A2 | Climbing up | 1 | 14 | 7 | NO (short 7) |
| C4 | Vehicle operating | 3 | 12 | 8 | NO (short 4) |
| B4 | Grinding/cutting | 2 | 13 | 14 | YES |
| C3 | Crane guiding | 3 | 12 | 14 | YES |
| C1 | Crane signalling | 4 | 11 | 11 | YES (exact) |
| D1 | Team carry | 4 | 11 | 41 | YES |
| B8 | Maintenance | 10 | 5 | 48 | YES |
| D2 | Loading/unloading | 8 | 7 | 94 | YES |
| B6 | Overhead reaching | 9 | 6 | 6 | YES (exact) |
| A5 | Pushing/pulling | 12 | 3 | 40 | YES |
| B3 | Welding (arc) | 12 | 3 | 78 | YES |
| B9 | Cleaning | 14 | 1 | 52 | YES |
| X1 | Unlisted | 0 | 15 | 116 | YES |

**Total clips to curate from pool: ~110**
**5 classes cannot reach 15**: D3, A3, C2, A2, C4 — genuinely rare in plant footage
