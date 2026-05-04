# SteelBench audit protocol

This directory describes how to apply the SteelBench audit protocol —
the methodological contribution introduced in Section 4 of the paper —
to a dataset of your own.

## What the audit measures

The audit empirically bounds the influence of model-assisted annotation
on a human-curated benchmark. It produces four kinds of evidence:

1. **Override rate**: fraction of fields the human annotator changed
   relative to the model pre-fill.
2. **Direction analysis**: of the changed fields, how many moved toward
   ground truth (productive anchoring) versus away (harmful anchoring).
3. **Calibration on a blind subset**: ECE on a held-out set where
   annotators saw no model pre-fill.
4. **Inter-expert agreement**: Cohen's kappa on doubly-annotated clips
   (two experts).

## Code

- `../scripts/anchoring_bias_analysis.py` — full direction analysis.
  Inputs: VLM normalized JSONL, human-finalized JSONL, optional
  expert GT JSONL. Outputs: hit rate, override rate, productive vs
  harmful anchoring shares.
- `../scripts/compute_blind_ece.py` — ECE on the blind subset.
- `../scripts/audit_and_repair_annotations.py` — schema + provenance audit
  (validates the entire annotation tree against the schema).
- `../scripts/audit_status.py` — readiness reporter (used to decide what
  goes into the main eval table).
- `../scripts/ab_prompt_frame_count.py` — prompt-sensitivity A/B test
  for the frame-density ablation (illustrates the audit applied to a
  prompt-design choice).
- `../scripts/test_schema_validator.py` — unit tests for the schema
  validator the audit relies on.

## Data shipped for paper reproducibility

Look in `../paper_reference/anchoring_analysis/`. JSON files:

- `analysis_a_anchoring.json` — direction analysis output
- `analysis_b_blind_validation.json` — blind-set ECE comparison
- `analysis_c_override.json` — override rates per field
- `analysis_d_debiasing.json` — bias-correction sensitivity
- `ece_audit.json` — ECE breakdown by GT provenance
- `full_analysis_summary.json` — combined report

These are the empirical outputs the paper cites.

## Steps to apply the protocol elsewhere

1. **Build a pre-fill set.** Run a single VLM (or any pre-annotator)
   on each clip, save normalized JSON output.

2. **Have humans verify.** With pre-fill shown to the annotator
   ("anchored" condition). Save final JSON.

3. **Build a blind subset.** For ~15-20% of clips, have humans annotate
   without seeing any pre-fill. Save final JSON tagged with
   `calibration_condition: blind`.

4. **Run direction analysis.** Compare pre-fill vs final on the
   anchored set. For each field that changed, ask: did the change
   improve the agreement with expert GT? Productive vs harmful.

5. **Compute calibration on blind.** Compare anchored ECE vs blind
   ECE. The gap is the anchoring effect on calibration.

6. **Inter-expert agreement.** Have ~10% of clips doubly-annotated
   by two domain experts. Cohen's kappa on the per-clip labels.

7. **Compose the four numbers** into the audit table that goes into
   your paper:

```
Override rate                : <X%> [CI lower, upper]
Productive anchoring rate    : <Y%>
Harmful anchoring rate       : <Z%>
Anchored ECE                 : <a>
Blind ECE                    : <b>
Inter-expert kappa           : <k>
```

The paper's Section 4 walks through what each number means and how to
interpret deltas. The point of releasing the audit code: anyone
publishing a model-assisted dataset should report these numbers.
