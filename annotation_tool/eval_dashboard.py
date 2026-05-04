#!/usr/bin/env python3
"""Evaluation & Benchmarking Dashboard for SteelBench.

Separate from the annotation QC dashboard. Displays VLM evaluation results,
novel metrics (M1-M7), degradation analysis, and paper-ready visualizations.

Usage:
    streamlit run annotation_tool/eval_dashboard.py
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from annotation_tool.agreement import ACTION_TAXONOMY, ACTION_GROUPS

# ---------- Paths ----------
EVAL_RESULTS_DIR = os.environ.get(
    "STEELBENCH_EVAL_DIR", os.path.join(PROJECT_ROOT, "eval_results"))
VLM_RESULTS_DIR = os.environ.get(
    "STEELBENCH_VLM_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "vlm_results"))
ANNOTATIONS_DIR = os.environ.get(
    "STEELBENCH_ANNOTATIONS_DIR",
    os.path.join(PROJECT_ROOT, "annotation_tool", "data", "annotations"))


# ---------- Data Loaders ----------

@st.cache_data(ttl=60)
def load_eval_results() -> dict:
    """Load all model evaluation result files from eval_results/."""
    results = {}
    if not os.path.exists(EVAL_RESULTS_DIR):
        return results
    for fname in os.listdir(EVAL_RESULTS_DIR):
        if fname.endswith("_eval.json"):
            model_name = fname.replace("_eval.json", "")
            with open(os.path.join(EVAL_RESULTS_DIR, fname)) as f:
                results[model_name] = json.load(f)
    return results


@st.cache_data(ttl=60)
def load_model_predictions() -> dict:
    """Load all model prediction files from eval_results/."""
    predictions = {}
    if not os.path.exists(EVAL_RESULTS_DIR):
        return predictions
    for fname in os.listdir(EVAL_RESULTS_DIR):
        if fname.endswith("_predictions.json"):
            model_name = fname.replace("_predictions.json", "")
            with open(os.path.join(EVAL_RESULTS_DIR, fname)) as f:
                predictions[model_name] = json.load(f)
    return predictions


@st.cache_data(ttl=60)
def load_model_stats() -> dict:
    """Load per-model stats files."""
    stats = {}
    if not os.path.exists(EVAL_RESULTS_DIR):
        return stats
    for fname in os.listdir(EVAL_RESULTS_DIR):
        if fname.endswith("_stats.json"):
            model_name = fname.replace("_stats.json", "")
            with open(os.path.join(EVAL_RESULTS_DIR, fname)) as f:
                stats[model_name] = json.load(f)
    return stats


@st.cache_data(ttl=60)
def load_novel_metrics() -> dict:
    """Load novel metrics results if available."""
    path = os.path.join(EVAL_RESULTS_DIR, "novel_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_x1_summary() -> dict:
    """Load X1/open vocabulary summary."""
    path = os.path.join(EVAL_RESULTS_DIR, "x1_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------- Dashboard ----------

def main():
    st.set_page_config(
        page_title="SteelBench Evaluation Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("SteelBench — Evaluation & Benchmarking Dashboard")
    st.caption("VLM performance analysis, novel metrics, and paper-ready results")

    eval_results = load_eval_results()
    predictions = load_model_predictions()
    model_stats = load_model_stats()
    novel = load_novel_metrics()
    x1_summary = load_x1_summary()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Per-Model Results",
        "Open Vocabulary (X1)",
        "Novel Metrics (M1-M7)",
        "Degradation Analysis",
        "Model Comparison",
    ])

    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.header("Evaluation Overview")

        # Summary metrics
        n_models = len(eval_results)
        n_predictions = len(predictions)
        n_stats = len(model_stats)

        c1, c2, c3 = st.columns(3)
        c1.metric("Models with Eval Results", n_models)
        c2.metric("Models with Predictions", n_predictions)
        c3.metric("Models with Stats", n_stats)

        if model_stats:
            st.subheader("Model Summary")
            rows = []
            for model_name, stats in model_stats.items():
                rows.append({
                    "Model": stats.get("model_name", model_name),
                    "Total Clips": stats.get("total_clips", 0),
                    "Success": stats.get("success", 0),
                    "Parse Failures": stats.get("parse_failures", 0),
                    "Failure Rate": f"{stats.get('parse_failure_rate', 0):.1%}",
                    "X1 Workers": stats.get("x1_count", 0),
                })
            st.dataframe(rows, use_container_width=True)

        if not eval_results and not predictions and not model_stats:
            st.info(
                "No evaluation data found. Run the pipeline:\n\n"
                "1. `python -m annotation_tool.vlm_annotator --limit 100`\n"
                "2. `python -m annotation_tool.extract_predictions`\n"
                "3. Complete human annotations for ground truth\n"
                "4. Run evaluation: `python -m annotation_tool.run_eval`"
            )

    # ==================== TAB 2: PER-MODEL RESULTS ====================
    with tab2:
        st.header("Per-Model Evaluation Results")

        if eval_results:
            # Build Table 4-style comparison
            st.subheader("Track CL — Closed-Label Results")
            table_rows = []
            for model_name, results in eval_results.items():
                track_cl = results.get("track_cl", {})
                row = {"Model": model_name}

                sa_acc = track_cl.get("action_accuracy_sa", {})
                row["SA Accuracy"] = f"{sa_acc.get('accuracy', 0):.1%}" if sa_acc else "—"

                mai_acc = track_cl.get("action_accuracy_mai", {})
                row["MAI Accuracy"] = f"{mai_acc.get('accuracy', 0):.1%}" if mai_acc else "—"

                mf1 = track_cl.get("macro_f1", {})
                row["Macro F1"] = f"{mf1.get('macro_f1', 0):.3f}" if mf1 else "—"

                st_acc = track_cl.get("scene_type_accuracy", {})
                row["Scene Type Acc"] = f"{st_acc.get('accuracy', 0):.1%}" if st_acc else "—"

                wc = track_cl.get("worker_count_mae", {})
                row["Worker MAE"] = f"{wc.get('mae', 0):.2f}" if wc else "—"

                ppe = track_cl.get("ppe_accuracy", {})
                row["PPE Accuracy"] = f"{ppe.get('overall_accuracy', 0):.1%}" if ppe else "—"

                comp = track_cl.get("composite_clip_accuracy", {})
                row["Composite Acc"] = f"{comp.get('accuracy', 0):.1%}" if comp else "—"

                table_rows.append(row)

            if table_rows:
                st.dataframe(table_rows, use_container_width=True)

            # Track OV
            st.subheader("Track OV — Open-Vocabulary Results")
            ov_rows = []
            for model_name, results in eval_results.items():
                track_ov = results.get("track_ov", {})
                if track_ov:
                    gx = track_ov.get("group_x_selection_rate", {})
                    rc = track_ov.get("reasonable_classification_rate", {})
                    ov_rows.append({
                        "Model": model_name,
                        "X1 Selection Rate": f"{gx.get('rate', 0):.1%}" if gx else "—",
                        "Reasonable Class. Rate": f"{rc.get('rate', 0):.1%}" if rc else "—",
                    })
            if ov_rows:
                st.dataframe(ov_rows, use_container_width=True)
            else:
                st.info("No Track OV results available yet.")

            # Detailed per-model expandable sections
            st.subheader("Detailed Results")
            for model_name, results in eval_results.items():
                with st.expander(f"Details: {model_name}"):
                    st.json(results)
        else:
            st.info("No evaluation results yet. Run eval_metrics.evaluate_all() "
                    "on model predictions vs ground truth.")

    # ==================== TAB 3: OPEN VOCABULARY (X1) ====================
    with tab3:
        st.header("Open Vocabulary (X1) Analysis")

        if x1_summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Clips", x1_summary.get("total_clips", 0))
            c2.metric("VLM-A X1 Clips", x1_summary.get("vlm_a_x1_clips", 0))
            c3.metric("VLM-B X1 Clips", x1_summary.get("vlm_b_x1_clips", 0))
            c4.metric("Both X1", x1_summary.get("both_x1_clips", 0))

            t1, t2 = st.columns(2)
            t1.metric("Taxonomy Coverage (VLM-A)",
                      f"{x1_summary.get('taxonomy_coverage_vlm_a', 0):.1%}")
            t2.metric("Taxonomy Coverage (VLM-B)",
                      f"{x1_summary.get('taxonomy_coverage_vlm_b', 0):.1%}")

            # Closest class distribution
            closest = x1_summary.get("x1_closest_classes", {})
            if closest:
                st.subheader("X1 Closest Existing Classes")
                sorted_closest = sorted(closest.items(), key=lambda x: -x[1])
                st.dataframe(
                    [{"Class": k, "Count": v} for k, v in sorted_closest],
                    use_container_width=True,
                )

            # X1 descriptions
            descriptions = x1_summary.get("x1_descriptions", [])
            if descriptions:
                st.subheader(f"X1 Free-Text Descriptions ({len(descriptions)})")
                st.dataframe(descriptions, use_container_width=True)
        else:
            st.info("Run `python -m annotation_tool.extract_predictions` to generate X1 summary.")

    # ==================== TAB 4: NOVEL METRICS (M1-M7) ====================
    with tab4:
        st.header("Novel Metrics (M1-M7)")

        if novel:
            # M1: AUDC
            m1 = novel.get("m1_audc", {})
            if m1:
                st.subheader("M1: AUDC (Area Under Degradation Curve)")
                a1, a2, a3 = st.columns(3)
                a1.metric("AUDC", f"{m1.get('audc', 0):.4f}")
                a2.metric("nAUDC", f"{m1.get('naudc', 0):.4f}")
                a3.metric("Valid Bins", m1.get("n_valid_bins", 0))

                bin_accs = m1.get("bin_accuracies", [])
                if bin_accs:
                    st.line_chart(
                        {"Accuracy": [a if a is not None else 0 for a in bin_accs]},
                    )

            # M1b: Degradation profile
            m1p = novel.get("m1_degradation_profile", {})
            if m1p:
                st.markdown(f"**Degradation Profile**: {m1p.get('profile', 'unknown')}")
                if m1p.get("profile") == "cliff_edge":
                    st.warning(f"Cliff-edge degradation detected. "
                               f"Tipping point at quality = {m1p.get('tipping_point', '?')}")
                elif m1p.get("profile") == "linear":
                    st.info(f"Linear degradation. Slope = {m1p.get('linear_slope', '?')}, "
                            f"R-squared = {m1p.get('r_squared', '?')}")

            # M2: CRG
            m2 = novel.get("m2_crg", {})
            if m2:
                st.subheader("M2: CRG (Compositional Reasoning Gap)")
                b1, b2, b3 = st.columns(3)
                crg_val = m2.get("crg")
                b1.metric("CRG", f"{crg_val:.4f}" if crg_val is not None else "N/A")
                b2.metric("Perceived Correct", m2.get("perceived_count", 0))
                b3.metric("Composed Correct", m2.get("composed_count", 0))

                # Decomposed CRG
                st.markdown("**Decomposed CRG** (which component failures cause safety errors):")
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("CRG (Action OK)", f"{m2.get('crg_action', 0):.4f}")
                dc2.metric("CRG (Context OK)", f"{m2.get('crg_context', 0):.4f}")
                dc3.metric("CRG (PPE OK)", f"{m2.get('crg_ppe', 0):.4f}")

            # M3: SR@K
            m3 = novel.get("m3_sr_at_k", {})
            if m3:
                st.subheader("M3: SR@K (Safety-Weighted Recall)")
                s1, s2, s3 = st.columns(3)
                s1.metric("SR@critical", f"{m3.get('sr_critical', 0):.4f}")
                s2.metric("SR@major", f"{m3.get('sr_major', 0):.4f}")
                s3.metric("SR@all", f"{m3.get('sr_all', 0):.4f}")

                f1, f2, f3 = st.columns(3)
                f1.metric("FAR@critical", f"{m3.get('far_critical', 0):.4f}")
                f2.metric("FAR@major", f"{m3.get('far_major', 0):.4f}")
                f3.metric("FAR@all", f"{m3.get('far_all', 0):.4f}")

            # M4: DWA
            m4 = novel.get("m4_dwa", {})
            if m4:
                st.subheader("M4: DWA (Distance-Weighted Accuracy)")
                d1, d2 = st.columns(2)
                d1.metric("DWA", f"{m4.get('dwa', 0):.4f}")
                d2.metric("Mean Distance", f"{m4.get('mean_distance', 0):.4f}")

                dist = m4.get("distance_distribution", {})
                if dist:
                    st.bar_chart(dist)

            # M5: TACS
            m5 = novel.get("m5_tacs", {})
            if m5:
                st.subheader("M5: TACS (Transition-Aware Consistency)")
                t1, t2, t3 = st.columns(3)
                t1.metric("TACS (aggregate)", f"{m5.get('tacs_aggregate', 0):.4f}")
                t2.metric("Sequences", m5.get("n_sequences", 0))
                t3.metric("Total Pairs", m5.get("total_pairs", 0))

            # M6: F2-detect
            m6 = novel.get("m6_f2_detect", {})
            if m6:
                st.subheader("M6: F2-detect (Asymmetric Detection)")
                e1, e2, e3 = st.columns(3)
                e1.metric("F2-detect", f"{m6.get('f2_detect', 0):.4f}")
                e2.metric("Precision", f"{m6.get('precision', 0):.4f}")
                e3.metric("Recall", f"{m6.get('recall', 0):.4f}")

            # M7: DRS
            m7 = novel.get("m7_drs", {})
            if m7:
                st.subheader("M7: DRS (Deployment Readiness Score)")
                st.metric("DRS", f"{m7.get('drs', 0):.4f} "
                          f"({m7.get('passes', 0)}/{m7.get('total_thresholds', 6)} passed)")

                per_metric = m7.get("per_metric", {})
                if per_metric:
                    drs_rows = []
                    for metric_name, info in per_metric.items():
                        drs_rows.append({
                            "Metric": metric_name,
                            "Threshold": info.get("threshold", ""),
                            "Value": info.get("value", "N/A"),
                            "Passed": "Yes" if info.get("passed") else "No",
                        })
                    st.dataframe(drs_rows, use_container_width=True)

        else:
            st.info("No novel metrics computed yet. Run evaluation pipeline first.")

    # ==================== TAB 5: DEGRADATION ANALYSIS ====================
    with tab5:
        st.header("Degradation Analysis")

        if eval_results:
            st.subheader("Complexity Degradation (SA -> MAI -> MAC -> SEQ)")
            for model_name, results in eval_results.items():
                track_cl = results.get("track_cl", {})
                degradation = track_cl.get("complexity_degradation", {})
                if degradation:
                    st.markdown(f"**{model_name}**")
                    slope = degradation.get("slope")
                    r2 = degradation.get("r_squared")
                    if slope is not None:
                        st.markdown(f"Slope: {slope:.4f}, R-squared: {r2:.4f}")
                    per_type = degradation.get("per_type_accuracy", {})
                    if per_type:
                        st.bar_chart(per_type)

            st.subheader("Visual Degradation (by Severity S0-S4)")
            st.info("Visual degradation analysis requires severity-tagged clips. "
                    "Enhance visual condition system first (Task #7).")
        else:
            st.info("No evaluation results available yet.")

    # ==================== TAB 6: MODEL COMPARISON ====================
    with tab6:
        st.header("Model Comparison")

        if len(model_stats) >= 2:
            st.subheader("Action Distribution Comparison")
            for model_name, stats in model_stats.items():
                action_dist = stats.get("action_distribution", {})
                if action_dist:
                    with st.expander(f"{stats.get('model_name', model_name)}"):
                        sorted_dist = sorted(action_dist.items(),
                                             key=lambda x: -x[1])[:15]
                        st.bar_chart(
                            {k: v for k, v in sorted_dist}
                        )

            st.subheader("Confidence Score Comparison")
            conf_rows = []
            for model_name, stats in model_stats.items():
                cs = stats.get("confidence_summary", {})
                if cs:
                    conf_rows.append({
                        "Model": stats.get("model_name", model_name),
                        "Mean Confidence": f"{cs.get('mean', 0):.3f}",
                        "Min": f"{cs.get('min', 0):.3f}",
                        "Max": f"{cs.get('max', 0):.3f}",
                    })
            if conf_rows:
                st.dataframe(conf_rows, use_container_width=True)
        elif model_stats:
            st.info("Need at least 2 models for comparison. "
                    "Currently have 1 model's stats.")
        else:
            st.info("No model stats available yet.")


if __name__ == "__main__":
    main()
