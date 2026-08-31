"""Build supplementary evaluation figures from locked aggregate result tables.

This script does not fit or select models and does not read patient-level data.
It renders additional publication figures while preserving Figures 1--8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(__file__).resolve().parents[1]
SEMANTIC = PROJECT / "04_results_locked" / "01_semantic_audited_results"
STATE = PROJECT / "05_results_derived" / "03_corrected_eicu_study"
SENSITIVITY = PROJECT / "05_results_derived" / "07_semantic_sensitivity_summary.csv"
CREATININE = (
    PROJECT / "05_results_derived" / "08_eicu_creatinine_observability_audit"
    / "creatinine_observability_by_lookback.csv"
)

DB_ORDER = ["MIMIC-III", "MIMIC-IV", "eICU"]
DIRECTION_ORDER = [
    "M-III → M-IV", "M-III → eICU", "M-IV → M-III",
    "M-IV → eICU", "eICU → M-III", "eICU → M-IV",
]
SEMANTIC_ENDPOINTS = [
    "next_aki_progression", "next_aki_stage2plus", "next_aki_stage3",
    "hospital_death", "icu_death",
]
STATE_ENDPOINTS = [
    "next_aki_progression", "next_aki_stage2_onset", "next_aki_stage3_onset",
    "hospital_death", "icu_death",
]
ENDPOINT_LABELS = {
    "next_aki_progression": "AKI progression",
    "next_aki_stage2plus": "AKI stage 2+",
    "next_aki_stage3": "AKI stage 3",
    "next_aki_stage2_onset": "AKI stage-2 onset",
    "next_aki_stage3_onset": "AKI stage-3 onset",
    "hospital_death": "Hospital death",
    "icu_death": "ICU death",
}
ENDPOINT_SHORT = {
    "next_aki_progression": "AKI progression",
    "next_aki_stage2plus": "AKI stage 2+",
    "next_aki_stage3": "AKI stage 3",
    "next_aki_stage2_onset": "Stage-2 onset",
    "next_aki_stage3_onset": "Stage-3 onset",
    "hospital_death": "Hospital death",
    "icu_death": "ICU death",
}
MODEL_LABELS = {
    "demographic": "Demographic",
    "xgboost": "Concept XGBoost",
    "random_transformer": "Random transformer",
    "semantic_transformer": "Semantic transformer",
    "kdigo_lookup": "KDIGO lookup",
    "state_logistic": "State logistic",
    "state_xgboost": "State XGBoost",
}
MODEL_COLORS = {
    "semantic_transformer": "#E07B39",
    "state_logistic": "#3B6FB6",
    "state_xgboost": "#D64541",
    "demographic": "#7A8CA5",
    "xgboost": "#B64747",
    "random_transformer": "#77A9D8",
    "kdigo_lookup": "#667085",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def direction(frame: pd.DataFrame) -> pd.Series:
    short = {"MIMIC-III": "M-III", "MIMIC-IV": "M-IV", "eICU": "eICU"}
    return frame.source_database.map(short) + " → " + frame.target_database.map(short)


def save(fig, stem: str, figures: Path, outputs: dict[str, str]) -> None:
    pdf = figures / f"{stem}.pdf"
    png = figures / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    outputs[pdf.name] = sha256(pdf)
    outputs[png.name] = sha256(png)


def heatmap_figure(
    frame: pd.DataFrame, models: list[str], endpoints: list[str], title: str,
    stem: str, figures: Path, source: Path, outputs: dict[str, str],
) -> None:
    data = frame.copy()
    data["direction"] = direction(data)
    data["endpoint_label"] = data.endpoint.map(ENDPOINT_SHORT)
    data.to_csv(source / f"{stem}_source_data.csv", index=False)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        subset = data[data.model.eq(model)]
        matrix = subset.pivot(index="direction", columns="endpoint_label", values="auroc")
        matrix = matrix.reindex(
            index=DIRECTION_ORDER,
            columns=[ENDPOINT_SHORT[value] for value in endpoints],
        )
        sns.heatmap(
            matrix, vmin=0.45, vmax=0.82, center=0.60, cmap="vlag",
            annot=True, fmt=".2f", linewidths=.45, linecolor="white",
            cbar=ax is axes[-1], cbar_kws={"label": "AUROC", "shrink": .72}, ax=ax,
        )
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Source → target" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=38)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(title, x=.01, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, stem, figures, outputs)


def common_endpoint_forest(figures: Path, source: Path, outputs: dict[str, str]) -> None:
    semantic = pd.read_csv(SEMANTIC / "external_results.csv")
    semantic_ci = pd.read_csv(SEMANTIC / "hospital_cluster_bootstrap.csv")
    state = pd.read_csv(STATE / "corrected_external_results.csv")
    state_ci = pd.read_csv(STATE / "corrected_hospital_bootstrap.csv")
    endpoints = ["next_aki_progression", "hospital_death", "icu_death"]
    rows = []
    for source_name in ["MIMIC-III", "MIMIC-IV"]:
        q = semantic[
            semantic.source_database.eq(source_name)
            & semantic.target_database.eq("eICU")
            & semantic.model.eq("semantic_transformer")
            & semantic.endpoint.isin(endpoints)
        ]
        ci = semantic_ci[
            semantic_ci.source_database.eq(source_name)
            & semantic_ci.target_database.eq("eICU")
            & semantic_ci.model.eq("semantic_transformer")
            & semantic_ci.endpoint.isin(endpoints)
        ]
        q = q.merge(ci[["endpoint", "auroc_lo", "auroc_hi"]], on="endpoint")
        for row in q.itertuples():
            rows.append({"source_database": source_name, "model": "semantic_transformer", "endpoint": row.endpoint, "auroc": row.auroc, "lo": row.auroc_lo, "hi": row.auroc_hi, "uncertainty_unit": "hospital"})
        for model in ["state_logistic", "state_xgboost"]:
            q = state[
                state.source_database.eq(source_name) & state.target_database.eq("eICU")
                & state.model.eq(model) & state.endpoint.isin(endpoints)
            ]
            ci = state_ci[
                state_ci.source_database.eq(source_name) & state_ci.target_database.eq("eICU")
                & state_ci.model.eq(model) & state_ci.endpoint.isin(endpoints)
            ]
            q = q.merge(ci[["endpoint", "auroc_lo", "auroc_hi"]], on="endpoint")
            for row in q.itertuples():
                rows.append({"source_database": source_name, "model": model, "endpoint": row.endpoint, "auroc": row.auroc, "lo": row.auroc_lo, "hi": row.auroc_hi, "uncertainty_unit": "hospital"})
    data = pd.DataFrame(rows)
    data.to_csv(source / "SupplementaryFigure3_common_endpoint_forest_source_data.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.8), sharex=True)
    models = ["semantic_transformer", "state_logistic", "state_xgboost"]
    offsets = {"semantic_transformer": -.18, "state_logistic": 0, "state_xgboost": .18}
    markers = {"MIMIC-III": "o", "MIMIC-IV": "s"}
    for ax, endpoint in zip(axes, endpoints):
        for source_name in ["MIMIC-III", "MIMIC-IV"]:
            for index, model in enumerate(models):
                row = data[
                    data.source_database.eq(source_name) & data.model.eq(model)
                    & data.endpoint.eq(endpoint)
                ].iloc[0]
                y = index + offsets[model] + (-.06 if source_name == "MIMIC-III" else .06)
                ax.errorbar(
                    row.auroc, y, xerr=[[row.auroc-row.lo], [row.hi-row.auroc]],
                    fmt=markers[source_name], color=MODEL_COLORS[model], ms=4.4,
                    lw=1.1, capsize=2, markerfacecolor="white" if source_name == "MIMIC-III" else MODEL_COLORS[model],
                )
        ax.axvline(.5, color="#9AA0A6", lw=.8, ls="--")
        ax.set_title(ENDPOINT_LABELS[endpoint], fontweight="bold")
        ax.set_xlim(.45, .84)
        ax.set_xlabel("")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS[value] for value in models] if ax is axes[0] else [])
        ax.invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", color="#333333", markerfacecolor="white", ls="", label="MIMIC-III source"),
        plt.Line2D([], [], marker="s", color="#333333", markerfacecolor="#333333", ls="", label="MIMIC-IV source"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.suptitle("Physiological state outperforms semantic context for common eICU endpoints", x=.01, ha="left", fontweight="bold", y=1.08)
    fig.supxlabel("AUROC (hospital-cluster 95% interval)", y=.01)
    fig.tight_layout(rect=(0, .05, 1, .96))
    save(fig, "SupplementaryFigure3_common_endpoint_forest", figures, outputs)


def metric_matrix(figures: Path, source: Path, outputs: dict[str, str]) -> None:
    data = pd.read_csv(SEMANTIC / "lodo_results.csv")
    data = data[
        data.source_database.eq("MIMIC-III + MIMIC-IV") & data.target_database.eq("eICU")
    ].copy()
    data["ap_over_prevalence"] = data.average_precision / data.event_rate
    data.to_csv(source / "SupplementaryFigure4_primary_lodo_metric_matrix_source_data.csv", index=False)
    metrics = [
        ("auroc", "AUROC", "viridis", .48, .68),
        ("ap_over_prevalence", "AP / prevalence", "magma", .85, 2.2),
        ("brier", "Brier score (lower better)", "viridis_r", .06, .29),
        ("ece_10bin", "ECE, 10 bins (lower better)", "viridis_r", 0, .18),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.2))
    for ax, (metric, title, cmap, lower, upper) in zip(axes.flat, metrics):
        matrix = data.pivot(index="model", columns="endpoint", values=metric)
        matrix = matrix.reindex(
            index=["demographic", "xgboost", "random_transformer", "semantic_transformer"],
            columns=SEMANTIC_ENDPOINTS,
        )
        matrix.index = [MODEL_LABELS[value] for value in matrix.index]
        matrix.columns = [ENDPOINT_LABELS[value] for value in matrix.columns]
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, vmin=lower, vmax=upper, linewidths=.45, linecolor="white", cbar_kws={"shrink": .7}, ax=ax)
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Primary MIMIC-III + MIMIC-IV to eICU evaluation is metric-dependent", x=.01, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "SupplementaryFigure4_primary_lodo_metric_matrix", figures, outputs)


def sensitivity_forest(figures: Path, source: Path, outputs: dict[str, str]) -> None:
    lodo = pd.read_csv(SEMANTIC / "lodo_results.csv")
    ci = pd.read_csv(SEMANTIC / "hospital_cluster_bootstrap.csv")
    all_result = lodo[
        lodo.source_database.eq("MIMIC-III + MIMIC-IV") & lodo.target_database.eq("eICU")
        & lodo.model.eq("semantic_transformer")
    ][["endpoint", "auroc"]]
    all_ci = ci[
        ci.evaluation.eq("lodo") & ci.source_database.eq("MIMIC-III + MIMIC-IV")
        & ci.target_database.eq("eICU") & ci.model.eq("semantic_transformer")
    ][["endpoint", "auroc_lo", "auroc_hi"]]
    all_result = all_result.merge(all_ci, on="endpoint")
    all_result["mode"] = "all_concepts"
    sensitivity = pd.read_csv(SENSITIVITY).rename(columns={"auroc_lo": "auroc_lo", "auroc_hi": "auroc_hi"})
    data = pd.concat([all_result, sensitivity[["mode", "endpoint", "auroc", "auroc_lo", "auroc_hi"]]], ignore_index=True)
    data.to_csv(source / "SupplementaryFigure5_semantic_sensitivity_forest_source_data.csv", index=False)
    modes = ["all_concepts", "precise_time", "medication_only", "label_proximal_excluded"]
    mode_labels = {"all_concepts": "All concepts", "precise_time": "Precise time", "medication_only": "Medication only", "label_proximal_excluded": "Label-proximal excluded"}
    colors = dict(zip(modes, ["#E07B39", "#3B6FB6", "#2A9D8F", "#8E6BBE"]))
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 3.6), sharex=True, sharey=True)
    for ax, endpoint in zip(axes, SEMANTIC_ENDPOINTS):
        subset = data[data.endpoint.eq(endpoint)].set_index("mode").reindex(modes)
        for y, mode in enumerate(modes):
            row = subset.loc[mode]
            ax.errorbar(row.auroc, y, xerr=[[row.auroc-row.auroc_lo], [row.auroc_hi-row.auroc]], fmt="o", color=colors[mode], ms=4.5, lw=1.2, capsize=2)
        ax.axvline(.5, color="#9AA0A6", lw=.8, ls="--")
        ax.set_title(ENDPOINT_LABELS[endpoint], fontweight="bold")
        ax.set_xlim(.48, .70)
        ax.set_xlabel("")
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels([mode_labels[value] for value in modes] if ax is axes[0] else [])
        ax.invert_yaxis()
    fig.suptitle("Semantic transport is robust for mortality but not uniformly for AKI", x=.01, ha="left", fontweight="bold", y=1.02)
    fig.supxlabel("AUROC (hospital-cluster 95% interval)", y=.01)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, "SupplementaryFigure5_semantic_sensitivity_forest", figures, outputs)


def creatinine_curve(figures: Path, source: Path, outputs: dict[str, str]) -> None:
    data = pd.read_csv(CREATININE)
    data.to_csv(source / "SupplementaryFigure6_creatinine_observability_source_data.csv", index=False)
    labels = {"4": "4 h", "8": "8 h", "12": "12 h", "24": "24 h", "48": "48 h", "168": "168 h", "any_prior": "Any prior", "any_during_stay": "Any stay"}
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x, data.observed_fraction, color="#2A9D8F", marker="o", lw=2)
    ax.fill_between(x, 0, data.observed_fraction, color="#2A9D8F", alpha=.12)
    for xx, value in zip(x, data.observed_fraction):
        ax.text(xx, min(value + .045, 1.02), f"{value:.1%}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[str(value)] for value in data.lookback_hours])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Episodes with observed creatinine")
    ax.set_xlabel("Lookback availability before the first eligible decision")
    ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    ax.axvline(3.5, color="#667085", lw=.9, ls="--")
    ax.text(3.55, .16, "24-hour availability\nexceeds 92%", color="#4B5563", fontsize=8)
    ax.set_title("Four-hour sparsity does not imply stay-level creatinine absence", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "SupplementaryFigure6_creatinine_observability", figures, outputs)


def evaluation_inventory(source: Path) -> None:
    rows = [
        (1, "Six-direction demographic baseline", "external", "external_results.csv"),
        (2, "Six-direction concept XGBoost", "external", "external_results.csv"),
        (3, "Six-direction random-embedding transformer", "external", "external_results.csv"),
        (4, "Six-direction semantic transformer", "external", "external_results.csv"),
        (5, "Primary two-source LODO to eICU", "external", "lodo_results.csv"),
        (6, "Reciprocal LODO targeting MIMIC-III", "sensitivity", "lodo_results.csv"),
        (7, "Reciprocal LODO targeting MIMIC-IV", "sensitivity", "lodo_results.csv"),
        (8, "Patient-cluster uncertainty", "uncertainty", "patient_cluster_bootstrap.csv"),
        (9, "eICU hospital-cluster uncertainty", "uncertainty", "hospital_cluster_bootstrap.csv"),
        (10, "Target-domain adaptation", "secondary", "target_adaptation_results.csv"),
        (11, "Exact-code support stress test", "secondary", "coding_stress_results.csv"),
        (12, "Concept-positive versus concept-empty strata", "secondary", "concept_availability_results.csv"),
        (13, "Nested source sample-size analysis", "secondary", "sample_size_curves.csv"),
        (14, "Concept zero-occlusion attribution", "exploratory", "concept_importance.csv"),
        (15, "Precise-time semantic sensitivity", "prespecified sensitivity", "04_semantic_sensitivity_precise_time"),
        (16, "Medication-only semantic sensitivity", "prespecified sensitivity", "05_semantic_sensitivity_medication_only"),
        (17, "Label-proximal exclusion sensitivity", "post-hoc sensitivity", "06_semantic_sensitivity_label_proximal_excluded"),
        (18, "Six-direction KDIGO lookup", "physiological state", "corrected_external_results.csv"),
        (19, "Six-direction state logistic regression", "physiological state", "corrected_external_results.csv"),
        (20, "Six-direction state XGBoost", "physiological state", "corrected_external_results.csv"),
        (21, "Source-only calibration and thresholds", "calibration", "corrected_calibration_bins.csv"),
        (22, "Physiological feature-group ablations", "mechanism", "corrected_ablation_results.csv"),
    ]
    pd.DataFrame(rows, columns=["evaluation_id", "evaluation_module", "analysis_class", "authoritative_source"]).to_csv(source / "SupplementaryTable1_evaluation_inventory.csv", index=False)


def detailed_tables(source: Path) -> None:
    state = pd.read_csv(STATE / "corrected_external_results.csv")
    patient = pd.read_csv(STATE / "corrected_patient_bootstrap.csv")
    hospital = pd.read_csv(STATE / "corrected_hospital_bootstrap.csv")
    table2 = state[state.target_database.eq("eICU")].merge(
        patient[["source_database", "target_database", "model", "endpoint", "auroc_lo", "auroc_hi"]].rename(
            columns={"auroc_lo": "patient_auroc_lo", "auroc_hi": "patient_auroc_hi"}
        ), on=["source_database", "target_database", "model", "endpoint"], how="left",
    ).merge(
        hospital[["source_database", "target_database", "model", "endpoint", "auroc_lo", "auroc_hi"]].rename(
            columns={"auroc_lo": "hospital_auroc_lo", "auroc_hi": "hospital_auroc_hi"}
        ), on=["source_database", "target_database", "model", "endpoint"], how="left",
    )
    table2.to_csv(source / "SupplementaryTable2_eicu_state_model_evaluation.csv", index=False)

    lodo = pd.read_csv(SEMANTIC / "lodo_results.csv")
    patient = pd.read_csv(SEMANTIC / "patient_cluster_bootstrap.csv")
    hospital = pd.read_csv(SEMANTIC / "hospital_cluster_bootstrap.csv")
    table3 = lodo[
        lodo.source_database.eq("MIMIC-III + MIMIC-IV") & lodo.target_database.eq("eICU")
    ].copy()
    table3["ap_over_prevalence"] = table3.average_precision / table3.event_rate
    filters = (
        patient.evaluation.eq("lodo") & patient.source_database.eq("MIMIC-III + MIMIC-IV")
        & patient.target_database.eq("eICU")
    )
    patient = patient[filters]
    filters = (
        hospital.evaluation.eq("lodo") & hospital.source_database.eq("MIMIC-III + MIMIC-IV")
        & hospital.target_database.eq("eICU")
    )
    hospital = hospital[filters]
    table3 = table3.merge(
        patient[["model", "endpoint", "auroc_lo", "auroc_hi"]].rename(
            columns={"auroc_lo": "patient_auroc_lo", "auroc_hi": "patient_auroc_hi"}
        ), on=["model", "endpoint"], how="left",
    ).merge(
        hospital[["model", "endpoint", "auroc_lo", "auroc_hi"]].rename(
            columns={"auroc_lo": "hospital_auroc_lo", "auroc_hi": "hospital_auroc_hi"}
        ), on=["model", "endpoint"], how="left",
    )
    table3.to_csv(source / "SupplementaryTable3_primary_lodo_model_evaluation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=PROJECT / "06_figures_locked" / "03_supplementary_evaluation_figures")
    parser.add_argument("--source-data-dir", type=Path, default=PROJECT / "05_results_derived" / "09_supplementary_evaluation_source_data")
    args = parser.parse_args()
    ensure_empty(args.figures_dir)
    ensure_empty(args.source_data_dir)
    style()
    outputs: dict[str, str] = {}
    semantic = pd.read_csv(SEMANTIC / "external_results.csv")
    state = pd.read_csv(STATE / "corrected_external_results.csv")
    heatmap_figure(semantic, ["demographic", "xgboost", "random_transformer", "semantic_transformer"], SEMANTIC_ENDPOINTS, "Complete semantic-model transport across six ICU directions", "SupplementaryFigure1_semantic_transport_heatmaps", args.figures_dir, args.source_data_dir, outputs)
    heatmap_figure(state, ["kdigo_lookup", "state_logistic", "state_xgboost"], STATE_ENDPOINTS, "Complete physiological-state transport across six ICU directions", "SupplementaryFigure2_state_transport_heatmaps", args.figures_dir, args.source_data_dir, outputs)
    common_endpoint_forest(args.figures_dir, args.source_data_dir, outputs)
    metric_matrix(args.figures_dir, args.source_data_dir, outputs)
    sensitivity_forest(args.figures_dir, args.source_data_dir, outputs)
    creatinine_curve(args.figures_dir, args.source_data_dir, outputs)
    evaluation_inventory(args.source_data_dir)
    detailed_tables(args.source_data_dir)
    source_hashes = {path.name: sha256(path) for path in args.source_data_dir.iterdir() if path.is_file()}
    manifest = {
        "status": "SUPPLEMENTARY_EVALUATION_FIGURES_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_fitting_performed": False,
        "patient_level_data_read": False,
        "figures_preserved": "Figures 1-8 unchanged",
        "figure_outputs_sha256": outputs,
        "source_data_sha256": source_hashes,
        "input_sha256": {
            "external_results.csv": sha256(SEMANTIC / "external_results.csv"),
            "lodo_results.csv": sha256(SEMANTIC / "lodo_results.csv"),
            "hospital_cluster_bootstrap.csv": sha256(SEMANTIC / "hospital_cluster_bootstrap.csv"),
            "corrected_external_results.csv": sha256(STATE / "corrected_external_results.csv"),
            "corrected_hospital_bootstrap.csv": sha256(STATE / "corrected_hospital_bootstrap.csv"),
            "semantic_sensitivity_summary.csv": sha256(SENSITIVITY),
            "creatinine_observability_by_lookback.csv": sha256(CREATININE),
        },
        "script_sha256": sha256(Path(__file__)),
        "interpretation_boundary": [
            "descriptive visualization of existing locked or audited aggregate results",
            "no model selection, refitting, target adaptation, or new hypothesis testing",
            "endpoint definitions for stage thresholds and onset risk sets are kept separate",
        ],
    }
    manifest_path = args.source_data_dir / "supplementary_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()