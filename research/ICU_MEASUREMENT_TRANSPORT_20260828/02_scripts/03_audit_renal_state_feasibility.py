"""Read-only feasibility audit for a longitudinal renal-state prediction model.

The audit never modifies source data and exports no row-level identifiers.  It
checks what the current four-hour transition contract can support before any
dynamic model is trained, with particular attention to temporal leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
TRUST_ROOT = Path(os.environ.get("TRUST_AKI_ROOT", str(PROJECT.parent / "outputs" / "trust_aki")))
TRANSITION_ROOT = TRUST_ROOT / "coarse_clinical_v1" / "adult_full_v002_contractfix"
DATABASES = {
    "MIMIC-III": TRANSITION_ROOT / "mimic3_coarse_clinical_v1_transitions.csv",
    "MIMIC-IV": TRANSITION_ROOT / "mimic4_coarse_clinical_v1_transitions.csv",
    "eICU": TRANSITION_ROOT / "eicu_coarse_clinical_v1_transitions.csv",
}
TIME_COLUMNS = [
    "state_window_start", "state_window_end", "decision_time",
    "action_window_start", "action_window_end", "next_state_time",
]
STATE_COLUMNS = [f"s_{i}" for i in range(15)]
STATE_NAMES = [
    "heart_rate", "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "respiratory_rate", "spo2", "temperature", "creatinine", "lactate",
    "white_blood_cells", "platelets", "bun", "current_kdigo_stage",
    "predecision_vasopressor", "prior_aki_trend",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_values(frame: pd.DataFrame, column: str, axis: str) -> pd.Series:
    if axis == "relative_hours":
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def hours_between(later: pd.Series, earlier: pd.Series, axis: str) -> pd.Series:
    if axis == "relative_hours":
        return later - earlier
    return (later - earlier).dt.total_seconds() / 3600.0


def quantiles(values: pd.Series) -> dict[str, float]:
    result = values.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    return {str(key): float(value) for key, value in result.items()}


def audit_database(name: str, path: Path) -> dict:
    columns = [
        "subject_id", "episode_id", "time_axis", "action_valid", "done",
        *TIME_COLUMNS, *STATE_COLUMNS, "next_s_12",
        "fluid_ml_per_kg_4h", "vasopressor_any_4h",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[pd.to_numeric(frame["action_valid"], errors="coerce").eq(1)].copy()
    axis_values = frame["time_axis"].dropna().astype(str).unique()
    if len(axis_values) != 1:
        raise ValueError(f"{name}: expected one time axis, found {axis_values.tolist()}")
    axis = axis_values[0]
    parsed = {column: time_values(frame, column, axis) for column in TIME_COLUMNS}
    state_width = hours_between(parsed["state_window_end"], parsed["state_window_start"], axis)
    state_to_decision = hours_between(parsed["decision_time"], parsed["state_window_end"], axis)
    decision_to_action = hours_between(parsed["action_window_start"], parsed["decision_time"], axis)
    action_width = hours_between(parsed["action_window_end"], parsed["action_window_start"], axis)
    action_to_next = hours_between(parsed["next_state_time"], parsed["action_window_end"], axis)

    ordered = frame.assign(_decision=parsed["decision_time"]).sort_values(
        ["episode_id", "_decision"], kind="mergesort"
    )
    episode = ordered.groupby("episode_id", sort=False)
    window_count = episode.size()
    first_decision = episode["_decision"].transform("min")
    history_hours = hours_between(ordered["_decision"], first_decision, axis)
    max_history = history_hours.groupby(ordered["episode_id"]).max()
    decision_gap = episode["_decision"].diff()
    if axis != "relative_hours":
        decision_gap = decision_gap.dt.total_seconds() / 3600.0

    missingness = {}
    for column, label in zip(STATE_COLUMNS, STATE_NAMES):
        missingness[label] = float(pd.to_numeric(frame[column], errors="coerce").isna().mean())

    return {
        "database": name,
        "input": str(path),
        "input_sha256": sha256(path),
        "time_axis": axis,
        "n_rows": int(len(frame)),
        "n_episodes": int(frame["episode_id"].nunique()),
        "n_patients": int(frame["subject_id"].astype(str).nunique()),
        "windows_per_episode": quantiles(window_count),
        "max_observed_history_hours": quantiles(max_history),
        "episode_history_availability": {
            "at_least_24h": float((max_history >= 24 - 1e-6).mean()),
            "at_least_48h": float((max_history >= 48 - 1e-6).mean()),
            "at_least_72h": float((max_history >= 72 - 1e-6).mean()),
        },
        "temporal_contract_checks": {
            "state_window_exactly_4h": bool(np.isclose(state_width, 4, atol=1e-6).all()),
            "state_end_equals_decision": bool(np.isclose(state_to_decision, 0, atol=1e-6).all()),
            "action_starts_at_decision": bool(np.isclose(decision_to_action, 0, atol=1e-6).all()),
            "action_window_exactly_4h": bool(np.isclose(action_width, 4, atol=1e-6).all()),
            "next_state_equals_action_end": bool(np.isclose(action_to_next, 0, atol=1e-6).all()),
            "successive_decisions_4h_when_present": bool(
                np.isclose(decision_gap.dropna(), 4, atol=1e-6).all()
            ),
        },
        "observability": {
            "state_missingness": missingness,
            "current_action_complete": bool(
                pd.to_numeric(frame["fluid_ml_per_kg_4h"], errors="coerce").notna().all()
                and pd.to_numeric(frame["vasopressor_any_4h"], errors="coerce").notna().all()
            ),
            "current_kdigo_complete": bool(pd.to_numeric(frame["s_12"], errors="coerce").notna().all()),
            "next_kdigo_observed_rate": float(pd.to_numeric(frame["next_s_12"], errors="coerce").notna().mean()),
            "terminal_transition_rate": float(pd.to_numeric(frame["done"], errors="coerce").eq(1).mean()),
        },
    }


def gold_headers() -> list[dict]:
    rows = []
    for path in sorted(TRUST_ROOT.rglob("*.gold.csv")):
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        rows.append({"path": str(path), "bytes": path.stat().st_size, "columns": columns})
    return rows


def run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    databases = [audit_database(name, path) for name, path in DATABASES.items()]
    # Availability must be stated by data layer. A field absent from the
    # modelling transition is not thereby absent from the source database.
    eicu_layer_availability = {
        "hospital_id": {
            "raw_source": "patient.hospitalid (present)",
            "gold": "not retained in audited Gold candidates",
            "transition": "not retained",
        },
        "icu_and_hospital_discharge": {
            "raw_source": "patient unit/hospital discharge offsets, statuses, and locations (present)",
            "gold": "episode outcomes retained; discharge offsets not retained in current Gold artifacts",
            "transition": "episode outcomes retained; event offsets not retained",
        },
        "death_time": {
            "raw_source": "no independent death timestamp; Expired discharge status plus discharge offset supplies a source-recorded event time with explicit provenance",
            "gold": "not retained as an event-time field",
            "transition": "not retained as an event-time field",
        },
        "urine_output": {
            "raw_source": "intakeOutput item paths, labels, and numeric values (present and used for KDIGO)",
            "gold": "used in KDIGO components but raw urine amount/interval not retained as model fields",
            "transition": "not retained",
        },
        "fluid_input_output_net_balance": {
            "raw_source": "intakeOutput intake/output/net totals plus item-level cells (present)",
            "gold": "narrow crystalloid action retained; complete balance not retained",
            "transition": "narrow crystalloid action retained; complete balance not retained",
        },
        "direct_kdigo_observation_flags": {
            "raw_source": "derivable only after component-specific ETL",
            "gold": "rebuild_audit_v003 contains direct and carried-forward stage flags",
            "transition": "not retained in current coarse contract",
        },
        "standard_and_custom_labs": {
            "raw_source": "lab and customLab tables are present",
            "gold": "selected standard labs retained; customLab coverage not established",
            "transition": "selected state labs retained per four-hour window",
        },
        "vasopressor_dose": {
            "raw_source": "infusionDrug dose, rate, volume, and patient-weight fields are present",
            "gold": "binary exposure retained; equivalent dose not standardized",
            "transition": "binary exposure retained; equivalent dose not retained",
        },
        "longitudinal_history": {
            "raw_source": "source offsets can extend beyond 48 hours",
            "gold": "current build is truncated by maximum_followup_hours=48",
            "transition": "at most 12 windows/about 44 hours between decisions",
        },
    }
    report = {
        "status": "RENAL_STATE_PHASE1_FEASIBILITY_AUDITED_V2_LAYER_AWARE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "cohort_anchor": "first confirmed AKI index; supports post-AKI progression/recovery, not incident AKI onset",
        "databases": databases,
        "gold_candidates": gold_headers(),
        "eicu_layer_availability": eicu_layer_availability,
        "current_transition_contract": {
            "available": [
                "four-hour repeated 15-variable clinical states",
                "current and next KDIGO state",
                "post-decision four-hour fluid and vasopressor action",
                "episode-level death outcomes",
            ],
            "interpretation": "availability statements apply to this transition layer only; see eicu_layer_availability for raw and Gold provenance",
            "required_but_not_retained_or_not_harmonized": [
                "baseline creatinine candidates and provenance",
                "urine amount/interval and rolling mL/kg/h",
                "complete fluid input/output/net balance",
                "vasopressor equivalent dose",
                "KRT timing and modality",
                "discharge/death event times with provenance",
                "eICU hospital ID",
                "direct/carried-forward KDIGO flags"
            ],
        },
        "leakage_rules": [
            "At landmark t, use state windows ending at or before decision_time t only.",
            "fluid_ml_per_kg_4h and vasopressor_any_4h belong to [t,t+4h) and are forbidden as predictors at t.",
            "The same action variables may enter only as lagged history at a later landmark.",
            "next_s_* and any measurement after decision_time are outcomes, never predictors.",
            "All splits and bootstrap samples must be clustered by patient; eICU final intervals also require hospital clusters after hospital ID is restored.",
        ],
        "decision": {
            "can_run_contract_native_dynamic_xgboost": True,
            "can_run_72h_lookback": False,
            "can_predict_incident_aki_in_current_cohort": False,
            "can_run_scr_uo_distributional_model": False,
            "can_run_competing_risk_model_from_transition_contract": False,
            "next_step": "Correct eICU state extraction and retain source-available provenance fields in a new versioned contract before rerunning dynamic or fusion models.",
        },
    }
    (output_dir / "renal_state_feasibility_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_rows = []
    for database in databases:
        summary_rows.append({
            "database": database["database"],
            "n_rows": database["n_rows"],
            "n_episodes": database["n_episodes"],
            "n_patients": database["n_patients"],
            "median_windows": database["windows_per_episode"]["0.5"],
            "max_windows": database["windows_per_episode"]["1.0"],
            "history_24h_rate": database["episode_history_availability"]["at_least_24h"],
            "history_48h_rate": database["episode_history_availability"]["at_least_48h"],
            "history_72h_rate": database["episode_history_availability"]["at_least_72h"],
            "creatinine_missing_rate": database["observability"]["state_missingness"]["creatinine"],
            "map_missing_rate": database["observability"]["state_missingness"]["mean_arterial_pressure"],
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "renal_state_feasibility_summary.csv", index=False)
    print(json.dumps({"status": report["status"], "output": str(output_dir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT / "05_results_derived" / "02_renal_state_feasibility",
    )
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()