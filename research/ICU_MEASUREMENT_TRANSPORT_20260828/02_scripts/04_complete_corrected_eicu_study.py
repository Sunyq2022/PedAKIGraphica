"""Complete the corrected eICU clinical-state transport study.

One entry point performs raw-source governance, constructs immutable corrected
Gold/transition artifacts, evaluates source-only calibrated models in all six
external directions, computes patient/hospital clustered intervals and
ablations, and renders publication figures. Raw data are read-only. No
row-level predictions or patient identifiers are exported in result tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT.parent
TRUST_AKI_ROOT = Path(os.environ.get("TRUST_AKI_ROOT", str(PARENT / "outputs" / "trust_aki")))
RAW_EICU_DEFAULT = Path(os.environ["EICU_ROOT"])
TRANSITION_ROOT = Path(os.environ.get("TRANSITION_ROOT", str(TRUST_AKI_ROOT / "coarse_clinical_v1" / "adult_full_v002_contractfix")))
ORIGINAL_GOLD = Path(os.environ.get("EICU_ORIGINAL_GOLD", str(TRUST_AKI_ROOT / "adult_full_v001" / "eicu_full_kdigo_v001_full_transitions.gold.csv")))
LOCKED_SEMANTIC = PROJECT / "04_results_locked" / "01_semantic_audited_results" / "external_results.csv"
DATABASE_PATHS = {
    "MIMIC-III": TRANSITION_ROOT / "mimic3_coarse_clinical_v1_transitions.csv",
    "MIMIC-IV": TRANSITION_ROOT / "mimic4_coarse_clinical_v1_transitions.csv",
}
STATE_NAMES = [
    "heart_rate", "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "respiratory_rate", "spo2", "temperature", "creatinine", "lactate",
    "white_blood_cells", "platelets", "bun", "current_kdigo_stage",
    "predecision_vasopressor", "prior_aki_trend",
]
STATE = [f"s_{i}" for i in range(15)]
NEXT_STATE = [f"next_s_{i}" for i in range(15)]
ENDPOINTS = ["next_aki_progression", "next_aki_stage2_onset", "next_aki_stage3_onset", "hospital_death", "icu_death"]
MODELS = ["kdigo_lookup", "state_logistic", "state_xgboost"]
SEED = 20260828


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty formal directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def window_ids(stay: pd.Series, offset_minutes: pd.Series, index_hours: dict[int, float]) -> tuple[pd.Series, pd.Series]:
    hours = pd.to_numeric(offset_minutes, errors="coerce") / 60.0
    relative = hours - stay.map(index_hours)
    return relative, np.floor((relative - 1e-9) / 4.0)


def aggregate_aperiodic_bp(raw: Path, stays: set[int], index_hours: dict[int, float]) -> pd.DataFrame:
    parts = []
    columns = ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]
    for chunk in pd.read_csv(raw / "vitalAperiodic.csv.gz", usecols=columns, chunksize=500_000, low_memory=False):
        x = chunk[chunk["patientunitstayid"].isin(stays)].copy()
        if x.empty:
            continue
        relative, wid = window_ids(x["patientunitstayid"], x["observationoffset"], index_hours)
        x["relative_hours"], x["window_id"] = relative, wid
        x = x[x["relative_hours"].between(-4, 48) & x["window_id"].notna()].copy()
        x["window_id"] = x["window_id"].astype(int)
        for column, bounds in {"noninvasivesystolic": (30, 300), "noninvasivediastolic": (10, 200), "noninvasivemean": (20, 250)}.items():
            value = pd.to_numeric(x[column], errors="coerce")
            x[column] = value.where(value.between(*bounds))
        parts.append(x.groupby(["patientunitstayid", "window_id"], as_index=False)[columns[2:]].mean())
    if not parts:
        raise RuntimeError("no aperiodic BP records mapped to the formal cohort")
    return pd.concat(parts).groupby(["patientunitstayid", "window_id"], as_index=False)[columns[2:]].mean().rename(columns={"patientunitstayid": "icu_stay_id"})


def aggregate_nurse_temperature(raw: Path, stays: set[int], index_hours: dict[int, float]) -> pd.DataFrame:
    parts = []
    columns = ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"]
    for chunk in pd.read_csv(raw / "nurseCharting.csv.gz", usecols=columns, chunksize=500_000, low_memory=False):
        x = chunk[chunk["patientunitstayid"].isin(stays)].copy()
        label = x["nursingchartcelltypevallabel"].astype("string").str.strip()
        name = x["nursingchartcelltypevalname"].astype("string").str.strip()
        x = x[label.eq("Temperature") & name.isin(["Temperature (C)", "Temperature (F)"])].copy()
        if x.empty:
            continue
        value = pd.to_numeric(x["nursingchartvalue"], errors="coerce")
        is_f = x["nursingchartcelltypevalname"].astype("string").str.strip().eq("Temperature (F)")
        x["nurse_temperature_c"] = np.where(is_f, (value - 32) * 5 / 9, value)
        x["nurse_temperature_c"] = x["nurse_temperature_c"].where(x["nurse_temperature_c"].between(25, 45))
        relative, wid = window_ids(x["patientunitstayid"], x["nursingchartoffset"], index_hours)
        x["relative_hours"], x["window_id"] = relative, wid
        x = x[x["relative_hours"].between(-4, 48) & x["window_id"].notna()].copy()
        x["window_id"] = x["window_id"].astype(int)
        parts.append(x.groupby(["patientunitstayid", "window_id"], as_index=False)["nurse_temperature_c"].mean())
    if not parts:
        raise RuntimeError("no nurse-charted temperature mapped to the formal cohort")
    return pd.concat(parts).groupby(["patientunitstayid", "window_id"], as_index=False)["nurse_temperature_c"].mean().rename(columns={"patientunitstayid": "icu_stay_id"})


def aggregate_io(raw: Path, stays: set[int], index_hours: dict[int, float]) -> pd.DataFrame:
    parts = []
    columns = ["patientunitstayid", "intakeoutputoffset", "intaketotal", "outputtotal", "nettotal", "cellpath", "celllabel", "cellvaluenumeric"]
    for chunk in pd.read_csv(raw / "intakeOutput.csv.gz", usecols=columns, chunksize=500_000, low_memory=False):
        x = chunk[chunk["patientunitstayid"].isin(stays)].copy()
        if x.empty:
            continue
        relative, wid = window_ids(x["patientunitstayid"], x["intakeoutputoffset"], index_hours)
        x["relative_hours"], x["window_id"] = relative, wid
        x = x[x["relative_hours"].between(-4, 48) & x["window_id"].notna()].copy()
        if x.empty:
            continue
        x["window_id"] = x["window_id"].astype(int)
        label = x["celllabel"].astype("string")
        path = x["cellpath"].astype("string")
        numeric = pd.to_numeric(x["cellvaluenumeric"], errors="coerce")
        urine = label.str.contains("urine", case=False, regex=False, na=False) & ~label.str.contains("count|occurrence", case=False, regex=True, na=False) & path.str.contains(r"I&O\|Output \(ml\)\|", regex=True, na=False)
        x["urine_ml"] = numeric.where(urine & numeric.between(0, 10_000))
        x["intake_total_recorded"] = pd.to_numeric(x["intaketotal"], errors="coerce")
        x["output_total_recorded"] = pd.to_numeric(x["outputtotal"], errors="coerce")
        x["net_total_recorded"] = pd.to_numeric(x["nettotal"], errors="coerce")
        x = x.sort_values("intakeoutputoffset")
        parts.append(x.groupby(["patientunitstayid", "window_id"], as_index=False).agg(
            urine_ml_4h=("urine_ml", "sum"), urine_rows=("urine_ml", "count"),
            intake_total_recorded=("intake_total_recorded", "last"),
            output_total_recorded=("output_total_recorded", "last"),
            net_total_recorded=("net_total_recorded", "last")))
    if not parts:
        raise RuntimeError("no I/O records mapped to the formal cohort")
    return pd.concat(parts).sort_values(["patientunitstayid", "window_id"]).groupby(["patientunitstayid", "window_id"], as_index=False).last().rename(columns={"patientunitstayid": "icu_stay_id"})


def custom_lab_audit(raw: Path) -> pd.DataFrame:
    x = pd.read_csv(raw / "customLab.csv.gz", low_memory=False)
    name = x["labothername"].astype("string").str.strip().str.lower()
    creatinine_like = name.str.contains("creat|crt", regex=True, na=False) & ~name.str.contains("ratio|clearance", regex=True, na=False)
    return pd.DataFrame({
        "audit_item": ["all_custom_lab_rows", "creatinine_like_rows", "creatinine_like_unique_labels", "incorporated_into_state"],
        "value": [len(x), int(creatinine_like.sum()), int(name[creatinine_like].nunique()), 0],
        "interpretation": [
            "raw customLab rows", "labels require hospital-specific unit validation", "normalized candidate labels",
            "not incorporated because result/unit semantics are not standardized; recorded as a governance limitation",
        ],
    })


def patient_governance(raw: Path, stays: set[int]) -> pd.DataFrame:
    columns = ["patientunitstayid", "hospitalid", "wardid", "unitdischargeoffset", "unitdischargestatus", "hospitaldischargeoffset", "hospitaldischargestatus", "hospitaldischargelocation", "unitdischargelocation"]
    x = pd.read_csv(raw / "patient.csv.gz", usecols=columns, low_memory=False)
    x = x[x["patientunitstayid"].isin(stays)].copy()
    if x["patientunitstayid"].duplicated().any():
        raise RuntimeError("patient table is not one row per ICU stay")
    return x.rename(columns={"patientunitstayid": "icu_stay_id", "hospitalid": "hospital_id", "wardid": "ward_id"})


def build_corrected_contract(raw: Path, data_dir: Path) -> tuple[Path, Path, dict]:
    current_path = TRANSITION_ROOT / "eicu_coarse_clinical_v1_transitions.csv"
    transition = pd.read_csv(current_path, low_memory=False)
    transition["icu_stay_id"] = pd.to_numeric(transition["icu_stay_id"], errors="raise").astype(int)
    transition["window_id"] = np.rint((pd.to_numeric(transition["state_window_start"]) - pd.to_numeric(transition["index_time"])) / 4).astype(int)
    if transition.duplicated(["icu_stay_id", "window_id"]).any():
        raise RuntimeError("transition stay/window key is not unique")
    index_hours = transition.drop_duplicates("icu_stay_id").set_index("icu_stay_id")["index_time"].astype(float).to_dict()
    stays = set(index_hours)
    bp = aggregate_aperiodic_bp(raw, stays, index_hours)
    temp = aggregate_nurse_temperature(raw, stays, index_hours)
    io = aggregate_io(raw, stays, index_hours)
    patient = patient_governance(raw, stays)
    custom = custom_lab_audit(raw)

    gold = pd.read_csv(ORIGINAL_GOLD, low_memory=False)
    gold["icu_stay_id"] = pd.to_numeric(gold["icu_stay_id"], errors="raise").astype(int)
    gold = gold[gold["icu_stay_id"].isin(stays)].copy()
    gold["window_id"] = pd.to_numeric(gold["window_id"], errors="raise").astype(int)
    gold = gold.merge(bp, on=["icu_stay_id", "window_id"], how="left", validate="one_to_one")
    gold = gold.merge(temp, on=["icu_stay_id", "window_id"], how="left", validate="one_to_one")
    gold = gold.merge(io, on=["icu_stay_id", "window_id"], how="left", validate="one_to_one")
    gold = gold.merge(patient, on="icu_stay_id", how="left", validate="many_to_one")
    before = {c: float(gold[c].isna().mean()) for c in ["sbp", "dbp", "map", "temp", "creatinine"]}
    gold["sbp_source"] = np.select([gold["sbp"].notna(), gold["noninvasivesystolic"].notna()], ["periodic_systemic", "aperiodic_noninvasive"], default="missing")
    gold["dbp_source"] = np.select([gold["dbp"].notna(), gold["noninvasivediastolic"].notna()], ["periodic_systemic", "aperiodic_noninvasive"], default="missing")
    gold["map_source"] = np.select([gold["map"].notna(), gold["noninvasivemean"].notna()], ["periodic_systemic", "aperiodic_noninvasive"], default="missing")
    gold["temperature_source"] = np.select([gold["temp"].notna(), gold["nurse_temperature_c"].notna()], ["vital_periodic", "nurse_charting"], default="missing")
    gold["sbp"] = gold["sbp"].combine_first(gold["noninvasivesystolic"])
    gold["dbp"] = gold["dbp"].combine_first(gold["noninvasivediastolic"])
    gold["map"] = gold["map"].combine_first(gold["noninvasivemean"])
    gold["temp"] = gold["temp"].combine_first(gold["nurse_temperature_c"])
    gold["kdigo_stage_index_anchor"] = gold["window_id"].eq(-1).astype(int)
    gold["kdigo_stage_component_observed"] = gold["kdigo_components_observed"].notna().astype(int)
    gold["kdigo_stage_direct_observed"] = (gold["kdigo_stage_index_anchor"].eq(1) | gold["kdigo_stage_component_observed"].eq(1)).astype(int)
    gold["kdigo_stage_carried_forward"] = (gold["kdigo_stage"].notna() & gold["kdigo_stage_direct_observed"].eq(0)).astype(int)
    gold["icu_discharge_time"] = pd.to_numeric(gold["unitdischargeoffset"], errors="coerce") / 60
    gold["hospital_discharge_time"] = pd.to_numeric(gold["hospitaldischargeoffset"], errors="coerce") / 60
    gold["death_event_time"] = np.where(gold["hospitaldischargestatus"].astype(str).eq("Expired"), gold["hospital_discharge_time"], np.nan)
    gold["death_event_time_source"] = np.where(gold["hospitaldischargestatus"].astype(str).eq("Expired"), "patient.hospitaldischargeoffset_when_status_expired", "not_applicable")
    after = {c: float(gold[c].isna().mean()) for c in ["sbp", "dbp", "map", "temp", "creatinine"]}

    window_columns = ["icu_stay_id", "window_id", "sbp", "dbp", "map", "temp", "sbp_source", "dbp_source", "map_source", "temperature_source", "urine_ml_4h", "urine_rows", "intake_total_recorded", "output_total_recorded", "net_total_recorded", "kdigo_stage_direct_observed", "kdigo_stage_carried_forward"]
    w = gold[window_columns].copy()
    current = w.add_prefix("current_").rename(columns={"current_icu_stay_id": "icu_stay_id", "current_window_id": "window_id"})
    future = w.copy(); future["window_id"] -= 1
    future = future.add_prefix("future_").rename(columns={"future_icu_stay_id": "icu_stay_id", "future_window_id": "window_id"})
    corrected = transition.merge(current, on=["icu_stay_id", "window_id"], how="left", validate="one_to_one")
    corrected = corrected.merge(future, on=["icu_stay_id", "window_id"], how="left", validate="one_to_one")
    corrected = corrected.merge(patient, on="icu_stay_id", how="left", validate="many_to_one")
    for state_col, feature in [("s_1", "sbp"), ("s_2", "dbp"), ("s_3", "map"), ("s_6", "temp")]:
        corrected[state_col] = corrected[f"current_{feature}"].combine_first(corrected[state_col])
        corrected["next_" + state_col] = corrected[f"future_{feature}"].combine_first(corrected["next_" + state_col])
    corrected["kdigo_stage_direct_observed"] = corrected["current_kdigo_stage_direct_observed"].fillna(0).astype(int)
    corrected["kdigo_stage_carried_forward"] = corrected["current_kdigo_stage_carried_forward"].fillna(0).astype(int)
    corrected["next_kdigo_stage_direct_observed"] = corrected["future_kdigo_stage_direct_observed"].fillna(0).astype(int)
    corrected["next_kdigo_stage_carried_forward"] = corrected["future_kdigo_stage_carried_forward"].fillna(0).astype(int)
    corrected["database_version"] = "eicu-crd-2.0-corrected-state-v1"

    if gold.duplicated(["icu_stay_id", "window_id"]).any():
        raise RuntimeError("corrected Gold stay/window key is not unique")
    if patient["hospital_id"].isna().any() or patient["hospital_id"].nunique() != 138:
        raise RuntimeError("formal eICU cohort must map completely to 138 source hospitals")
    if not (after["map"] < before["map"] - .50 and after["temp"] < before["temp"] - .50):
        raise RuntimeError("corrected source extraction did not materially rescue MAP and temperature")
    if corrected.duplicated(["icu_stay_id", "window_id"]).any() or len(corrected) != len(transition):
        raise RuntimeError("corrected transition key or row count changed")

    gold_path = data_dir / "eicu_corrected_gold.csv.gz"
    transition_path = data_dir / "eicu_corrected_transitions.csv.gz"
    gold.to_csv(gold_path, index=False, compression="gzip")
    corrected.to_csv(transition_path, index=False, compression="gzip")
    manifest = {
        "status": "CORRECTED_EICU_CONTRACT_COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_locked_to_existing_transition": True, "n_stays": len(stays), "n_transition_rows": len(corrected), "n_gold_rows": len(gold),
        "window_semantics": "state=(start,end], decision=end; no post-decision source used in current state",
        "source_priority": {"blood_pressure": "periodic systemic then aperiodic non-invasive", "temperature": "vitalPeriodic then nurseCharting C/F"},
        "missingness_before": before, "missingness_after": after,
        "custom_lab_audit": custom.to_dict(orient="records"),
        "raw_input_sha256": {name: sha256(raw / name) for name in ["patient.csv.gz", "vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz", "nurseCharting.csv.gz", "lab.csv.gz", "customLab.csv.gz", "intakeOutput.csv.gz"]},
        "source_transition_sha256": sha256(current_path), "source_gold_sha256": sha256(ORIGINAL_GOLD),
        "outputs": {gold_path.name: sha256(gold_path), transition_path.name: sha256(transition_path)},
        "limitations": ["customLab creatinine-like values not incorporated without unit validation", "recorded I/O totals retained with provenance but not treated as item-level sums", "death event time is discharge offset conditional on Expired status, not an independent timestamp"],
    }
    (data_dir / "data_governance_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return gold_path, transition_path, manifest


def load_first(path: Path, corrected: bool = False) -> pd.DataFrame:
    extra = ["hospital_id", "kdigo_stage_direct_observed", "next_kdigo_stage_direct_observed"] if corrected else []
    columns = ["subject_id", "episode_id", "decision_time", "action_valid", *STATE, "next_s_12", "hospital_death", "icu_death", *extra]
    x = pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)
    x = x[pd.to_numeric(x["action_valid"], errors="coerce").eq(1)].sort_values(["episode_id", "decision_time"], kind="mergesort").drop_duplicates("episode_id").reset_index(drop=True)
    current = pd.to_numeric(x["s_12"], errors="coerce"); future = pd.to_numeric(x["next_s_12"], errors="coerce")
    valid = current.between(0, 3) & future.between(0, 3)
    x["next_aki_progression"] = np.where(valid, future.gt(current).astype(float), np.nan)
    x["next_aki_stage2_onset"] = np.where(valid & current.lt(2), future.ge(2).astype(float), np.nan)
    x["next_aki_stage3_onset"] = np.where(valid & current.lt(3), future.ge(3).astype(float), np.nan)
    for c in [*STATE, *ENDPOINTS]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x["subject_id"] = x["subject_id"].astype(str)
    if "hospital_id" not in x: x["hospital_id"] = np.nan
    return x


class Transform:
    def fit(self, frame: pd.DataFrame) -> "Transform":
        v = frame[STATE].apply(pd.to_numeric, errors="coerce")
        self.lo = v.quantile(.01).to_numpy(float); self.hi = v.quantile(.99).to_numpy(float); self.med = v.median().to_numpy(float)
        if not np.isfinite(self.med).all(): raise RuntimeError("source feature has no observations")
        z = np.where(np.isfinite(np.clip(v.to_numpy(float), self.lo, self.hi)), np.clip(v.to_numpy(float), self.lo, self.hi), self.med)
        self.mean = z.mean(0); self.std = z.std(0); self.std = np.where(self.std > 1e-6, self.std, 1)
        return self
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        v = frame[STATE].to_numpy(float); observed = np.isfinite(v).astype(np.float32); v = np.clip(v, self.lo, self.hi); v = np.where(np.isfinite(v), v, self.med)
        return np.column_stack([((v - self.mean) / self.std).astype(np.float32), observed])


def patient_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    patients = frame["subject_id"].drop_duplicates().to_numpy(); rng = np.random.default_rng(SEED); rng.shuffle(patients); cut = max(1, int(.8 * len(patients))); dev = set(patients[:cut]); mask = frame["subject_id"].isin(dev).to_numpy(); return mask, ~mask


def platt_fit(y: np.ndarray, score: np.ndarray) -> LogisticRegression | None:
    if np.unique(y).size < 2: return None
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(score).reshape(-1, 1), y)
    return m


def platt_apply(model: LogisticRegression | None, score: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1] if model is not None else np.asarray(score)


def threshold_youden(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.quantile(p, np.linspace(.01, .99, 99))); best = (.5, -np.inf)
    for threshold in candidates:
        pred = p >= threshold; sens = pred[y == 1].mean() if np.any(y == 1) else 0; spec = (~pred[y == 0]).mean() if np.any(y == 0) else 0
        if sens + spec - 1 > best[1]: best = (float(threshold), float(sens + spec - 1))
    return best[0]


def fit_direction(source: pd.DataFrame, target: pd.DataFrame, feature_indices: np.ndarray | None = None, endpoints: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    endpoints = endpoints or ENDPOINTS; dev_mask, val_mask = patient_split(source); transform = Transform().fit(source.loc[dev_mask]); xd = transform.transform(source.loc[dev_mask]); xv = transform.transform(source.loc[val_mask]); xt = transform.transform(target)
    if feature_indices is not None: xd, xv, xt = xd[:, feature_indices], xv[:, feature_indices], xt[:, feature_indices]
    pred = {m: np.full((len(target), len(endpoints)), np.nan) for m in MODELS}; thresholds = {}; source_stage = source.loc[dev_mask, "s_12"].round().clip(0, 3); val_stage = source.loc[val_mask, "s_12"].round().clip(0, 3); target_stage = target["s_12"].round().clip(0, 3)
    for j, endpoint in enumerate(endpoints):
        yd = source.loc[dev_mask, endpoint].to_numpy(float); yv = source.loc[val_mask, endpoint].to_numpy(float); vd = np.isfinite(yd); vv = np.isfinite(yv)
        if np.unique(yd[vd]).size < 2 or np.unique(yv[vv]).size < 2: continue
        prevalence = yd[vd].mean(); rates = source.loc[dev_mask].loc[vd].assign(_y=yd[vd]).groupby(source_stage[vd])["_y"].agg(["sum", "count"]); lookup = ((rates["sum"] + 20 * prevalence) / (rates["count"] + 20)).to_dict(); val_raw = val_stage.map(lookup).fillna(prevalence).to_numpy(float); test_raw = target_stage.map(lookup).fillna(prevalence).to_numpy(float); cal = platt_fit(yv[vv].astype(int), val_raw[vv]); val_p = platt_apply(cal, val_raw); pred["kdigo_lookup"][:, j] = platt_apply(cal, test_raw); thresholds[("kdigo_lookup", endpoint)] = threshold_youden(yv[vv].astype(int), val_p[vv])
        lr = LogisticRegression(C=.2, max_iter=1000, class_weight="balanced", random_state=SEED).fit(xd[vd], yd[vd].astype(int)); val_raw = lr.predict_proba(xv)[:, 1]; test_raw = lr.predict_proba(xt)[:, 1]; cal = platt_fit(yv[vv].astype(int), val_raw[vv]); val_p = platt_apply(cal, val_raw); pred["state_logistic"][:, j] = platt_apply(cal, test_raw); thresholds[("state_logistic", endpoint)] = threshold_youden(yv[vv].astype(int), val_p[vv])
        pos = yd[vd].sum(); neg = vd.sum() - pos; xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.03, subsample=.8, colsample_bytree=.8, min_child_weight=10, reg_lambda=2, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=4, random_state=SEED, scale_pos_weight=max(1, float(neg / max(pos, 1)))).fit(xd[vd], yd[vd].astype(int)); val_raw = xgb.predict_proba(xv)[:, 1]; test_raw = xgb.predict_proba(xt)[:, 1]; cal = platt_fit(yv[vv].astype(int), val_raw[vv]); val_p = platt_apply(cal, val_raw); pred["state_xgboost"][:, j] = platt_apply(cal, test_raw); thresholds[("state_xgboost", endpoint)] = threshold_youden(yv[vv].astype(int), val_p[vv])
    return pred, thresholds


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6; logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)); m = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1,1), y); return float(m.intercept_[0]), float(m.coef_[0,0])


def evaluate(y: np.ndarray, predictions: dict[str, np.ndarray], thresholds: dict, source: str, target: str) -> tuple[list[dict], list[dict]]:
    rows=[]; bins=[]
    for model, matrix in predictions.items():
        for j, endpoint in enumerate(ENDPOINTS):
            valid=np.isfinite(y[:,j])&np.isfinite(matrix[:,j]); yy=y[valid,j].astype(int); pp=np.clip(matrix[valid,j],1e-6,1-1e-6)
            if np.unique(yy).size<2: continue
            intercept,slope=calibration_stats(yy,pp); threshold=thresholds[(model,endpoint)]; binary=pp>=threshold; tp=np.sum(binary&(yy==1)); tn=np.sum(~binary&(yy==0)); fp=np.sum(binary&(yy==0)); fn=np.sum(~binary&(yy==1)); ece=0
            q=pd.qcut(pp,q=min(10,len(np.unique(pp))),duplicates="drop"); grouped=pd.DataFrame({"y":yy,"p":pp,"bin":q}).groupby("bin",observed=True)
            for k,g in grouped:
                ece += len(g)/len(yy)*abs(g.y.mean()-g.p.mean()); bins.append({"source_database":source,"target_database":target,"model":model,"endpoint":endpoint,"bin_mean_prediction":g.p.mean(),"bin_event_rate":g.y.mean(),"n":len(g)})
            rows.append({"source_database":source,"target_database":target,"model":model,"endpoint":endpoint,"n":len(yy),"event_rate":yy.mean(),"auroc":roc_auc_score(yy,pp),"average_precision":average_precision_score(yy,pp),"brier":brier_score_loss(yy,pp),"ece_10bin":ece,"calibration_intercept":intercept,"calibration_slope":slope,"source_selected_threshold":threshold,"sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"ppv":tp/max(tp+fp,1),"npv":tn/max(tn+fn,1)})
    return rows,bins


def cluster_bootstrap(y: np.ndarray, pred: dict[str,np.ndarray], clusters: np.ndarray, source: str, target: str, reps: int, unit: str) -> list[dict]:
    groups,codes=np.unique(pd.Series(clusters).astype(str),return_inverse=True); rng=np.random.default_rng(SEED+(0 if unit=="patient" else 17)); draws=np.zeros((reps,len(groups)),dtype=np.int16)
    for r in range(reps): draws[r]=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
    rows=[]
    for model,matrix in pred.items():
        for j,endpoint in enumerate(ENDPOINTS):
            valid=np.isfinite(y[:,j])&np.isfinite(matrix[:,j]); yy=y[valid,j].astype(int); pp=matrix[valid,j]; cc=codes[valid]; values=[]
            for r in range(reps):
                w=draws[r,cc]; present=w>0
                if np.unique(yy[present]).size==2: values.append(roc_auc_score(yy,pp,sample_weight=w))
            rows.append({"source_database":source,"target_database":target,"model":model,"endpoint":endpoint,"cluster_unit":unit,"n_clusters":len(groups),"bootstrap_reps":len(values),"auroc_lo":np.quantile(values,.025),"auroc_hi":np.quantile(values,.975)})
    return rows


def profile(frames: dict[str,pd.DataFrame], original_eicu: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for database,frame in {**frames,"eICU-original":original_eicu}.items():
        for c,name in zip(STATE,STATE_NAMES): rows.append({"database":database,"state_column":c,"state_name":name,"n":len(frame),"observed_n":frame[c].notna().sum(),"missing_rate":frame[c].isna().mean(),"median":frame[c].median(),"q25":frame[c].quantile(.25),"q75":frame[c].quantile(.75)})
    return pd.DataFrame(rows)


def ablation_indices(name: str) -> np.ndarray:
    all_idx=np.arange(30); remove=[]
    if name=="no_kdigo": remove=[12,27]
    elif name=="no_creatinine": remove=[7,22]
    elif name=="no_prior_trend": remove=[14,29]
    elif name=="no_missingness": return np.arange(15)
    elif name=="no_labs": remove=list(range(7,12))+list(range(22,27))
    return np.array([i for i in all_idx if i not in remove])


def make_figures(results: pd.DataFrame, patient_ci: pd.DataFrame, hospital_ci: pd.DataFrame, profiles: pd.DataFrame, calibration: pd.DataFrame, ablations: pd.DataFrame, figures: Path) -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":7.5,"xtick.labelsize":8,"ytick.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    db_colors={"MIMIC-III":"#3B6FB6","MIMIC-IV":"#E07B39","eICU":"#2A9D8F","eICU-original":"#9AA0A6"}
    feature_labels={"systolic_bp":"Systolic BP","diastolic_bp":"Diastolic BP","mean_arterial_pressure":"Mean arterial pressure","temperature":"Temperature","creatinine":"Creatinine"}
    focus=profiles[profiles.state_name.isin(feature_labels)].copy();focus["feature"]=focus.state_name.map(feature_labels)
    order=list(feature_labels.values())
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.2,3.35),gridspec_kw={"width_ratios":[.9,1.35]})
    paired=focus[focus.database.isin(["eICU-original","eICU"])].copy();paired["ETL"]=paired.database.map({"eICU-original":"Original","eICU":"Corrected"})
    sns.barplot(paired,x="feature",y="missing_rate",hue="ETL",order=order,palette={"Original":db_colors["eICU-original"],"Corrected":db_colors["eICU"]},ax=ax1)
    ax1.set_ylim(0,1);ax1.set_ylabel("Missing four-hour windows");ax1.set_xlabel("");ax1.yaxis.set_major_formatter(lambda x,pos:f"{x:.0%}");ax1.tick_params(axis="x",rotation=28);ax1.legend(title=None,frameon=False,ncol=2,loc="upper left");ax1.set_title("A  eICU extraction correction",loc="left",fontweight="bold")
    corrected=focus[~focus.database.eq("eICU-original")];sns.barplot(corrected,x="feature",y="missing_rate",hue="database",order=order,palette=db_colors,ax=ax2);ax2.set_ylabel("Missing four-hour windows");ax2.set_xlabel("");ax2.yaxis.set_major_formatter(lambda x,pos:f"{x:.0%}");ax2.tick_params(axis="x",rotation=28);ax2.legend(title=None,frameon=False,ncol=3,loc="upper left");ax2.set_title("B  Corrected cross-database observability",loc="left",fontweight="bold")
    fig.suptitle("Source completion separates extraction loss from measurement sparsity",fontweight="bold",x=.02,ha="left",y=1.01);fig.tight_layout();fig.savefig(figures/"Figure6_data_governance.pdf",bbox_inches="tight");fig.savefig(figures/"Figure6_data_governance.png",dpi=600,bbox_inches="tight");plt.close(fig)
    q=results[results.endpoint.eq("next_aki_progression")].merge(patient_ci[patient_ci.endpoint.eq("next_aki_progression")],on=["source_database","target_database","model","endpoint"]);abbr={"MIMIC-III":"M-III","MIMIC-IV":"M-IV","eICU":"eICU"};q["direction"]=q.source_database.map(abbr)+" → "+q.target_database.map(abbr);order=["M-III → M-IV","M-III → eICU","M-IV → M-III","M-IV → eICU","eICU → M-III","eICU → M-IV"];model_specs=[("kdigo_lookup","o","#667085","KDIGO lookup"),("state_logistic","s","#3B6FB6","State logistic"),("state_xgboost","D","#D64541","State XGBoost")];fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,3.6),gridspec_kw={"width_ratios":[1.35,1]})
    for mi,(model,marker,color,label) in enumerate(model_specs):
        z=q[q.model.eq(model)].set_index("direction").reindex(order);y=np.arange(len(order))+(mi-1)*.16;ax1.errorbar(z.auroc,y,xerr=[z.auroc-z.auroc_lo,z.auroc_hi-z.auroc],fmt=marker,color=color,label=label,capsize=2,markersize=4,lw=1);ax2.scatter(z.event_rate,z.average_precision,marker=marker,color=color,label=label,s=25,alpha=.9)
    ax1.axvline(.5,color="#555",ls="--",lw=.8);ax1.set_yticks(np.arange(len(order)),order);ax1.invert_yaxis();ax1.set_xlim(.45,.78);ax1.set_xlabel("AUROC (patient-clustered 95% CI)");ax1.set_title("A  Six zero-shot transfer directions",loc="left",fontweight="bold");ax1.legend(frameon=False,loc="lower right")
    ax2.plot([0,.2],[0,.2],ls="--",color="#555",lw=.8);ax2.set_xlim(0,.19);ax2.set_ylim(0,.27);ax2.set_xlabel("Target prevalence");ax2.set_ylabel("Average precision");ax2.set_title("B  Precision relative to prevalence",loc="left",fontweight="bold")
    fig.suptitle("Corrected clinical-state prediction of next-window AKI progression",fontweight="bold",x=.02,ha="left",y=1.01);fig.tight_layout();fig.savefig(figures/"Figure7_corrected_transport.pdf",bbox_inches="tight");fig.savefig(figures/"Figure7_corrected_transport.png",dpi=600,bbox_inches="tight");plt.close(fig)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,3.45),gridspec_kw={"width_ratios":[1.15,1]});c=calibration[(calibration.target_database.eq("eICU"))&(calibration.endpoint.eq("next_aki_progression"))].copy();c["Source"]=c.source_database.map({"MIMIC-III":"M-III","MIMIC-IV":"M-IV"});c["Model"]=c.model.map({"kdigo_lookup":"KDIGO lookup","state_logistic":"State logistic","state_xgboost":"State XGBoost"});sns.lineplot(c,x="bin_mean_prediction",y="bin_event_rate",hue="Model",style="Source",markers=True,dashes=True,palette=["#667085","#3B6FB6","#D64541"],ax=ax1);ax1.plot([0,.35],[0,.35],ls="--",color="#333",lw=.8);ax1.set_xlim(0,.35);ax1.set_ylim(0,.35);ax1.set_xlabel("Mean predicted risk");ax1.set_ylabel("Observed event rate");ax1.set_title("A  eICU reliability",loc="left",fontweight="bold");ax1.legend(frameon=False,ncol=2,fontsize=6.5,loc="upper left")
    pivot=ablations.assign(source_database=ablations.source_database.map({"MIMIC-III":"M-III","MIMIC-IV":"M-IV"}),ablation=ablations.ablation.map({"no_kdigo":"KDIGO","no_creatinine":"Creatinine","no_prior_trend":"Prior trend","no_missingness":"Missing flags","no_labs":"All labs"})).pivot(index="source_database",columns="ablation",values="delta_auroc_vs_full").reindex(index=["M-III","M-IV"],columns=["KDIGO","All labs","Creatinine","Prior trend","Missing flags"]);sns.heatmap(pivot,center=0,vmin=-.15,vmax=.15,cmap="vlag",annot=True,fmt=".3f",linewidths=.5,ax=ax2,cbar_kws={"label":"Δ AUROC","shrink":.75});ax2.set_title("B  Progression ablation",loc="left",fontweight="bold");ax2.set_xlabel("Feature group removed");ax2.set_ylabel("Source → eICU");ax2.tick_params(axis="x",rotation=25)
    fig.suptitle("Calibration and representation dependence",fontweight="bold",x=.02,ha="left",y=1.01);fig.tight_layout();fig.savefig(figures/"Figure8_calibration_ablation.pdf",bbox_inches="tight");fig.savefig(figures/"Figure8_calibration_ablation.png",dpi=600,bbox_inches="tight");plt.close(fig)


def run(args: argparse.Namespace) -> None:
    for directory in [args.data_dir,args.results_dir,args.figures_dir]: ensure_new_directory(directory)
    _, corrected_transition, governance = build_corrected_contract(args.raw_eicu,args.data_dir)
    frames={name:load_first(path) for name,path in DATABASE_PATHS.items()};original_eicu=load_first(TRANSITION_ROOT/"eicu_coarse_clinical_v1_transitions.csv");frames["eICU"]=load_first(corrected_transition,corrected=True)
    metrics=[];bins=[];patient_ci=[];hospital_ci=[];cache={}
    for source in frames:
        for target in frames:
            if source==target:continue
            prediction,threshold=fit_direction(frames[source],frames[target]);cache[(source,target)]=(prediction,threshold);y=frames[target][ENDPOINTS].to_numpy(float);m,c=evaluate(y,prediction,threshold,source,target);metrics+=m;bins+=c;patient_ci+=cluster_bootstrap(y,prediction,frames[target].subject_id.to_numpy(),source,target,args.bootstrap_reps,"patient")
            if target=="eICU": hospital_ci+=cluster_bootstrap(y,prediction,frames[target].hospital_id.to_numpy(),source,target,args.bootstrap_reps,"hospital")
            print(f"completed {source} -> {target}",flush=True)
    result=pd.DataFrame(metrics);patient=pd.DataFrame(patient_ci);hospital=pd.DataFrame(hospital_ci);cal=pd.DataFrame(bins);prof=profile(frames,original_eicu)
    key=["source_database","target_database","model","endpoint"]
    if result.shape[0] != 90 or patient.shape[0] != 90 or result.duplicated(key).any() or patient.duplicated(key).any():
        raise RuntimeError("six-direction result or patient-bootstrap grid is incomplete")
    if hospital.shape[0] != 30 or set(hospital["n_clusters"]) != {138} or hospital.duplicated(key).any():
        raise RuntimeError("eICU hospital-cluster bootstrap grid is incomplete")
    ab=[]
    for source in ["MIMIC-III","MIMIC-IV"]:
        full=result[(result.source_database.eq(source))&(result.target_database.eq("eICU"))&(result.model.eq("state_xgboost"))&(result.endpoint.eq("next_aki_progression"))].auroc.iloc[0]
        for name in ["no_kdigo","no_creatinine","no_prior_trend","no_missingness","no_labs"]:
            prediction,threshold=fit_direction(frames[source],frames["eICU"],ablation_indices(name),["next_aki_progression"]);y=frames["eICU"][["next_aki_progression"]].to_numpy(float);valid=np.isfinite(y[:,0])&np.isfinite(prediction["state_xgboost"][:,0]);auc=roc_auc_score(y[valid,0],prediction["state_xgboost"][valid,0]);ab.append({"source_database":source,"target_database":"eICU","ablation":name,"auroc":auc,"full_auroc":full,"delta_auroc_vs_full":auc-full})
    ablations=pd.DataFrame(ab)
    semantic=pd.read_csv(LOCKED_SEMANTIC);semantic=semantic[semantic.model.eq("semantic_transformer")][["source_database","target_database","endpoint","auroc","average_precision"]].rename(columns={"auroc":"semantic_auroc","average_precision":"semantic_average_precision"});comparison=result.merge(semantic,on=["source_database","target_database","endpoint"],how="left");comparison["delta_auroc_vs_semantic"]=comparison.auroc-comparison.semantic_auroc
    outputs={"corrected_external_results.csv":result,"corrected_patient_bootstrap.csv":patient,"corrected_hospital_bootstrap.csv":hospital,"corrected_calibration_bins.csv":cal,"corrected_ablation_results.csv":ablations,"corrected_state_profile.csv":prof,"corrected_semantic_comparison.csv":comparison}
    for name,frame in outputs.items():frame.to_csv(args.results_dir/name,index=False)
    make_figures(result,patient,hospital,prof,cal,ablations,args.figures_dir)
    manifest={"status":"CORRECTED_EICU_STUDY_COMPLETE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"python":platform.python_version(),"analysis_unit":"first eligible transition per ICU episode","prediction_horizon_hours":4,"endpoints":ENDPOINTS,"risk_sets":{"stage2_onset":"current stage <2","stage3_onset":"current stage <3"},"source_internal_validation":"80/20 patient split; Platt calibration and threshold selection on source validation only","models":MODELS,"bootstrap_reps":args.bootstrap_reps,"directions":6,"data_governance_manifest":governance,"input_sha256":{**{k:sha256(v) for k,v in DATABASE_PATHS.items()},"eICU-corrected":sha256(corrected_transition)},"script_sha256":sha256(Path(__file__)),"result_sha256":{name:sha256(args.results_dir/name) for name in outputs},"figure_files":sorted(p.name for p in args.figures_dir.iterdir()),"interpretation_boundary":["eICU correction retains the historical cohort to isolate source-extraction effects","target labels are never used for model fitting, calibration, threshold selection, or ablation choice","hospital-cluster intervals use source-recorded eICU hospital IDs","no causal, treatment-effect, or deployment claim"]};(args.results_dir/"study_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps({"status":manifest["status"],"results":str(args.results_dir),"figures":str(args.figures_dir)},indent=2))


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--raw-eicu",type=Path,default=RAW_EICU_DEFAULT);p.add_argument("--data-dir",type=Path,default=PROJECT/"03_data/02_eicu_corrected_contract_v1");p.add_argument("--results-dir",type=Path,default=PROJECT/"05_results_derived/03_corrected_eicu_study");p.add_argument("--figures-dir",type=Path,default=PROJECT/"06_figures_locked/02_corrected_eicu_study_figures");p.add_argument("--bootstrap-reps",type=int,default=1000);run(p.parse_args())


if __name__=="__main__":main()
