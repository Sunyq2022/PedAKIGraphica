"""Focused semantic-timing and label-proximity sensitivity analyses.

Each invocation evaluates one concept-filtering mode for the principal
database-balanced MIMIC-III + MIMIC-IV to eICU LODO estimand. Outputs are
versioned and never overwrite the locked all-concept analysis. The
label-proximal exclusion is explicitly post hoc.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT / "02_scripts" / "01_run_grasp_icu_study.py"
LOCKED = PROJECT / "04_results_locked" / "01_semantic_audited_results"
MODES = ("precise_time", "medication_only", "label_proximal_excluded")
MODE_CLASS = {
    "precise_time": "prespecified timing sensitivity",
    "medication_only": "prespecified conservative timing sensitivity",
    "label_proximal_excluded": "post-hoc label-proximity sensitivity",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("semantic_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import core analysis from {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_locked_embeddings() -> dict[str, np.ndarray]:
    path = LOCKED / "semantic_embedding_lookup.npz"
    with np.load(path, allow_pickle=False) as payload:
        keys = payload["keys"].tolist()
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    return dict(zip(keys, vectors))


def point_metrics(core, y: np.ndarray, predictions: dict[str, np.ndarray], mode: str) -> list[dict]:
    rows: list[dict] = []
    for model, pred in predictions.items():
        for j, endpoint in enumerate(core.ENDPOINTS):
            valid = np.isfinite(y[:, j])
            yy = y[valid, j].astype(int)
            pp = pred[valid, j]
            rows.append({
                "mode": mode,
                "mode_class": MODE_CLASS[mode],
                "source_database": "MIMIC-III + MIMIC-IV",
                "target_database": "eICU",
                "model": model,
                "endpoint": endpoint,
                "n": len(yy),
                "event_rate": float(yy.mean()),
                "auroc": float(roc_auc_score(yy, pp)),
                "average_precision": float(average_precision_score(yy, pp)),
                "brier": float(brier_score_loss(yy, pp)),
            })
    return rows


def _bootstrap_endpoint(payload):
    endpoint, yy, pp, endpoint_groups, reps, seed = payload
    rng = np.random.default_rng(seed)
    unique_groups = None if endpoint_groups is None else np.unique(endpoint_groups)
    values = {name: [] for name in pp}
    deltas = {name: [] for name in pp if name != "semantic_transformer"}
    for _ in range(reps):
        if endpoint_groups is None:
            index = rng.integers(0, len(yy), len(yy))
            weight = np.bincount(index, minlength=len(yy))
        else:
            sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
            group_counts = pd.Series(sampled).value_counts()
            weight = pd.Series(endpoint_groups).map(group_counts).fillna(0).to_numpy(dtype=int)
        positive_weight = weight > 0
        if np.unique(yy[positive_weight]).size < 2:
            continue
        scores = {name: roc_auc_score(yy, pred, sample_weight=weight) for name, pred in pp.items()}
        for name, score in scores.items():
            values[name].append(score)
        for name in deltas:
            deltas[name].append(scores["semantic_transformer"] - scores[name])
    return endpoint, len(unique_groups) if unique_groups is not None else len(yy), values, deltas


def paired_bootstrap(
    core,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    clusters: np.ndarray | None,
    reps: int,
    seed: int,
    mode: str,
    unit: str,
    workers: int,
) -> list[dict]:
    groups = None if clusters is None else pd.Series(clusters).fillna("missing").astype(str).to_numpy()
    payloads = []
    for j, endpoint in enumerate(core.ENDPOINTS):
        valid = np.isfinite(y[:, j])
        payloads.append((
            endpoint,
            y[valid, j].astype(int),
            {name: values[valid, j] for name, values in predictions.items()},
            None if groups is None else groups[valid],
            reps,
            seed + j,
        ))
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
        results = list(executor.map(_bootstrap_endpoint, payloads))
    rows: list[dict] = []
    for endpoint, n_clusters, values, deltas in results:
        for name, distribution in values.items():
            if not distribution:
                raise RuntimeError(f"No valid {unit} bootstrap samples for {endpoint}/{name}")
            rows.append({
                "mode": mode,
                "mode_class": MODE_CLASS[mode],
                "uncertainty_unit": unit,
                "source_database": "MIMIC-III + MIMIC-IV",
                "target_database": "eICU",
                "endpoint": endpoint,
                "model": name,
                "n_clusters": n_clusters,
                "bootstrap_reps_requested": reps,
                "bootstrap_reps_valid": len(distribution),
                "auroc_lo": float(np.quantile(distribution, 0.025)),
                "auroc_hi": float(np.quantile(distribution, 0.975)),
                "delta_semantic_minus_model_lo": (
                    float(np.quantile(deltas[name], 0.025)) if name in deltas else np.nan
                ),
                "delta_semantic_minus_model_hi": (
                    float(np.quantile(deltas[name], 0.975)) if name in deltas else np.nan
                ),
            })
    return rows


def concept_audit(core, frames: dict[str, pd.DataFrame], mode: str) -> list[dict]:
    rows = []
    for database, frame in frames.items():
        raw = pd.read_csv(
            core.CF[database],
            usecols=lambda column: column in {
                "episode_id", "description", "event_type", "event_time_precision", "window_id"
            },
            dtype="string",
            low_memory=False,
        )
        raw = raw[pd.to_numeric(raw.window_id, errors="coerce").eq(0)].copy()
        if mode == "precise_time":
            keep = raw.event_time_precision.isin(["timestamp", "relative_minute"])
        elif mode == "medication_only":
            keep = raw.event_type.eq("medication")
        else:
            proximal = (
                r"dialys|renal replacement|hemofiltration|hemodia|haemodia|cvvh|crrt|"
                r"do not resuscitate|\bdnr\b|palliative|comfort care|expired|death"
            )
            keep = ~raw.description.fillna("").str.contains(proximal, case=False, regex=True)
        nonempty = frame.descriptions.apply(len).gt(0)
        rows.append({
            "mode": mode,
            "mode_class": MODE_CLASS[mode],
            "database": database,
            "episodes": len(frame),
            "episodes_with_concepts": int(nonempty.sum()),
            "episode_concept_coverage": float(nonempty.mean()),
            "retained_unique_descriptions": len({value for values in frame.descriptions for value in values}),
            "retained_concept_events": int(frame.n_events.fillna(0).sum()),
            "all_first_window_concept_events": len(raw),
            "filtered_concept_events": int(keep.sum()),
            "concept_events_removed": int((~keep).sum()),
            "episodes_affected_by_filter": int(raw.loc[~keep, "episode_id"].nunique()),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    core = load_core()
    core.seed(core.SEED)
    output = args.output_dir or PROJECT / "05_results_derived" / f"04_semantic_sensitivity_{args.mode}"
    if output.exists() and any(output.iterdir()) and not (args.overwrite or args.resume_bootstrap):
        raise FileExistsError(f"{output} is not empty; use a new directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    frames, _ = core.frames(args.mode)
    if args.max_episodes is not None:
        frames = {
            name: frame.sample(min(args.max_episodes, len(frame)), random_state=core.SEED).reset_index(drop=True)
            for name, frame in frames.items()
        }
    audit = concept_audit(core, frames, args.mode)
    pd.DataFrame(audit).to_csv(output / "concept_filter_audit.csv", index=False)

    sources = ["MIMIC-III", "MIMIC-IV"]
    source_name = " + ".join(sources)
    target = frames["eICU"]
    checkpoint = output / "prediction_checkpoint_sensitive.npz"
    if args.resume_bootstrap:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume without {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as saved:
            y = saved["y"]
            predictions = {name: saved[name] for name in core.MODELS}
        subject_id = target.subject_id.astype(str).to_numpy(dtype=str)
        hospital_id = target.hospital_id.astype(str).to_numpy(dtype=str)
        if len(y) != len(target) or not np.array_equal(y, target[core.ENDPOINTS].to_numpy(float), equal_nan=True):
            raise RuntimeError("Checkpoint labels do not match the current target contract")
    else:
        semantic = load_locked_embeddings()
        used_descriptions = {value for frame in frames.values() for values in frame.descriptions for value in values}
        missing = sorted(used_descriptions - semantic.keys())
        if missing:
            raise RuntimeError(f"Locked semantic lookup misses {len(missing)} retained descriptions")
        train = core.balanced_pool(frames, sources, core.SEED + 2)
        y = target[core.ENDPOINTS].to_numpy(float)
        predictions: dict[str, np.ndarray] = {}
        predictions["demographic"] = core.mlp_pred(train, target, args.epochs, core.SEED + 202)
        predictions["xgboost"], _ = core.xgb_pred(train, target, core.SEED + 202)
        predictions["random_transformer"], _, _, _ = core.transformer_pred(
            train, target, semantic, "random", args.epochs, core.SEED + 212
        )
        predictions["semantic_transformer"], _, _, _ = core.transformer_pred(
            train, target, semantic, "semantic", args.epochs, core.SEED + 222
        )
        subject_id = target.subject_id.astype(str).to_numpy(dtype=str)
        hospital_id = target.hospital_id.astype(str).to_numpy(dtype=str)
        np.savez_compressed(
            checkpoint, y=y, subject_id=subject_id, hospital_id=hospital_id, **predictions,
        )
        result_rows = point_metrics(core, y, predictions, args.mode)
        pd.DataFrame(result_rows).to_csv(output / "sensitivity_results.csv", index=False)
    bootstrap_rows = []
    bootstrap_rows += paired_bootstrap(
        core, y, predictions, None, args.bootstrap_reps, core.SEED + 232, args.mode, "episode", args.bootstrap_workers
    )
    bootstrap_rows += paired_bootstrap(
        core, y, predictions, subject_id, args.bootstrap_reps,
        core.SEED + 237, args.mode, "patient", args.bootstrap_workers
    )
    bootstrap_rows += paired_bootstrap(
        core, y, predictions, hospital_id, args.bootstrap_reps,
        core.SEED + 242, args.mode, "hospital", args.bootstrap_workers
    )
    pd.DataFrame(bootstrap_rows).to_csv(output / "sensitivity_bootstrap.csv", index=False)

    files = ["concept_filter_audit.csv", "sensitivity_results.csv", "sensitivity_bootstrap.csv"]
    manifest = {
        "status": "SMOKE_TEST" if args.max_episodes is not None else "SEMANTIC_SENSITIVITY_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "mode_class": MODE_CLASS[args.mode],
        "estimand": f"database-balanced {source_name} to eICU LODO",
        "endpoints": core.ENDPOINTS,
        "models": core.MODELS,
        "epochs": args.epochs,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_workers": args.bootstrap_workers,
        "bootstrap_seed_strategy": "base seed plus endpoint index; identical within-endpoint resamples across models",
        "bootstrap_implementation": "cluster/episode draw counts passed as sklearn AUROC sample weights; algebraically equivalent to row replication",
        "resumed_from_prediction_checkpoint": bool(args.resume_bootstrap),
        "max_episodes_test_only": args.max_episodes,
        "source_balance_episodes_per_database": min(len(frames[name]) for name in sources),
        "target_episodes": len(target),
        "target_patients": int(target.subject_id.astype(str).nunique()),
        "target_hospitals": int(target.hospital_id.astype(str).nunique()),
        "input_sha256": {
            "transitions": {name: sha256(path) for name, path in core.DB.items()},
            "concepts": {name: sha256(path) for name, path in core.CF.items()},
            "demographics": {
                name: [sha256(path) for path in paths if path is not None]
                for name, paths in core.RAW.items()
            },
            "locked_semantic_embedding_lookup": sha256(LOCKED / "semantic_embedding_lookup.npz"),
        },
        "script_sha256": sha256(Path(__file__)),
        "core_script_sha256": sha256(CORE_PATH),
        "software": {"python": platform.python_version(), "packages": core.package_versions()},
        "outputs": {name: sha256(output / name) for name in files},
        "interpretation_boundary": [
            "predictive transportability only",
            "target labels are not used for training or model selection",
            "precise_time leaves MIMIC-III without retained first-window concepts under the current source precision contract",
            "label_proximal_excluded is post hoc and not prespecified",
            "no causal, clinical-utility, or deployment claim",
        ],
    }
    (output / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "mode": args.mode, "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-workers", type=int, default=5)
    parser.add_argument("--resume-bootstrap", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Smoke-test cap only")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()