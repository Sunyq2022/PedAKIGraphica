"""Validate key manuscript numbers against the locked semantic CSVs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "04_results_locked" / "01_semantic_audited_results"
MANUSCRIPT = ROOT / "07_manuscript" / "01_current_manuscript" / "01_draft.md"
MANIFEST = MANUSCRIPT.with_name("02_claim_to_result_manifest.json")
ENDPOINTS = ["next_aki_progression", "next_aki_stage2plus", "next_aki_stage3", "hospital_death", "icu_death"]


def f(value):
    return f"{float(value):.4f}"


def main():
    text = MANUSCRIPT.read_text(encoding="utf-8").replace("−", "-").replace("–", "-")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["authoritative_root"] == "04_results_locked/01_semantic_audited_results"
    lodo = pd.read_csv(LOCKED / "lodo_results.csv")
    episode = pd.read_csv(LOCKED / "external_bootstrap_intervals.csv")
    hospital = pd.read_csv(LOCKED / "hospital_cluster_bootstrap.csv")
    filt = {"source_database": "MIMIC-III + MIMIC-IV", "target_database": "eICU"}
    point = lodo.copy()
    for key, value in filt.items():
        point = point[point[key].eq(value)]
    point = point[point.model.eq("semantic_transformer")].set_index("endpoint")
    ep = episode[(episode.evaluation.eq("lodo")) & (episode.model.eq("semantic_transformer"))]
    hp = hospital[(hospital.evaluation.eq("lodo")) & (hospital.model.eq("semantic_transformer"))]
    for key, value in filt.items():
        ep = ep[ep[key].eq(value)]
        hp = hp[hp[key].eq(value)]
    ep = ep.set_index("endpoint")
    hp = hp.set_index("endpoint")
    paired = episode[(episode.evaluation.eq("lodo")) & (episode.model.eq("random_transformer"))]
    for key, value in filt.items():
        paired = paired[paired[key].eq(value)]
    paired = paired.set_index("endpoint")
    for endpoint in ENDPOINTS:
        assert f(point.loc[endpoint, "auroc"]) in text, endpoint
        assert f(ep.loc[endpoint, "auroc_lo"]) in text and f(ep.loc[endpoint, "auroc_hi"]) in text, endpoint
        assert f(hp.loc[endpoint, "auroc_lo"]) in text and f(hp.loc[endpoint, "auroc_hi"]) in text, endpoint
        assert f(paired.loc[endpoint, "delta_vs_semantic_lo"]) in text, endpoint
        assert f(paired.loc[endpoint, "delta_vs_semantic_hi"]) in text, endpoint
    obsolete = ["0.5117", "0.5190", "0.5578", "0.6511", "0.6476", "0.5775"]
    assert not any(value in text for value in obsolete), "Obsolete primary LODO value remains"
    sensitivity_claims = {
        "04_semantic_sensitivity_precise_time": ["next_aki_progression", "hospital_death", "icu_death"],
        "05_semantic_sensitivity_medication_only": ["hospital_death", "icu_death"],
        "06_semantic_sensitivity_label_proximal_excluded": ["next_aki_stage3", "hospital_death", "icu_death"],
    }
    for directory, claimed_endpoints in sensitivity_claims.items():
        root = ROOT / "05_results_derived" / directory
        result = pd.read_csv(root / "sensitivity_results.csv")
        bootstrap = pd.read_csv(root / "sensitivity_bootstrap.csv")
        semantic = result[result.model.eq("semantic_transformer")]
        hospital_semantic = bootstrap[
            bootstrap.model.eq("semantic_transformer") & bootstrap.uncertainty_unit.eq("hospital")
        ]
        for endpoint in claimed_endpoints:
            value = semantic[semantic.endpoint.eq(endpoint)].auroc.item()
            assert f(value) in text, f"Missing sensitivity AUROC {directory}/{endpoint}/{f(value)}"
        if not directory.endswith("label_proximal_excluded"):
            for endpoint in ["hospital_death", "icu_death"]:
                row = hospital_semantic[hospital_semantic.endpoint.eq(endpoint)].iloc[0]
                assert f(row.auroc_lo) in text and f(row.auroc_hi) in text
    state_root = ROOT / "05_results_derived" / "03_corrected_eicu_study"
    state = pd.read_csv(state_root / "corrected_external_results.csv")
    state_hospital = pd.read_csv(state_root / "corrected_hospital_bootstrap.csv")
    mortality = state[
        state.target_database.eq("eICU") & state.model.eq("state_xgboost")
        & state.endpoint.isin(["hospital_death", "icu_death"])
    ]
    mortality_hospital = state_hospital[
        state_hospital.target_database.eq("eICU") & state_hospital.model.eq("state_xgboost")
        & state_hospital.endpoint.isin(["hospital_death", "icu_death"])
    ]
    assert len(mortality) == 4 and len(mortality_hospital) == 4
    for row in mortality.itertuples():
        assert f(row.auroc) in text
        assert f(row.average_precision) in text
    assert f(mortality.calibration_slope.min()) in text
    assert f(mortality.calibration_slope.max()) in text
    for row in mortality_hospital.itertuples():
        assert f(row.auroc_lo) in text and f(row.auroc_hi) in text
        assert row.bootstrap_reps == 1000
    inventory = pd.read_csv(
        ROOT / "05_results_derived" / "09_supplementary_evaluation_source_data"
        / "SupplementaryTable1_evaluation_inventory.csv"
    )
    assert len(inventory) == 22 and inventory.evaluation_id.tolist() == list(range(1, 23))
    print("PASS: locked semantic, sensitivity, corrected-state and supplementary contracts match the manuscript")


if __name__ == "__main__":
    main()