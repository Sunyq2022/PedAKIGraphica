# 00 Derived results index

1. `02_renal_state_feasibility/` — corrected layer-aware feasibility and eICU extraction audit.
2. `03_corrected_eicu_study/` — completed corrected clinical-state transport analysis,
   calibration, source-selected operating thresholds, 1,000-replicate clustered intervals,
   onset risk sets, and ablations. This is the sole active clinical-state result contract.
3. `04_semantic_sensitivity_precise_time/` — completed prespecified principal-LODO
   precise-time semantic sensitivity with 1,000 episode, patient and hospital resamples.
4. `05_semantic_sensitivity_medication_only/` — completed prespecified conservative
   medication-only sensitivity with the same uncertainty contract.
5. `06_semantic_sensitivity_label_proximal_excluded/` — completed post-hoc exclusion of
   dialysis/RRT and end-of-life descriptions; interpret as robustness, not causal ablation.
6. `07_semantic_sensitivity_summary.csv` — manuscript-facing semantic-model summary of
   point estimates, hospital-cluster intervals and paired semantic-minus-random intervals.
7. `08_eicu_creatinine_observability_audit/` — first-episode raw-source audit separating
   four-hour direct measurement sparsity from 24-hour and stay-level availability.
8. `09_supplementary_evaluation_source_data/` — aggregate source data for six supplementary
   figures, an inventory of 22 distinct evaluation modules, complete eICU state-model and primary
   LODO evaluation tables, and a SHA-256 reproducibility manifest. No model fitting or patient-level
   data access is performed by this display-only analysis.

The former `01_clinical_state_benchmark/` was a provisional diagnostic based on an incomplete
eICU extraction. It was removed after item 2 superseded it; file hashes and the reason for
retirement are retained in `../09_environment/04_submission_cleanup_audit.json`.

Future analyses must use the next free numbered directory and must not overwrite these outputs.