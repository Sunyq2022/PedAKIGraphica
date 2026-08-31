# ICU measurement and transportability study

This is the public reproducibility companion for the adult ICU measurement and
transportability study developed by the Children’s Health Data Science Team,
Children’s Hospital of Nanjing Medical University.

The study compares semantic and physiological-state representations across
MIMIC-III v1.4, MIMIC-IV v3.1 and the eICU Collaborative Research Database v2.0.
It contains six directed external-transfer analyses, leave-one-database-out
analyses, prespecified semantic-timing sensitivities, a post-hoc
label-proximity sensitivity, and a corrected eICU physiological-state analysis.

## Public scope

The package contains:

- `02_scripts/` — active analysis and aggregate-figure generation scripts;
- `04_results_locked/` — locked semantic-transfer aggregate results and the
  frozen description-embedding lookup;
- `05_results_derived/` — corrected eICU, sensitivity, feasibility and
  creatinine-observability aggregate results;
- `06_figures_locked/` — publication figures in PNG and PDF formats;
- `09_environment/requirements.txt` — the recorded Python environment.

Patient-level records, row-level transition tables, raw MIMIC/eICU files,
restricted source paths, hospital-level observability detail, prediction files,
and external reference-article assets are intentionally excluded. MIMIC and
eICU data must be accessed under their own credentialing, data-use and
publication conditions.

## Selected results

The corrected physiological-state XGBoost model reached AUROC 0.7298–0.7380
for next-window AKI progression and 0.7726–0.7759 for hospital death when
trained in MIMIC-III or MIMIC-IV and evaluated in eICU. The frozen semantic
model reached AUROC 0.5137 for AKI progression and 0.6486 for hospital death
in the principal balanced MIMIC-to-eICU leave-one-database-out analysis.
These are predictive transport results only; they are not causal effects,
treatment-policy values, clinical utility, clinical validation, or deployment
evidence.

## Reproduction boundary

The scripts are provided for transparent inspection and rerunning by
credentialed users with locally configured data. Before running, set
`TRANSITION_ROOT`, `MIMIC3_ROOT`, `MIMIC4_ROOT` and `EICU_ROOT` to local data
locations. The public package deliberately does not provide those datasets.

For aggregate-only supplementary figures and tables, use:

```powershell
python 02_scripts/07_build_supplementary_evaluation_figures.py
```

The current checked-in figures and result tables are the release artifacts;
reruns should write to a new versioned output directory and must not overwrite
the locked results.

## Interpretation boundary

The study supports an endpoint-dependent distinction between semantic coding
alignment, physiological-state availability and source-extraction quality.
It does not support claims about reinforcement-learning value, treatment
effects, counterfactual policy performance, clinical utility, or prospective
deployment.
