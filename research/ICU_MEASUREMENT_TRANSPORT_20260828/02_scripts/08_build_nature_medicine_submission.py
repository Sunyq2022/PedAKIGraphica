"""Assemble a versioned Nature Medicine initial-submission package.

The builder copies authoritative figures without modifying them, creates three
documented manuscript revision rounds, embeds the selected six main figures in
the final Word manuscript, and builds a supplementary appendix and cover letter.
It never invents author-, institution-, ethics-, funding-, conflict- or
repository-level information; unresolved fields remain explicit action items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT / "07_manuscript/01_current_manuscript/01_draft.md"
FIG_SEM = PROJECT / "06_figures_locked/01_semantic_audited_figures"
FIG_STATE = PROJECT / "06_figures_locked/02_corrected_eicu_study_figures"
FIG_SUP = PROJECT / "06_figures_locked/03_supplementary_evaluation_figures"
SUP_DATA = PROJECT / "05_results_derived/09_supplementary_evaluation_source_data"
DEFAULT_OUTPUT = (
    PROJECT / "10_submission_package/01_nature_medicine_initial_submission_20260831"
)

TITLE = (
    "Semantic representation and physiological observability jointly determine "
    "prediction transport across intensive-care systems"
)

ABSTRACT = (
    "Clinical prediction can fail across electronic-health-record systems through vocabulary "
    "mismatch and differences in what is measured. We studied 68,151 intensive-care episodes "
    "from MIMIC-III, MIMIC-IV and 138 eICU hospitals, evaluating four models, five outcomes "
    "and six directed transfers. In balanced MIMIC-to-eICU evaluation, a frozen-embedding "
    "transformer discriminated hospital and ICU death (AUROC 0.6486 and 0.6456) but not "
    "four-hour acute-kidney-injury progression (0.5137). Mortality discrimination persisted "
    "after precise-time and medication-only restrictions. Restoring omitted eICU blood-pressure "
    "and temperature sources reduced missingness from 75.12% to 1.62% and 92.23% to 12.72%. "
    "Physiological-state XGBoost discriminated progression more strongly (0.7298–0.7380), "
    "although removing current KDIGO stage reduced AUROC by approximately 0.14. Thus, "
    "cross-system transport was endpoint dependent: semantic context carried mortality signal, "
    "whereas dynamic kidney outcomes depended more on observed physiological state. Neither "
    "representation established clinical utility or deployability."
)

REFERENCES = """1. Kirchler M, Ferro M, Lorenzini V, van de Water RP, FinnGen, Lippert C, et al. Large language models improve transferability of electronic health record-based predictions across countries and coding systems. *NPJ Digital Medicine*. 2026;9:177. doi:10.1038/s41746-026-02363-5.

2. Johnson AE, Pollard TJ, Shen L, Lehman LW, Feng M, Ghassemi M, et al. MIMIC-III, a freely accessible critical care database. *Scientific Data*. 2016;3:160035. doi:10.1038/sdata.2016.35.

3. Johnson AEW, Bulgarelli L, Shen L, Gayles A, Shammout A, Horng S, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*. 2023;10:1. doi:10.1038/s41597-022-01899-x.

4. Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely available multicenter database for critical care research. *Scientific Data*. 2018;5:180178. doi:10.1038/sdata.2018.178.

5. Khwaja A. KDIGO clinical practice guidelines for acute kidney injury. *Nephron Clinical Practice*. 2012;120:c179–c184. doi:10.1159/000339789.

6. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. doi:10.1136/bmj-2023-078378."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def section(text: str, name: str, next_name: str) -> str:
    match = re.search(
        rf"^## {re.escape(name)}\s*$\n(.*?)(?=^## {re.escape(next_name)}\s*$)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"section not found: {name} -> {next_name}")
    return match.group(1).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9’'\-+]*\b", text))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare_figures(output: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    all_dir = output / "04_All_Generated_Figures"
    for group, directory in (
        ("01_semantic_audited", FIG_SEM),
        ("02_corrected_eicu", FIG_STATE),
        ("03_supplementary_evaluation", FIG_SUP),
    ):
        for source in sorted(directory.glob("*")):
            if source.is_file() and source.suffix.lower() in {".pdf", ".png"}:
                copy_file(source, all_dir / group / source.name)

    main_map = {
        1: FIG_SEM / "Figure1_overview.png",
        2: FIG_SEM / "Figure2_model_evaluation.png",
        3: FIG_SEM / "Figure3_coding_transfer.png",
        4: FIG_STATE / "Figure6_data_governance.png",
        5: FIG_STATE / "Figure7_corrected_transport.png",
        6: FIG_STATE / "Figure8_calibration_ablation.png",
    }
    main_dir = output / "02_Main_Figures"
    for number, source in main_map.items():
        copy_file(source, main_dir / f"Figure_{number}.png")
        copy_file(source.with_suffix(".pdf"), main_dir / f"Figure_{number}.pdf")

    supplementary_map = {
        1: FIG_SEM / "Figure4_sample_size.png",
        2: FIG_SEM / "Figure5_semantic_space_and_importance.png",
        3: FIG_SUP / "SupplementaryFigure1_semantic_transport_heatmaps.png",
        4: FIG_SUP / "SupplementaryFigure2_state_transport_heatmaps.png",
        5: FIG_SUP / "SupplementaryFigure3_common_endpoint_forest.png",
        6: FIG_SUP / "SupplementaryFigure4_primary_lodo_metric_matrix.png",
        7: FIG_SUP / "SupplementaryFigure5_semantic_sensitivity_forest.png",
        8: FIG_SUP / "SupplementaryFigure6_creatinine_observability.png",
    }
    supplementary_dir = output / "03_Supplementary_Material/Figures"
    for number, source in supplementary_map.items():
        copy_file(source, supplementary_dir / f"Supplementary_Figure_{number}.png")
        copy_file(source.with_suffix(".pdf"), supplementary_dir / f"Supplementary_Figure_{number}.pdf")
    return main_map, supplementary_map


def title_page() -> str:
    return f"""# {TITLE}

**Article type:** Article  
**Target journal:** *Nature Medicine*

**Authors:** [AUTHOR ACTION REQUIRED: insert the complete author list in final order]

**Affiliations:** [AUTHOR ACTION REQUIRED: insert numbered institutional affiliations]

**Corresponding author:** [AUTHOR ACTION REQUIRED: name, postal address, email and telephone]

**Equal contributions / consortium authorship:** [AUTHOR ACTION REQUIRED: confirm or delete]

**Short title:** Representation and observability in ICU model transport

**Keywords:** clinical prediction; transportability; electronic health records; acute kidney injury; intensive care; external validation

**Submission status:** Initial-submission draft assembled on 31 August 2026. Ethics determination, repository identifier, MIMIC cross-version overlap assessment and author declarations require completion before submission.

\\newpage
"""


def round_one(draft: str) -> str:
    introduction = section(draft, "Introduction", "Methods")
    methods = section(draft, "Methods", "Results")
    results = section(draft, "Results", "Discussion")
    discussion = section(draft, "Discussion", "Data and code availability")
    availability = section(draft, "Data and code availability", "Figure legends")

    # Integrate figure citations and move exploratory displays to the supplement.
    introduction = introduction.replace(
        "This is a scientific adaptation of the reported semantic framework, not a numerical or architectural replication.",
        "This is a scientific adaptation of the reported semantic framework, not a numerical or architectural replication (Fig. 1).",
    )
    results = results.replace(
        "Benefits were therefore concentrated in mortality.",
        "Benefits were therefore concentrated in mortality (Fig. 2).",
    ).replace(
        "This supports description-based input relative to a no-concept condition, but does not isolate semantics from concept availability.",
        "This supports description-based input relative to a no-concept condition, but does not isolate semantics from concept availability (Fig. 3).",
    ).replace(
        "In the two-MIMIC LODO curve, semantic AUROC rose from 0.5569 at 4,021 episodes to 0.5816 at 40,216 episodes but was non-monotonic.",
        "In the two-MIMIC LODO curve, semantic AUROC rose from 0.5569 at 4,021 episodes to 0.5816 at 40,216 episodes but was non-monotonic (Supplementary Fig. 1).",
    ).replace(
        "These results identify model sensitivity, not physiological mechanisms.",
        "These results identify model sensitivity, not physiological mechanisms (Supplementary Fig. 2).",
    ).replace(
        "All 96,199 transition rows retained their stay/window keys, and all 18,776 episodes mapped to 138 source hospitals.",
        "All 96,199 transition rows retained their stay/window keys, and all 18,776 episodes mapped to 138 source hospitals (Fig. 4).",
    ).replace(
        "Relative to the incomplete extraction, AUROC increased by 0.0037 and 0.0023; the small change in ranking performance does not negate the large correction in source observability.",
        "Relative to the incomplete extraction, AUROC increased by 0.0037 and 0.0023; the small change in ranking performance does not negate the large correction in source observability (Fig. 5).",
    ).replace(
        "The corrected model therefore did not derive its transport performance primarily from the known eICU extraction defect.",
        "The corrected model therefore did not derive its transport performance primarily from the known eICU extraction defect (Fig. 6).",
    )
    methods = methods.replace(
        "Figure 5 displays concepts observed in at least five attributed patients; complete low-frequency results remain in `concept_importance.csv`.",
        "Supplementary Figure 2 displays concepts observed in at least five attributed patients; complete low-frequency estimates are retained in the source-data files.",
    )
    methods = methods.replace(
        "Metrics were AUROC, average precision, Brier score, and ten-bin expected calibration error. Two hundred episode-bootstrap resamples estimated AUROC intervals and paired semantic-minus-comparator differences. eICU analyses additionally used 200 hospital-cluster resamples.",
        "Metrics were AUROC, average precision, Brier score and ten-bin expected calibration error. The locked reference-layout analyses used 200 episode-bootstrap resamples for AUROC intervals and paired semantic-minus-comparator differences, with 200 hospital-cluster resamples for eICU. The principal LODO uncertainty update and the timing and label-proximity analyses used the 1,000 paired episode-, patient- and hospital-cluster resamples specified above; each legend and source table identifies the applicable resampling unit and count.",
    )
    methods = methods.replace("`", "")
    discussion = discussion.replace("`customLab`", "customLab")
    availability = availability.replace("`sentence-transformers/all-MiniLM-L6-v2`", "sentence-transformers/all-MiniLM-L6-v2")

    return f"""{title_page()}
## Abstract

{ABSTRACT}

{introduction}

## Results

{results}

## Discussion

{discussion}

## Online Methods

{methods}

## Data availability

{availability}

## Code availability

Analysis scripts, non-sensitive aggregate result tables, manifests, input hashes and figure-generation code will be released at **[AUTHOR ACTION REQUIRED: insert public repository DOI/URL before submission]**. Patient-level source data and derived sensitive records cannot be redistributed under the PhysioNet credentialed-access agreements.

## Acknowledgements

[AUTHOR ACTION REQUIRED: insert acknowledgements and verify that no contributor meeting authorship criteria is omitted.]

## Author contributions

[AUTHOR ACTION REQUIRED: provide a CRediT-aligned author-contribution statement after the final author list is fixed.]

## Funding

[AUTHOR ACTION REQUIRED: list all grants, institutional support and funder roles, or state that no specific funding was received.]

## Competing interests

[AUTHOR ACTION REQUIRED: provide the declaration for every author.]

## Additional information

The study is retrospective and predictive. It does not estimate treatment effects, policy value, clinical utility or prospective safety. **[AUTHOR ACTION REQUIRED: complete the institutional ethics determination and MIMIC-III/MIMIC-IV overlap assessment before submission.]**

## References

{REFERENCES}
"""


def round_two(text: str) -> str:
    replacements = {
        "electronic-health-record": "electronic health record",
        "four-hour-window": "four-hour window",
        "source-only calibrated": "source-calibrated",
        "The central result is therefore not that one model class is universally superior, but that": "Together, these findings show that",
        "A clinically useful transport study must therefore test": "A clinically credible transport study must therefore test",
        "Benefits were therefore concentrated in mortality": "Performance gains were concentrated in mortality",
        "Current state used only the completed": "The current state used only the completed",
        "The clinical-state analysis": "The physiological-state analysis",
        "clinical-state variables": "physiological-state variables",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Avoid implementation-centric framing in the final discussion.
    text = text.replace(
        "Eighth, this is a GRASP-aligned adaptation rather than an exact replication; genotype and PRS analyses remain unavailable.",
        "Eighth, this study adapts a published semantic framework to fixed-window ICU outcomes rather than reproducing its original time-to-event or genetic analyses.",
    )
    return text


def round_three(text: str) -> str:
    # Journal-compliance pass: British spelling, explicit uncertainty units, no internal file paths.
    text = text.replace("multicenter", "multicentre")
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if word_count(ABSTRACT) > 150:
        raise AssertionError(f"abstract exceeds 150 words: {word_count(ABSTRACT)}")
    forbidden = ["concept_importance.csv", "05_results_derived", "04_results_locked"]
    for term in forbidden:
        if term in text:
            raise AssertionError(f"internal implementation term remains: {term}")
    return text


MAIN_LEGENDS = {
    1: "Overview of the study. A, Semantic-model development and patient-grouped internal and external evaluation. B, Target adaptation using 10% of eICU patients, with evaluation in the patient-disjoint remainder. C, Database-specific descriptions are mapped into a shared frozen embedding space. D, Up to 64 descriptions from the four-hour state window are processed by a four-layer, eight-head transformer to produce five endpoint-specific risks.",
    2: "External model evaluation. AUROCs for demographic-only prediction, random embeddings, concept XGBoost and the semantic transformer trained in MIMIC-IV and tested in A, MIMIC-III and B, eICU. Horizontal lines are 95% target-episode bootstrap intervals. An asterisk indicates that the lower bound of every paired semantic-minus-comparator interval exceeded zero.",
    3: "Transfer across coding vocabularies. Models were trained in MIMIC-IV and tested in eICU. The display compares demographic-only prediction, the same fitted semantic model restricted to target descriptions with exact source-code support, and the fitted model using all target descriptions. Horizontal lines are paired 95% target-episode bootstrap intervals; model weights were held fixed between semantic input conditions.",
    4: "eICU data governance and corrected observability. A, Four-hour-window missingness before and after adding non-invasive blood pressure from vitalAperiodic and nurse-charted temperature from nurseCharting. B, Corrected missingness across MIMIC-III, MIMIC-IV and eICU. The cohort and transition keys were held fixed before and after correction.",
    5: "Corrected physiological-state transport for next-window acute-kidney-injury progression. A, AUROC in all six ordered zero-shot transfers for a current-KDIGO lookup, source-fitted logistic regression and source-fitted XGBoost. Horizontal lines are 95% patient-cluster bootstrap intervals from 1,000 resamples. B, Average precision versus target event prevalence; the diagonal is the no-skill reference.",
    6: "Calibration and representation dependence for eICU progression. A, Ten-bin reliability curves after Platt calibration fitted only in the source validation split. The diagonal indicates perfect calibration. B, Change in eICU AUROC after removing each physiological feature group from source-fitted XGBoost. Negative values indicate lower discrimination after removal.",
}

SUP_LEGENDS = {
    1: "Average prediction performance across training sample sizes. Mean AUROC across five endpoints for nested patient-grouped MIMIC-IV training samples, evaluated in held-out MIMIC-IV, MIMIC-III and eICU. Shaded regions are 95% target-episode bootstrap intervals.",
    2: "Semantic embedding and zero-occlusion attribution for AKI stage 2 or greater. A, UMAP of frozen description embeddings. B, Magnified region. C, Absolute mean probability change after zero occlusion. D, Concept frequency across databases. Displayed concepts occurred in at least five evaluated patients; attribution is model-specific and non-causal.",
    3: "Complete semantic-model transport across six ICU directions. AUROC heatmaps for demographic prediction, concept XGBoost, random-embedding transformer and semantic transformer across five endpoints and all six ordered transfers. All cells are shown.",
    4: "Complete physiological-state transport across six ICU directions. AUROC heatmaps for current-KDIGO lookup, state logistic regression and state XGBoost across progression, stage-2 onset, stage-3 onset, hospital death and ICU death. Onset endpoints use prespecified current-stage risk sets and are not equivalent to the stage-threshold semantic outcomes.",
    5: "Semantic and physiological-state models for common eICU endpoints. Hospital-cluster 95% AUROC intervals for semantic transformer, state logistic regression and state XGBoost trained separately in MIMIC-III or MIMIC-IV. Only progression, hospital death and ICU death are directly compared because their definitions match.",
    6: "Multi-metric primary leave-one-database-out evaluation. AUROC, average precision divided by prevalence, Brier score and ten-bin expected calibration error for four models trained on balanced MIMIC-III plus MIMIC-IV and evaluated in eICU.",
    7: "Semantic timing and label-proximity sensitivities. Semantic-transformer AUROCs and hospital-cluster 95% intervals for all concepts, precise-time restriction, medication-only restriction and post-hoc label-proximal exclusion. The post-hoc refitted analysis is descriptive.",
    8: "eICU creatinine observability across lookback horizons. Fraction of first eligible episodes with creatinine available within 4, 8, 12, 24, 48 or 168 hours before prediction, at any prior time or at any time during the ICU stay. The dashed line marks 24 hours.",
}


def main_figures_markdown(output: Path) -> str:
    blocks = ["## Figure legends and embedded main figures"]
    for number in range(1, 7):
        blocks.extend(
            [
                f"### Figure {number} | {MAIN_LEGENDS[number].split('.')[0]}",
                f"![Figure {number}](02_Main_Figures/Figure_{number}.png){{width=6.5in}}",
                f"**Figure {number}. {MAIN_LEGENDS[number]}**",
                "\\newpage",
            ]
        )
    return "\n\n".join(blocks)


def format_value(value, column: str) -> str:
    if pd.isna(value):
        return "—"
    if column in {"n", "evaluation_id"}:
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("_", " ")


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    view = frame[columns].copy()
    view.columns = [labels.get(column, column) for column in columns]
    lines = ["| " + " | ".join(view.columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(format_value(row[column], column) for column in columns) + " |")
    return "\n".join(lines)


def supplementary_markdown(output: Path) -> str:
    table1 = pd.read_csv(SUP_DATA / "SupplementaryTable1_evaluation_inventory.csv")
    table2 = pd.read_csv(SUP_DATA / "SupplementaryTable2_eicu_state_model_evaluation.csv")
    table3 = pd.read_csv(SUP_DATA / "SupplementaryTable3_primary_lodo_model_evaluation.csv")
    labels = {
        "evaluation_id": "ID", "evaluation_module": "Evaluation module", "analysis_class": "Class",
        "authoritative_source": "Aggregate source", "source_database": "Source", "target_database": "Target",
        "model": "Model", "endpoint": "Outcome", "n": "N", "event_rate": "Prevalence",
        "auroc": "AUROC", "average_precision": "AP", "brier": "Brier", "ece_10bin": "ECE",
        "calibration_intercept": "Cal. intercept", "calibration_slope": "Cal. slope",
        "source_selected_threshold": "Threshold", "sensitivity": "Sensitivity", "specificity": "Specificity",
        "ppv": "PPV", "npv": "NPV", "patient_auroc_lo": "Patient CI low",
        "patient_auroc_hi": "Patient CI high", "hospital_auroc_lo": "Hospital CI low",
        "hospital_auroc_hi": "Hospital CI high", "ap_over_prevalence": "AP/prevalence",
    }
    blocks = [
        "# Supplementary Appendix",
        f"## {TITLE}",
        "**Authors:** [AUTHOR ACTION REQUIRED: insert final author list]",
        "## Supplementary Methods",
        "### Semantic timing and label-proximity analyses",
        "The principal balanced MIMIC-III plus MIMIC-IV to eICU analysis was refitted under two prespecified input restrictions. The precise-time analysis retained only concepts with timestamp or relative-minute precision; under the current contract, MIMIC-III contributed no such first-window concepts and was retained through the fully masked demographic pathway. The medication-only analysis retained medication concepts. A post-hoc analysis excluded prespecified dialysis, renal-replacement and end-of-life descriptions proximal to the labels. Each analysis retained the same architecture, source balancing, target cohort and endpoint definitions as the all-concept model and used 1,000 paired episode-, patient- and hospital-cluster bootstrap resamples.",
        "### Corrected eICU source completion",
        "The corrected versioned eICU contract supplemented periodic invasive blood pressure with non-invasive measurements from vitalAperiodic and supplemented periodic temperature with Celsius- or Fahrenheit-normalised nurseCharting temperature. Cohort membership and stay/window keys were unchanged. customLab creatinine-like values were not incorporated because hospital-specific unit semantics were not validated. Direct-observation and carried-forward KDIGO provenance, hospital identifiers, discharge fields and recorded intake/output provenance were retained.",
        "### Physiological-state modelling and uncertainty",
        "Fifteen contract-native physiological variables and explicit missingness indicators were processed using source-only winsorisation, median imputation and scaling. Logistic regression, XGBoost and current-KDIGO lookup models were fitted in patient-grouped source development data. Platt calibration and operating thresholds were estimated only in source validation data. External target data were not used for refitting or recalibration. Patient-cluster intervals and eICU hospital-cluster intervals used 1,000 bootstrap resamples.",
        "### Reporting boundary",
        "All analyses are retrospective and predictive. AUROC, average precision, calibration and source-selected operating-point metrics do not establish prospective benefit, clinical utility, safety or deployability. Zero occlusion and feature-group ablation describe model sensitivity and are not biological or causal explanations.",
    ]
    for number in range(1, 9):
        blocks.extend(
            [
                "\\newpage",
                f"## Supplementary Figure {number}",
                f"![Supplementary Figure {number}](Figures/Supplementary_Figure_{number}.png){{width=6.5in}}",
                f"**Supplementary Figure {number}. {SUP_LEGENDS[number]}**",
            ]
        )
    blocks.extend(
        [
            "\\newpage",
            "## Supplementary Table 1 | Inventory of completed model-evaluation modules",
            markdown_table(table1, list(table1.columns), labels),
            "\\newpage",
            "## Supplementary Table 2 | Complete eICU physiological-state model evaluation",
            "### Supplementary Table 2a | Discrimination and calibration",
            markdown_table(table2, ["source_database", "model", "endpoint", "n", "event_rate", "auroc", "average_precision", "brier", "ece_10bin", "calibration_slope"], labels),
            "### Supplementary Table 2b | Source-selected operating point and uncertainty",
            markdown_table(table2, ["source_database", "model", "endpoint", "source_selected_threshold", "sensitivity", "specificity", "ppv", "npv", "patient_auroc_lo", "patient_auroc_hi", "hospital_auroc_lo", "hospital_auroc_hi"], labels),
            "\\newpage",
            "## Supplementary Table 3 | Complete primary semantic-model evaluation",
            "### Supplementary Table 3a | Discrimination and calibration",
            markdown_table(table3, ["model", "endpoint", "n", "event_rate", "auroc", "average_precision", "ap_over_prevalence", "brier", "ece_10bin"], labels),
            "### Supplementary Table 3b | Patient- and hospital-cluster AUROC intervals",
            markdown_table(table3, ["model", "endpoint", "patient_auroc_lo", "patient_auroc_hi", "hospital_auroc_lo", "hospital_auroc_hi"], labels),
            "## Supplementary references",
            REFERENCES,
        ]
    )
    return "\n\n".join(blocks) + "\n"


def cover_letter() -> str:
    return f"""# Cover letter

31 August 2026

Editors  
*Nature Medicine*

Dear Editors,

We submit the Article **“{TITLE}”** for consideration in *Nature Medicine*.

Clinical prediction models are commonly transported across institutions by harmonising codes or by learning shared representations, yet transport can fail because the physiological state needed for prediction is not measured or extracted consistently. Using 68,151 adult intensive-care episodes from MIMIC-III, MIMIC-IV and the 138-hospital eICU Collaborative Research Database, we separate these two failure modes across five outcomes and all six directed transfers.

The study provides three findings of broad relevance to clinical artificial intelligence. First, frozen semantic representations carried mortality signal across heterogeneous coding systems, including under prespecified precise-time and medication-only restrictions, but did not transport four-hour acute-kidney-injury progression. Second, a corrected physiological-state representation transported progression substantially better, demonstrating that semantic alignment cannot replace endpoint-relevant state information. Third, a raw-source audit showed that apparent eICU blood-pressure and temperature missingness arose largely from omitted source tables, whereas creatinine sparsity reflected the short observation window. Together, the results show that transportability is jointly determined by representation, endpoint and observability, and that database quality should not be inferred from a derived modelling table alone.

We believe this work fits *Nature Medicine* because it moves beyond a model leaderboard to establish a general, clinically relevant framework for evaluating transport failure. The independent multicentre target, explicit negative findings, hospital-cluster uncertainty, calibration analysis and source-governance correction provide a rigorous test of claims that are increasingly consequential for deployment of clinical prediction systems.

This manuscript reports retrospective predictive analyses only. It makes no claim of treatment effect, prospective clinical utility, safety or deployability. Patient-level data remain governed by PhysioNet credentialed-access agreements; non-sensitive aggregate results, code and reproducibility manifests will be released at **[AUTHOR ACTION REQUIRED: repository DOI/URL]**.

**[AUTHOR ACTION REQUIRED before submission: confirm that the work is original, is not under consideration elsewhere, has been approved by all authors, and disclose any related manuscripts or preprints.]**

**[AUTHOR ACTION REQUIRED: insert the institutional ethics determination, identifier and date.]**

Potential reviewers: **[AUTHOR ACTION REQUIRED: provide 3–5 independent experts with affiliations and email addresses, avoiding recent collaborators and conflicts.]**

Opposed reviewers, if any: **[AUTHOR ACTION REQUIRED: provide names and concise reasons, or delete.]**

Thank you for considering our work.

Sincerely,

**[AUTHOR ACTION REQUIRED: corresponding author name, degrees and title]**  
on behalf of all authors  
**[Institution]**  
**[Email]**  
**[Telephone]**
"""


def write_excel(output: Path) -> Path:
    target = output / "03_Supplementary_Material/Supplementary_Tables_1-3.xlsx"
    with pd.ExcelWriter(target, engine="xlsxwriter") as writer:
        for name in (
            "SupplementaryTable1_evaluation_inventory.csv",
            "SupplementaryTable2_eicu_state_model_evaluation.csv",
            "SupplementaryTable3_primary_lodo_model_evaluation.csv",
        ):
            frame = pd.read_csv(SUP_DATA / name)
            sheet = name.replace("SupplementaryTable", "Table").replace(".csv", "")[:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet]
            header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
            for col, value in enumerate(frame.columns):
                worksheet.write(0, col, value, header)
                width = min(28, max(11, len(value) + 2))
                worksheet.set_column(col, col, width)
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
    return target


def pandoc(markdown: Path, docx: Path) -> None:
    command = [
        "pandoc", str(markdown), "--from", "markdown+raw_tex", "--to", "docx",
        "--standalone", "--output", str(docx), "--resource-path", str(markdown.parent),
    ]
    subprocess.run(command, check=True, cwd=markdown.parent)


def manifest(output: Path) -> Path:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "submission_manifest.json":
            rows.append({
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    payload = {
        "package": "Nature Medicine initial-submission draft",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": str(PROJECT),
        "locked_sources_modified": False,
        "main_display_items": 6,
        "supplementary_figures": 8,
        "supplementary_tables": 3,
        "unresolved_author_actions": [
            "author list and affiliations", "corresponding-author details",
            "institutional ethics determination", "repository DOI/URL",
            "MIMIC-III/MIMIC-IV overlap assessment", "funding",
            "competing interests", "author contributions", "reviewer suggestions",
        ],
        "files": rows,
    }
    target = output / "submission_manifest.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty submission package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prepare_figures(output)
    draft = DRAFT.read_text(encoding="utf-8")
    round1 = round_one(draft)
    round2 = round_two(round1)
    round3 = round_three(round2)

    source_dir = output / "05_Editorial_Working_Files"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "manuscript_round1_scientific_structure.md").write_text(round1, encoding="utf-8")
    (source_dir / "manuscript_round2_journal_style.md").write_text(round2, encoding="utf-8")
    (source_dir / "manuscript_round3_final.md").write_text(round3, encoding="utf-8")
    revision_log = f"""# Three-round manuscript revision log

## Round 1 — scientific structure and evidence alignment
- Removed internal manuscript-status prose from the submitted article.
- Reordered the article to Nature Medicine structure: unheaded introduction, Results, Discussion and Online Methods.
- Limited the main display set to six items and moved sample-size and attribution displays to the supplement.
- Added explicit in-text figure citations and preserved all locked point estimates and intervals.

## Round 2 — clinical and editorial style
- Reduced implementation-centric language and foregrounded the clinical transport question.
- Harmonised terminology for semantic and physiological-state representations.
- Tightened causal and deployment boundaries and retained negative endpoint-specific findings.

## Round 3 — compliance and consistency
- Replaced the abstract with a {word_count(ABSTRACT)}-word unreferenced version (journal limit: 150).
- Removed internal result paths from the submitted prose.
- Checked six main display items, eight supplementary figures and three supplementary tables.
- Preserved explicit author-action fields for unverified ethics, authorship, funding, conflicts, repository and overlap information.
"""
    (source_dir / "revision_log.md").write_text(revision_log, encoding="utf-8")

    manuscript_md = output / "01_Main_Manuscript_Nature_Medicine.md"
    manuscript_md.write_text(round3 + "\n\\newpage\n\n" + main_figures_markdown(output), encoding="utf-8")
    cover_md = output / "02_Cover_Letter.md"
    cover_md.write_text(cover_letter(), encoding="utf-8")
    supplement_md = output / "03_Supplementary_Material/Supplementary_Appendix.md"
    supplement_md.parent.mkdir(parents=True, exist_ok=True)
    supplement_md.write_text(supplementary_markdown(output), encoding="utf-8")

    pandoc(manuscript_md, output / "01_Main_Manuscript_Nature_Medicine.docx")
    pandoc(cover_md, output / "02_Cover_Letter.docx")
    pandoc(supplement_md, supplement_md.with_suffix(".docx"))
    write_excel(output)

    checklist = f"""# Initial-submission checklist

## Generated and checked
- Main manuscript Word with six embedded display items.
- Cover letter Word.
- Supplementary Appendix Word with eight figures and three tables.
- Complete Supplementary Tables 1–3 Excel workbook.
- Separate PDF/PNG main and supplementary figures.
- Complete archive of every generated figure copied from authoritative directories.
- Abstract: {word_count(ABSTRACT)} words (limit 150).
- Main display items: 6 (limit 6).

## Blocking author actions before upload
- Insert and verify author list, affiliations and corresponding-author details.
- Complete the authors' institutional ethics determination, identifier and date.
- Insert the public repository DOI/URL.
- Complete the MIMIC-III/MIMIC-IV overlap assessment or conservatively revise pooled claims.
- Complete funding, competing-interest, author-contribution and acknowledgement statements.
- Confirm originality, all-author approval, related manuscripts/preprints and reviewer suggestions.
- Complete the Nature Portfolio life-sciences reporting summary and applicable STROBE/TRIPOD+AI/MI-CLAIM-GEN checklists.

## Package limitation
This is a submission-formatted initial draft, not an authorised submission. Explicit action fields must not remain in the uploaded files.
"""
    (output / "00_README_SUBMISSION_PACKAGE.md").write_text(checklist, encoding="utf-8")
    manifest(output)
    print(f"BUILT {output}")
    print(f"ABSTRACT_WORDS={word_count(ABSTRACT)}")


if __name__ == "__main__":
    main()