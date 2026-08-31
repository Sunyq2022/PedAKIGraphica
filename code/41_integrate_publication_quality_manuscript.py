"""Integrate the publication-quality audit into the final manuscript package."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _00_v8_common import ROOT, dump_json

BASE = ROOT / "09_manuscript" / "06_automated_submission_package"
MANUSCRIPT = BASE / "01_Manuscript_PedAKI_Graphica_V8_AUTOMATED_COMPLETE.md"
AUDIT = ROOT / "04_results" / "18_final_audit"
BENCHMARK = ROOT / "04_results" / "19_multimethod_retrieval_benchmark"
QUALITY = AUDIT / "10_PUBLICATION_QUALITY_GATE.json"
EVAL = ROOT / "04_results" / "17_v8_llm_graph_comparison" / "06_evidence_grounded_auto_evaluation"
STATUS = ROOT / "07_status" / "01_V8_REBUILD_STATUS.json"
SUPPLEMENT = BASE / "12_supplementary_materials"
PUBLIC_REPOSITORY_URL = "https://github.com/Sunyq2022/PedAKIGraphica"
PUBLIC_GRAPH_URL = "https://sunyq2022.github.io/PedAKIGraphica/"


def replace(text: str, start: str, end: str, content: str) -> str:
    pattern = re.compile(rf"{re.escape(start)}.*?(?={re.escape(end)})", re.S)
    replacement = content.rstrip() + "\n\n"
    updated, count = pattern.subn(lambda _match: replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Section not found: {start}")
    return updated


def replace_any(text: str, starts: list[str], end: str, content: str) -> str:
    for start in starts:
        if start in text:
            return replace(text, start, end, content)
    raise RuntimeError(f"None of the section starts were found: {starts}")


def metric(quality: dict, method: str, name: str) -> str:
    item = quality["retrieval"]["methods"][method][name]
    return f"{item['estimate']:.3f} (95% CI {item['ci_95_lower']:.3f}–{item['ci_95_upper']:.3f})"


def paired(comparisons: pd.DataFrame, method: str, subset: str, name: str) -> pd.Series:
    return comparisons[
        comparisons["method"].eq(method)
        & comparisons["subset"].eq(subset)
        & comparisons["metric"].eq(name)
    ].iloc[0]


def q_text(value: float) -> str:
    """Format adjusted P values without reporting rounded zero."""
    return "<0.001" if value < 0.001 else f"={value:.3f}"


def integrate() -> None:
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    if quality["status"] != "PASS":
        raise RuntimeError("Publication-quality gate has not passed")
    sensitivity = pd.read_csv(AUDIT / "05_GRAPH_QUALITY_SENSITIVITY_METRICS.csv").set_index("scenario")
    comparisons = pd.read_csv(BENCHMARK / "03_PAIRED_METHOD_COMPARISONS_WITH_FDR.csv")
    evaluator = pd.read_csv(EVAL / "07_MACHINE_EVALUATOR_VALIDITY_AND_BIAS_AUDIT.csv").set_index("judge_model")
    text = MANUSCRIPT.read_text(encoding="utf-8")
    title = "PedAKI-Graphica: a traceable pediatric acute kidney injury evidence graph with leakage-controlled multimethod retrieval"
    text = re.sub(r"^# .+$", f"# {title}", text, count=1, flags=re.M)
    text = re.sub(r"^> \*\*(?:COMPLETE|PUBLICATION-QUALITY).*?\n\n", "", text, count=1, flags=re.M)
    text = re.sub(
        r"\*\*Article type:\*\*.*$",
        "**Article type:** Research article  ",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(r"^\*\*Version:\*\*.*\n", "", text, count=1, flags=re.M)
    abstract = f"""## Abstract

### Purpose
To develop an auditable evidence graph for pediatric acute kidney injury (AKI) and assess provenance coverage, structural sensitivity, and held-out evidence retrieval.

### Methods
A frozen snapshot of 17,667 PubMed Central records and 30 guidelines yielded 836 unique relation units while preserving the construction–held-out split. A two-stage computational review retained 225 construction and 136 held-out relations. We audited provenance, population scope, direction, and graph structure. Retrieval used 408 deterministic queries and 11,729 candidate documents after exclusion of 5,938 construction-supporting documents. BM25, frozen MiniLM, reciprocal-rank fusion, and two graph-expansion methods were compared using 5,000 relation-cluster bootstrap samples and false-discovery-rate correction.

### Results
All 18,851 accepted claim-level evidence rows supporting the 361 construction and held-out relation units were linked to frozen passages and sources. The construction graph comprised 225 aggregated SRO relations among 34 entities, supported by 5,945 sources. Flags identified 42 single-source relations, 143 without an explicit pediatric or neonatal tag, and 10 directional reverse conflicts. No null-model comparison remained significant after correction. BM25 achieved hit@5 {metric(quality, 'bm25', 'any_source_hit_at_5')} and MRR {metric(quality, 'bm25', 'mrr')}. Reciprocal-rank fusion had the highest MRR point estimate ({quality['retrieval']['methods']['bm25_minilm_rrf']['mrr']['estimate']:.3f}) but was not superior to BM25 after correction. Graph-expanded BM25 reduced MRR by {abs(paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').paired_mean_difference):.3f} (q{q_text(paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').bh_fdr_q)}).

### Conclusion
PedAKI-Graphica combines claim-level traceability with leakage-controlled retrieval evaluation. Unconstrained graph expansion introduced semantic drift and did not improve retrieval. The resource organizes literature evidence but is not clinically validated.

**Keywords:** pediatric acute kidney injury; knowledge graph; evidence provenance; information retrieval; reproducibility; graph audit
"""
    text = replace(text, "## Abstract", "## 1 Introduction", abstract)

    introduction = """## 1 Introduction

Acute kidney injury (AKI) is common in critically ill children and is associated with substantial morbidity and mortality [1–3]. Its recognition is complicated by age-dependent physiology, delayed serum-creatinine changes, incomplete urine-output data, and marked variation across neonatal, cardiac-surgical, oncological, infectious, and general intensive-care populations [19–22]. Although international criteria provide a shared staging framework, the relevant evidence remains dispersed across clinical settings, interventions, biomarkers, outcomes, and recovery trajectories.

Knowledge graphs can connect such evidence through typed entities and relations [4,5]. In a health-information setting, however, a graph is useful only if users can determine where each relation came from, what population it describes, and how confidently it was admitted. Provenance and FAIR data principles are therefore part of the scientific design, not ancillary metadata [6–8]. Without source-linked passages and explicit review boundaries, a compact relation can imply more certainty than the underlying literature supports.

An earlier implementation of PedAKI-Graphica established the feasibility of a disease-focused, provenance-aware relation graph, but did not enforce explicit semantic types for entities or predicate-specific domain and range constraints. The present study addresses these limitations and adds a stricter retrieval evaluation. We aimed to rebuild candidate generation around typed endpoints and separated evidence streams; construct an auditable, machine-adjudicated graph while preserving the construction–held-out split; examine topology and robustness under prespecified sensitivity analyses; and compare lexical, dense, hybrid, and graph-expanded retrieval in a candidate corpus from which all construction-supporting sources had been removed.

## 2 Related work and design rationale

Large biomedical knowledge graphs, including Hetionet, PrimeKG, and SPOKE, show how heterogeneous resources can be integrated at scale [11–13]. RDBridge and graphs derived from electronic health records illustrate complementary approaches based on literature mining and clinical data [14,15]. Disease-ontology work further shows that cross-resource integration depends on explicit semantic mappings rather than labels alone [23]. PedAKI-Graphica has a narrower purpose. Its basic unit is a source-linked subject–predicate–object (SRO) relation, and its main design priority is inspection of the evidence behind that relation rather than maximal graph size.

We retained BM25 as the principal transparent retrieval baseline because it is reproducible and exposes lexical failure modes [9]. Dense retrieval and retrieval-augmented generation can address broader semantic matching and answer-generation tasks [16–18,24,25], but their evaluation requires strict separation of construction evidence from held-out targets. The present benchmark therefore evaluates retrieval of supporting sources. It does not test clinical question answering, answer faithfulness, or bedside decision support.
"""
    text = replace(text, "## 1 Introduction", "## 3 Methods", introduction)

    methods_front = """## 3 Methods

### 3.1 Study design and data boundary
This computational evidence-resource study was not designed as a systematic review, meta-analysis, clinical trial, or prediction-model validation. PRISMA was cited only to clarify that distinction [10]. The literature boundary was a frozen PubMed Central snapshot created on August 29, 2026, which contained 17,667 records that met the upstream relevance, legal-access, XML, and metadata criteria when the snapshot was created. Records added later were outside the scope of the study. Patient-level databases, terminology releases, guidelines, and literature corpora were kept in separate data layers.

### 3.2 Guideline and literature processing
We audited 30 guideline documents, extracted 29, and quarantined one. The extracted guidelines contained 29,709 sentences, including 3,761 normative candidates, from which 421 typed relation candidates were generated. All 17,667 literature records were processed without extraction errors, yielding 2,533,818 passages and 37,867 typed candidates. Eligible passages were found in 17,289 documents; 9,829 documents contributed at least one candidate and 8,237 contributed at least one machine-eligible candidate. Documents with no eligible passage or candidate remained in the audit denominator.

### 3.3 Typed entity and relation schema
The schema assigns canonical identifiers and semantic types to both endpoints of each relation. Hereafter, “relation” and “SRO relation” refer to a unique normalized subject–predicate–object unit. A “candidate row” or “claim-level evidence row” is an individual source-passage assertion that may support such a unit; aggregation therefore permits many evidence rows to support one relation. Predicate-specific domain and range constraints prevent methodological or statistical attributes—including odds ratios, hazard ratios, confidence intervals, AUC values, thresholds, and model names—from being represented as graph entities. When available, these values are retained as attributes of the study result. Evidence was separated into four streams: `guideline_normative`, `literature_observational`, `study_result_attributes`, and `draft_guidance`. Draft guidance could not be promoted to finalized normative guidance.

### 3.4 Candidate gates and split preservation
A candidate relation was eligible only if it had typed endpoints, satisfied the predicate domain and range, contained a local relation cue, was directly supported by the evidence window, and was not negated or otherwise non-assertive. Validation identified no blocking schema defect among machine-eligible rows. The review set contained 23,904 candidate rows: 18,956 construction rows, 4,918 source-document-held-out rows, and 30 draft-context rows. Aggregation was performed separately within each split, producing 836 unique relation units without relation leakage.

### 3.5 Computational screening and designated review opinion
Qwen3 32B and DeepSeek-R1 32B independently screened all 836 relation units. For each unit, the input packet contained the normalized subject, predicate, and object with semantic types; split and evidence stream; population and clinical-context labels; claim and independent-source counts; source identifiers and titles; and up to three representative evidence windows. The models returned an overall disposition (`accept`, `revise`, `reject`, or `uncertain`), confidence, error category, short rationale, and component judgments for direct entailment, direction, predicate, entities, population/context, and treatment of statistics as attributes.

GLM-4.7 Flash adjudicated 744 units when the first two screens differed in overall disposition or any component judgment, either screen proposed revision, or either confidence was below 0.85; the remaining 92 units retained the concordant disposition. The 744-unit packet then underwent a designated second computational review against the supplied evidence, which changed 128 of the initial GLM dispositions. Combining those reviewed units with the 92 concordant units yielded 368 accept, 298 revise, 168 reject, and two uncertain decisions. `Revise` denoted a potentially recoverable SRO defect, such as an incorrect endpoint, direction, predicate, or population scope, rather than acceptance with minor wording changes. Because exact corrected subject, predicate, and object values were unavailable, revised units were excluded rather than rewritten automatically. The final layers comprised 225 construction relations, 136 accepted held-out relations kept in isolation, seven draft-context relations, and 468 excluded or revision-pending relations. These decisions constitute a designated computational review opinion, not signed validation by named experts.
"""
    text = replace(text, "## 3 Methods", "### 3.6 Claim-level provenance and edge-quality audit", methods_front)

    methods_quality = r"""### 3.6 Claim-level provenance and edge-quality audit
For every accepted construction or held-out relation, we resolved the frozen supporting candidate identifiers to a claim-level ledger. The 18,851 accepted claim-level evidence rows comprised 15,077 construction rows and 3,774 held-out rows supporting 225 and 136 relation units, respectively. Each row records the source and passage identifiers, evidence text, population, clinical context, polarity, uncertainty, section, and available quantitative attributes. Release required complete candidate resolution, confirmation that every source belonged to the frozen registry, non-empty evidence text, and exclusion of negated or non-assertive claims.

Construction edges were annotated with descriptive quality flags: single-source support, review confidence below 0.90, no explicit pediatric or neonatal population tag, adult-only scope, use of the broad `associated_with` predicate, a same-predicate reverse relation, and normative-guideline origin. These flags describe the evidence boundary; they do not validate an edge. Directional reverse pairs remained in the ledger but were excluded from directed-path interpretation. Rather than deriving a post hoc "gold" subgraph, we repeated the topology analysis in prespecified layers based on explicit pediatric or neonatal scope, support from multiple sources, review confidence of at least 0.90, exclusion of directional conflicts, exclusion of broad associations, and the intersection of pediatric scope, multiple sources, and high confidence.

### 3.7 Graph construction, null models, and robustness
Each unique SRO was retained as a directed relation. For descriptive topology, we used a simple undirected projection so that multiple predicates between the same pair of entities did not dominate density or path-based measures. For relation $e$, evidence weight was

$$
w_e = \alpha_{\ell(e)}\log(1+n_e),
$$

where $n_e$ is the number of independent supporting sources and $\alpha_{\ell(e)}$ is the prespecified evidence-layer weight: N1=5, N2=4, E1=3, E2=2, E3=1, and D=0.5. For an undirected entity pair $(u,v)$, parallel SRO weights were summed,

$$
W_{uv}=\sum_{e:\{s_e,o_e\}=\{u,v\}} w_e,
$$

whereas the unweighted adjacency used for density and path metrics was $A_{uv}=\mathbb{1}(W_{uv}>0)$. We generated 1,000 degree-preserving double-edge-swap realizations. For each metric, the empirical two-sided P value was computed from its position in the null distribution, and the 13 P values were adjusted by the Benjamini–Hochberg procedure [28]. Node-removal and evidence-ablation analyses were used to assess structural sensitivity, not biological mechanism or clinical validity."""
    text = replace_any(
        text,
        ["### 3.6 Graph construction and evidence weighting", "### 3.6 Claim-level provenance and edge-quality audit"],
        "### 3.8 Internal known-item BM25 diagnostic",
        methods_quality,
    )
    retrieval_methods = r"""### 3.11 Leakage-controlled multimethod held-out retrieval
The 136 held-out relations were supported by 1,474 relevant PMC documents, none of which supported the construction graph. We also removed every construction-supporting PMC document from the candidate corpus. Excluding these 5,938 documents from the 17,667-record snapshot left 11,729 candidates: 1,474 relevant documents and 10,255 hard-negative or other documents. Fifty-one held-out SRO tuples did not occur in the construction graph. Canonical, paraphrased, and contextualized formulations produced 408 deterministic queries.

Five prespecified methods were evaluated on the same queries and relevance sets: BM25; dense retrieval with frozen `all-MiniLM-L6-v2`; BM25–MiniLM reciprocal-rank fusion (RRF); BM25 after deterministic addition of the six highest-support construction-graph neighbors of the relation endpoints; and RRF of graph-expanded BM25 with MiniLM. For query $q$ and document $d$, BM25 used $k_1=1.5$ and $b=0.75$ [9]:

$$
\operatorname{BM25}(q,d)=\sum_{t\in q}\log\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right)
\frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1\left(1-b+b|d|/\overline{|d|}\right)}.
$$

MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` was loaded offline and was not fine-tuned on the benchmark. Embeddings were L2-normalized, so dense similarity was the inner product $s_{\mathrm{dense}}(q,d)=z_q^\top z_d$ [16]. RRF combined ranked lists as

$$
s_{\mathrm{RRF}}(d)=\sum_m\frac{1}{60+r_m(d)},
$$

where $r_m(d)$ is the rank of document $d$ under method $m$ [26]. Graph expansion used construction adjacency only and had no access to held-out evidence text.

For relevant-source set $R_q$ and the top-$k$ ranked set $D_q^{(k)}$, any-source hit and source recall were

$$
\operatorname{Hit@}k(q)=\mathbb{1}\left(R_q\cap D_q^{(k)}\neq\varnothing\right),\qquad
\operatorname{Recall@}k(q)=\frac{|R_q\cap D_q^{(k)}|}{|R_q|}.
$$

Macro recall averaged query-level recall; micro recall pooled retrieved and relevant source counts. MRR was $Q^{-1}\sum_q 1/r_q$, with zero assigned when no relevant source was retrieved. Binary-relevance nDCG@10 used logarithmic rank discount and ideal normalization [27]. Confidence intervals and paired method differences were estimated from 5,000 bootstrap samples clustered by held-out relation rather than query, preserving dependence among the three formulations of each relation. Paired P values were adjusted by the Benjamini–Hochberg procedure across method, subset, and metric contrasts [28].

### 3.12 Secondary automated-evaluator calibration experiment
A frozen corpus of 300 generation units from three models and four LLM/RAG arms was retained as a secondary methods experiment. Before comparing arms, we assessed the three models as evidence judges using locked perturbation anchors. None met all prespecified validity criteria; consequently, no formal claim-F1 comparison or Figure S6 was released. The experiment therefore provides no estimate of Graph-RAG effectiveness."""
    text = replace_any(
        text,
        ["### 3.11 Automated source-isolated held-out evaluation", "### 3.11 Leakage-controlled multimethod held-out retrieval"],
        "## 4 Results",
        retrieval_methods,
    )

    results = f"""## 4 Results

### 4.1 Corpus, relation layers, and complete provenance
The frozen corpus comprised 17,667 documents and 2,533,818 passages. Literature and guideline processing generated 38,288 typed candidate rows, which aggregated to 836 unique SRO relation units without crossing the construction–held-out split. Computational review retained 225 construction relations and 136 held-out relations. Seven draft-context relations remained separate, and 468 units were excluded or left revision-pending. The 361 accepted construction and held-out units were supported by 18,851 claim-level evidence rows: 15,077 construction and 3,774 held-out rows. Every row resolved to a frozen passage and source; none had empty evidence text, a negation flag, or a failed assertive-statement gate.

### 4.2 Edge quality and evidence scope
The construction graph connected 34 entities through 225 relations supported by 5,945 unique sources (Fig. 1). Forty-two relations had a single source, 29 had review confidence below 0.90, and 143 did not carry an explicit pediatric or neonatal population tag. Six relations were adult-only, 117 used the broad `associated_with` predicate, and six originated from normative guideline statements. Ten directional relations had a same-predicate reverse counterpart and were withheld from directed-path interpretation; a further 22 broad-association rows represented both orientations of the same endpoints. The 10 most frequent sources accounted for {quality['edge_quality']['top_10_source_incidence_share']*100:.2f}% of source–edge incidences.

### 4.3 Structural description and quality sensitivity
After projection, the 225 directed relations formed 141 undirected node pairs in one connected component (Fig. 2). Acute kidney injury, mortality, sepsis, severe AKI, and serum creatinine had the highest degree. Excluding directional-conflict rows did not alter the undirected topology because their reverse counterparts preserved the same node pairs. Degree-rank correlations with the full graph were {sensitivity.loc['multi_source','degree_rank_spearman_vs_all']:.3f} after single-source relations were excluded and {sensitivity.loc['review_confidence_ge_0_90','degree_rank_spearman_vs_all']:.3f} after relations with confidence below 0.90 were excluded. The explicit pediatric or neonatal layer retained 82 relations and 25 active nodes, with a rank correlation of {sensitivity.loc['pediatric_scope_explicit','degree_rank_spearman_vs_all']:.3f}; the pediatric, multi-source, high-confidence intersection retained 70 relations and 23 active nodes.

### 4.4 Exploratory null-model and robustness analysis
Four of the 13 degree-preserving null comparisons had unadjusted empirical P values below 0.05, but none remained significant after Benjamini–Hochberg correction; the minimum q value was {quality['null_model']['minimum_fdr_q']:.3f} (Fig. 3). Null-model deviations are therefore presented as exploratory standardized differences rather than confirmed graph properties. Targeted removal by degree, betweenness, or articulation status disrupted the largest component more rapidly than random removal, whereas the multi-source and confidence-filtered layers retained the principal navigation hubs.

### 4.5 Leakage-controlled multimethod retrieval
The candidate corpus contained 11,729 documents after every construction-supporting PMC source had been excluded; 1,474 were relevant held-out sources and 10,255 were hard-negative or other documents (Fig. 4). BM25 achieved hit@5 {metric(quality, 'bm25', 'any_source_hit_at_5')}, macro source recall@5 {metric(quality, 'bm25', 'macro_source_recall_at_5')}, micro source recall@5 {metric(quality, 'bm25', 'micro_source_recall_at_5')}, MRR {metric(quality, 'bm25', 'mrr')}, and nDCG@10 {metric(quality, 'bm25', 'ndcg_at_10')}.

BM25–MiniLM RRF produced the highest point estimates for hit@5 ({quality['retrieval']['methods']['bm25_minilm_rrf']['any_source_hit_at_5']['estimate']:.3f}) and MRR ({quality['retrieval']['methods']['bm25_minilm_rrf']['mrr']['estimate']:.3f}). Its paired MRR difference from BM25 was {paired(comparisons, 'bm25_minilm_rrf', 'all_relations', 'mrr').paired_mean_difference:+.3f} (95% CI {paired(comparisons, 'bm25_minilm_rrf', 'all_relations', 'mrr').ci_95_lower:+.3f} to {paired(comparisons, 'bm25_minilm_rrf', 'all_relations', 'mrr').ci_95_upper:+.3f}; q={paired(comparisons, 'bm25_minilm_rrf', 'all_relations', 'mrr').bh_fdr_q:.3f}), which did not establish superiority after multiplicity correction. MiniLM alone was inferior to BM25 for hit@5, MRR, and nDCG@10 after correction.

Graph-expanded BM25 performed worse than BM25. The paired MRR difference was {paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').paired_mean_difference:+.3f} overall (95% CI {paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').ci_95_lower:+.3f} to {paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').ci_95_upper:+.3f}; q{q_text(paired(comparisons, 'graph_expanded_bm25', 'all_relations', 'mrr').bh_fdr_q)}) and {paired(comparisons, 'graph_expanded_bm25', 'strict_novel_sro', 'mrr').paired_mean_difference:+.3f} in the 51 strict-novel SROs (q{q_text(paired(comparisons, 'graph_expanded_bm25', 'strict_novel_sro', 'mrr').bh_fdr_q)}). No graph-expanded method showed FDR-supported superiority.

### 4.6 Secondary evaluator-calibration result
All three models returned parseable JSON for the locked audit and had perfect format-pair consistency. Macro-F1 was {evaluator.loc['glm-4.7-flash:q4_K_M','macro_f1']:.3f} for GLM-4.7 Flash, {evaluator.loc['qwen3:32b','macro_f1']:.3f} for Qwen3 32B, and {evaluator.loc['deepseek-r1:32b','macro_f1']:.3f} for DeepSeek-R1 32B. Each model failed at least one prespecified validity criterion; no formal Graph-RAG arm comparison was released.

### 4.7 Figure and result provenance
Each main figure is linked to editable source data, its generating function, and SHA-256 hashes. The Supplementary Appendix presents additional evaluation figures and compact summary tables derived from the frozen quality, sensitivity, retrieval, and evaluator-audit outputs. Row-level ledgers remain available separately for audit and reproducibility."""
    text = replace(text, "## 4 Results", "## 5 Discussion", results)

    discussion = """## 5 Discussion

### 5.1 Principal findings
PedAKI-Graphica was designed as an evidence resource rather than a large general-purpose biomedical graph. Its main contribution is the continuity of the evidence chain: each aggregated SRO relation can be followed to its accepted claim-level evidence rows and then to the frozen passages and sources from which they were derived. This design makes the graph inspectable at the point where most literature-derived graphs become opaque—between a compact edge and the text that justified it. The release also separates relation admission from evidence quality. A relation can be traceable yet remain single-source, population-unspecified, broad in predicate meaning, or affected by a directional conflict. Treating these properties as explicit annotations rather than collapsing them into one confidence label preserves information needed for later review.

The results also show why a pediatric AKI evidence graph cannot be interpreted as a uniformly pediatric knowledge base. Only 82 of 225 relations carried an explicit pediatric or neonatal population tag, although many population-unspecified relations originated from literature relevant to pediatric AKI. Pediatric AKI spans neonatal physiology, critical care, cardiac surgery, nephrotoxic exposure, biomarkers, kidney replacement therapy, recovery, and long-term outcomes [19–22]. A useful evidence interface must therefore retain cross-age evidence while making its population boundary visible. The smaller explicitly pediatric sensitivity layer is not a defect to be hidden; it identifies where additional pediatric curation and primary research are most needed.

### 5.2 Semantic scope and graph interpretation
Semantic typing was central to the reconstruction. Earlier relation extraction could allow statistical constructs, thresholds, or model descriptors to behave like biomedical entities. Predicate-specific domain and range constraints now keep such quantities as attributes of study results. This distinction matters because ontology alignment and entity normalization determine which observations can legitimately be connected across sources [6,23]. It also limits a common failure mode of literature graphs: syntactically plausible edges whose endpoints do not support the intended biomedical interpretation.

The network view should nevertheless not be mistaken for a causal model. The broad `associated_with` predicate accounted for 117 relations, and reverse directional counterparts were present for 10 directional relations. These patterns may reflect heterogeneous study designs, population differences, temporal ambiguity, or extraction uncertainty. Directed path analysis should therefore use a restricted edge set and return to the underlying passages. The proposed interactive GitHub Pages release is particularly useful for this purpose: users can inspect neighborhoods, filters, quality flags, and evidence windows without embedding a static, text-heavy graph browser in the Supplementary Appendix.

### 5.3 Structural findings and robustness
Several familiar AKI concepts occupied central positions, but centrality alone does not establish biological importance. Highly connected concepts can arise because they are common outcomes, broad syndromic labels, or frequent indexing terms. The stability analyses provide a more informative interpretation. Degree rankings changed little after single-source or lower-confidence relations were removed, indicating that major navigation hubs were not created solely by the weakest evidence. In contrast, restricting the graph to explicit pediatric scope or to the pediatric, multi-source, high-confidence intersection reduced both coverage and component retention. The graph therefore has a stable navigational backbone, but the amount of evidence satisfying all desirable criteria is substantially smaller than the full network suggests.

The null-model results further constrain interpretation. Four topology measures had unadjusted empirical P values below 0.05, but none remained significant after correction across 13 comparisons. The degree-preserving null model asks whether the observed simple topology differs from graphs with the same degree sequence; it does not preserve entity type, predicate, direction, domain, or evidence weight. This is a stringent comparison because it conditions on the hub structure already created by recurring concepts such as acute kidney injury and mortality. Failure to reject these null comparisons does not show that the evidence is random, nor does it prove that degree sequence fully explains the graph. It means that the present sample and metrics did not establish higher-order organization beyond degree after multiplicity correction. Null models that additionally preserve semantic types or edge attributes would be required before making more specific claims about disease organization or semantic modularity.

### 5.4 Retrieval findings and the limits of graph expansion
The retrieval benchmark was deliberately stricter than a random document split. Every document that supported the construction graph was removed from the candidate corpus, so methods could not recover targets by retrieving the evidence used to build their expansion structure. The remaining 11,729 documents included 10,255 hard-negative or other documents. Under this setting, BM25 remained a strong and transparent baseline. Its hit@5 of approximately 21% was materially higher than its source recall, showing that retrieving one supporting document is much easier than recovering the distributed evidence base for a relation.

Reciprocal-rank fusion produced the highest MRR point estimate but did not outperform BM25 after paired multiplicity correction. This result is consistent with the complementary strengths of lexical and dense retrieval without implying that fusion will always improve a biomedical task. Dense retrieval used a general-purpose MiniLM encoder rather than a biomedical encoder, and the deterministic relation-label queries may favor lexical overlap. The supplementary stratified analyses show how performance varies across canonical, paraphrased, contextualized, seen-SRO/new-source, and strict-novel queries, which is more informative than a single pooled estimate.

The negative graph-expansion result is methodologically important and warrants separate emphasis for graph-augmented retrieval and Graph-RAG research. Expansion selected high-support neighbors of the relation endpoints but did not condition on predicate, edge direction, population, or query intent. The added terms therefore increased semantic breadth without guaranteeing relevance. In pediatric AKI, a central endpoint such as AKI can connect to mortality, sepsis, biomarkers, fluid balance, cardiac surgery, and recovery; indiscriminate expansion can move the query away from the specific evidence target. Biomedical knowledge-graph prompting and clinical RAG studies have reported potential gains from structured retrieval [18,24,25], but these gains depend on the retrieval source, task, filtering policy, and evaluation design. Our findings show that simple adjacency expansion cannot substitute for semantic relevance estimation and argue for predicate-aware, direction-aware, population-aware, or learned expansion.

### 5.5 Evaluation validity and reproducibility
The stopped secondary LLM experiment illustrates another boundary. All three candidate judges produced syntactically valid outputs, yet each failed at least one prespecified validity gate. Parseability and apparent consistency were therefore insufficient evidence that the models could adjudicate claim correctness. Releasing a Graph-RAG comparison after this failure would have replaced an unvalidated clinical claim with an unvalidated evaluation claim. The decision to withhold that contrast is part of the study result, not an incomplete analysis.

Reproducibility was addressed through frozen registries, split-preserving aggregation, source-exclusion checks, fixed model revisions, deterministic seeds, clustered bootstrap sampling, multiplicity correction, editable figure data, and file hashes. The graph browser, code, relation tables, and summary outputs should be released as versioned GitHub assets, with the interactive graph served through GitHub Pages and an immutable archive deposited separately. This division keeps the manuscript and Supplementary Appendix focused on scientific argument while preserving the full audit trail online.

### 5.6 Strengths, limitations, and next steps
Strengths include complete claim-level provenance, explicit quality and population flags, a candidate corpus with zero construction-source overlap, comparison of lexical, dense, hybrid, and graph-expanded retrieval, relation-cluster uncertainty estimates, and prespecified control of multiplicity. The study also reports negative findings rather than promoting graph structure or Graph-RAG without adequate support.

Several limitations remain. Relation admission was machine-adjudicated rather than signed off by named clinical experts. The 298 revision dispositions—35.6% of the 836 reviewed units—also show that automated extraction frequently produced potentially recoverable but not release-ready SROs; excluding rather than silently rewriting them protected precision at the cost of coverage. Source identifiers do not guarantee independence between publications, and population scope was often absent from the evidence window. The ontology and relation lexicon were deliberately compact. Queries were deterministic rather than clinician-authored. The general-domain MiniLM encoder was not biomedical-domain-tuned and may therefore underrepresent the potential of dense retrieval for specialized terminology; domain-adapted encoders could yield different method rankings on this benchmark. No external temporal corpus was available. The null model preserved degree but not semantic type or edge attributes. Full source text may not be redistributable under all licenses. Finally, the interactive graph will improve inspection but will not convert the resource into a clinical decision-support system.

Next steps should prioritize targeted expert review of high-centrality or high-conflict relations, clinician-authored retrieval questions, biomedical encoders, temporal validation on newly published literature, and predicate-aware graph expansion. Any clinical application would additionally require population-specific validation, governance for literature updates, explicit handling of contradictory evidence, and task-specific safety assessment."""
    text = replace(text, "## 5 Discussion", "## 6 Conclusion", discussion)
    conclusion = """## 6 Conclusion
PedAKI-Graphica links 18,851 accepted claim-level evidence rows to frozen passages and sources and aggregates them into 225 construction and 136 held-out SRO relation units. In a five-method benchmark that excluded every construction-supporting document, lexical–dense fusion yielded the highest retrieval point estimate but was not superior to BM25 after multiplicity correction; unconstrained graph-neighbor expansion reduced MRR. The resource supports reproducible evidence navigation and retrieval-method development, but it should not be interpreted as clinically validated knowledge or as evidence that graph expansion necessarily improves retrieval.
"""
    text = replace(text, "## 6 Conclusion", "## Statements and Declarations", conclusion)
    data_availability = f"""### Data availability
The V8 public release is available at [{PUBLIC_REPOSITORY_URL}]({PUBLIC_REPOSITORY_URL}). The interactive knowledge-graph browser is available at [{PUBLIC_GRAPH_URL}]({PUBLIC_GRAPH_URL}); this URL opens the graph visualization directly. The release contains the review-adjudicated construction relation metadata, frozen manifests, quality audits, retrieval outputs, figure source data, and hashes. Full source text, full-text passages, and patient-level data are excluded because redistribution is governed by source-specific licenses and the study did not analyze patient-level data. Repository licensing, immutable versioning, and DOI assignment remain author-controlled release actions.
"""
    text = replace(text, "### Data availability", "### Code availability", data_availability)
    code_availability = f"""### Code availability
The unique publication entry point is `03_code/42_run_final_publication_build.py`; `requirements-publication.txt` records the tested Python environment. The pipeline rebuilds the quality layer, figures, manuscript, Word/PDF package, governance manifests, and tests from frozen inputs. V8 code and public data are released at [{PUBLIC_REPOSITORY_URL}]({PUBLIC_REPOSITORY_URL}); the interactive graph is served at [{PUBLIC_GRAPH_URL}]({PUBLIC_GRAPH_URL}). Immutable versioning and DOI assignment remain author-controlled release actions.
"""
    text = replace(text, "### Code availability", "### Author contributions", code_availability)
    supplement = """## Supplementary information

The Supplementary Appendix provides additional evaluation figures and compact summary tables for edge quality, graph sensitivity, stratified retrieval performance, paired method effects, and automated-evaluator validity. Row-level provenance and benchmark data are supplied separately as machine-readable research files rather than reproduced in the appendix."""
    text = replace(text, "## Supplementary information", "## References", supplement)
    references = """## References

1. Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group. KDIGO clinical practice guideline for acute kidney injury. Kidney Int Suppl. 2012;2(1):1–138. https://kdigo.org/guidelines/acute-kidney-injury/
2. Kaddourah A, Basu RK, Bagshaw SM, Goldstein SL, for the AWARE Investigators. Epidemiology of acute kidney injury in critically ill children and young adults. N Engl J Med. 2017;376:11–20. https://doi.org/10.1056/NEJMoa1611391
3. Jetton JG, Boohaker LJ, Sethi SK, et al. Incidence and outcomes of neonatal acute kidney injury (AWAKEN): a multicentre, multinational, observational cohort study. Lancet Child Adolesc Health. 2017;1(3):184–194. https://doi.org/10.1016/S2352-4642(17)30069-X
4. Hogan A, Blomqvist E, Cochez M, et al. Knowledge graphs. ACM Comput Surv. 2022;54(4):Article 71. https://doi.org/10.1145/3447772
5. Papadakis E, Baryannis G, Batsakis S, et al. ADHD-KG: a knowledge graph of attention deficit hyperactivity disorder. Health Inf Sci Syst. 2023;11:52. https://doi.org/10.1007/s13755-023-00253-8
6. Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Res. 2004;32(Database issue):D267–D270. https://doi.org/10.1093/nar/gkh061
7. Wilkinson MD, Dumontier M, Aalbersberg IJ, et al. The FAIR guiding principles for scientific data management and stewardship. Sci Data. 2016;3:160018. https://doi.org/10.1038/sdata.2016.18
8. Moreau L, Missier P, editors. PROV-DM: The PROV Data Model. W3C Recommendation; 30 April 2013. https://www.w3.org/TR/prov-dm/
9. Robertson SE, Walker S, Jones S, Hancock-Beaulieu MM, Gatford M. Okapi at TREC-3. In: Overview of the Third Text REtrieval Conference (TREC-3). NIST Special Publication 500-226; 1995. p. 109–126. https://trec.nist.gov/pubs/trec3/t3_proceedings.html
10. Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ. 2021;372:n71. https://doi.org/10.1136/bmj.n71
11. Himmelstein DS, Lizee A, Hessler C, et al. Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife. 2017;6:e26726. https://doi.org/10.7554/eLife.26726
12. Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine. Sci Data. 2023;10:67. https://doi.org/10.1038/s41597-023-01960-3
13. Morris JH, Soman K, Akbas RE, et al. The scalable precision medicine open knowledge engine (SPOKE): a massive knowledge graph of biomedical information. Bioinformatics. 2023;39(2):btad080. https://doi.org/10.1093/bioinformatics/btad080
14. Xing H, Zhang D, Cai P, et al. RDBridge: a knowledge graph of rare diseases based on large-scale text mining. Bioinformatics. 2023;39(7):btad440. https://doi.org/10.1093/bioinformatics/btad440
15. Rotmensch M, Halpern Y, Tlimat A, Horng S, Sontag D. Learning a health knowledge graph from electronic medical records. Sci Rep. 2017;7:5994. https://doi.org/10.1038/s41598-017-05778-z
16. Karpukhin V, Oguz B, Min S, et al. Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics; 2020. p. 6769–6781. https://doi.org/10.18653/v1/2020.emnlp-main.550
17. Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. Adv Neural Inf Process Syst. 2020;33:9459–9474.
18. Soman K, Rose PW, Morris JH, et al. Biomedical knowledge graph-optimized prompt generation for large language models. Bioinformatics. 2024;40(9):btae560. https://doi.org/10.1093/bioinformatics/btae560
19. Starr MC, Charlton JR, Guillet R, et al. Advances in neonatal acute kidney injury. Pediatrics. 2021;148(5):e2021051220. https://doi.org/10.1542/peds.2021-051220
20. Selewski DT, Charlton JR, Jetton JG, et al. Neonatal acute kidney injury. Pediatrics. 2015;136(2):e463–e473. https://doi.org/10.1542/peds.2014-3819
21. Cleto-Yamane TL, Gomes CLR, Suassuna JHR, Nogueira PK. Acute kidney injury epidemiology in pediatrics. J Bras Nefrol. 2019;41(2):275–283. https://doi.org/10.1590/2175-8239-JBN-2018-0127
22. Dong J, Feng T, Thapa-Chhetry B, et al. Machine learning model for early prediction of acute kidney injury in pediatric critical care. Crit Care. 2021;25:288. https://doi.org/10.1186/s13054-021-03724-0
23. Kurbatova N, Swiers R. Disease ontologies for knowledge graphs. BMC Bioinformatics. 2021;22:377. https://doi.org/10.1186/s12859-021-04173-w
24. Ge J, Sun S, Owens J, et al. Development of a liver disease-specific large language model chat interface using retrieval-augmented generation. Hepatology. 2024;80(5):1158–1168. https://doi.org/10.1097/HEP.0000000000000834
25. Liu S, McCoy AB, Wright A. Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines. J Am Med Inform Assoc. 2025;32(4):605–615. https://doi.org/10.1093/jamia/ocaf008
26. Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In: Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM; 2009. p. 758–759. https://doi.org/10.1145/1571941.1572114
27. Järvelin K, Kekäläinen J. Cumulated gain-based evaluation of IR techniques. ACM Trans Inf Syst. 2002;20(4):422–446. https://doi.org/10.1145/582415.582418
28. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Series B Stat Methodol. 1995;57(1):289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
"""
    text = replace(text, "## References", "## Figure legends", references)
    text = re.sub(
        r"\*\*Fig\. 4\..*?(?=\n\n## Submission authorization boundary|\Z)",
        "**Fig. 4. Leakage-controlled multimethod held-out source retrieval.** Panel a compares any-source hit@5 with 95% relation-cluster bootstrap intervals across BM25, frozen MiniLM dense retrieval, BM25–MiniLM reciprocal-rank fusion, graph-expanded BM25, and graph-expanded fusion. Panel b compares MRR for the same methods. Panel c shows paired MRR differences versus BM25 for all 136 held-out relations and the 51 strict-novel SRO subset; multiplicity-adjusted results are reported in Supplementary Table S4. Panel d shows MRR by deterministic query type for BM25, BM25–MiniLM fusion, and graph-expanded BM25. The 11,729-document candidate corpus excluded all 5,938 construction-supporting PMC documents. The benchmark evaluates source retrieval, not clinical question answering or Graph-RAG effectiveness.",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        "Evidence depth and source diversity in the V8 machine-adjudicated construction graph.",
        "Evidence depth and source diversity in the machine-adjudicated construction graph.",
    )
    text = text.replace(
        "all authors verify V8-specific roles",
        "all authors verify study-specific roles",
    )
    text = re.sub(r"\n\n## Submission authorization boundary\n\n.*$", "", text, flags=re.S)
    MANUSCRIPT.write_text(text, encoding="utf-8")


def supplements() -> None:
    if SUPPLEMENT.exists():
        shutil.rmtree(SUPPLEMENT)
    SUPPLEMENT.mkdir(parents=True)
    (SUPPLEMENT / "README.md").write_text(
        "# Machine-readable research release\n\n"
        "Row-level provenance, relation, query, calibration, atomic-claim, and evaluator-verdict files are not "
        "journal supplementary materials. They should be deposited with the code and interactive graph in the "
        "versioned GitHub research repository after licensing and author review.\n",
        encoding="utf-8",
    )


def update_status() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    status.update({
        "status": "PUBLICATION_QUALITY_COMPUTATIONAL_GATES_PASS_AUTHOR_ACTIONS_PENDING",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "publication_quality_gate_pass": True,
        "complete_claim_traceability_rows": 18851,
        "corrected_retrieval_metrics_released": True,
        "multimethod_retrieval_benchmark_complete": True,
        "retrieval_candidate_corpus_documents": 11729,
        "construction_candidate_corpus_overlap": 0,
        "graph_expansion_superiority_supported": False,
        "null_model_fdr_reported": True,
        "edge_quality_sensitivity_complete": True,
        "formal_graph_rag_contrast_released": False,
        "clinical_validation_claimed": False,
        "submission_authorized": False,
    })
    status.pop("automatic_trigger_restart_command", None)
    dump_json(STATUS, status)


def main() -> None:
    integrate(); supplements(); update_status()
    print(json.dumps({"status": "PUBLICATION_MANUSCRIPT_INTEGRATED", "quality_gate": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
