# PedAKI-CARE Graph

**PedAKI-CARE** is the **Pediatric Acute Kidney Injury Contextual Applicability and Retrievable Evidence Graph**. It represents source provenance, population and care-setting context as explicit graph structure for pediatric AKI evidence retrieval.

The graph is developed within the PedAKI-Graphica research framework. Its retrieval study combines hybrid textual relevance with graph-derived provenance and population–setting applicability priors; the public browser exposes the graph rather than a clinical question-answering system.

Interactive knowledge-graph browser: **[https://sunyq2022.github.io/PedAKIGraphica/](https://sunyq2022.github.io/PedAKIGraphica/)**

The current public browser presents an aggregated, source-linked PedAKI-CARE view derived from 11,252 contract-validated evidence
assertions across 3,115 source records. The underlying study governed 30,094 usable abstracts and maintains guideline,
empirical-literature and terminology layers as distinct evidence roles. The browser highlights clinical concepts,
populations, care settings, controlled relations and a non-interpretive guideline index.

This is a development-stage computational research resource. Contract validation confirms structural and source-window
consistency; it does not constitute expert review, clinical validation, causal evidence or clinical decision support.
The public payload excludes abstract passages, full text, guideline passages, local paths, model raw outputs and
patient-level data.

## Team identity

This resource is developed by the **Children’s Health Data Science Team, Children’s Hospital of Nanjing Medical University**
(南京医科大学附属儿童医院儿童健康数据科学团队).

- **NCH-CHDS**: institutional team mark — Nanjing Children’s Hospital · Children’s Health Data Science.
- **CHDS**: compact signature — Children’s Health Data Science.
- **EVID-KG**: resource mark — evidence knowledge graph.

The browser uses an inline three-node emblem to represent children’s health, data, and science. It is a team/resource
identity mark, not a clinical validation or diagnostic symbol.

## Public-release contents

- `docs/`: self-contained GitHub Pages browser at the repository root and a static fallback network figure.
- `branding/`: local editable SVG identity assets for the team emblem, abbreviation, and full-name lockup.
- `data/public_release_status.json`: machine-readable scope and scientific boundaries.
- `research/PEDAKI_CURRENT_RELEASE/data/public_edges.csv`: aggregated public edge table.
- `research/PEDAKI_CURRENT_RELEASE/`: deterministic preparation and browser-payload builders.

## Rebuild the browser payload

```text
python research/PEDAKI_CURRENT_RELEASE/build_public_graph.py
```

The repository license and immutable archive/DOI remain author-controlled release actions. Public visibility alone
does not grant a reuse license.
