# PedAKIGraphica

Public release package for the PedAKI-Graphica computational evidence-resource study.

Interactive knowledge-graph browser: **[https://sunyq2022.github.io/PedAKIGraphica/](https://sunyq2022.github.io/PedAKIGraphica/)**

The browser presents the 225-relation review-adjudicated construction graph among 34 entities. This is a
machine-adjudicated intermediate graph; it is not a clinically validated knowledge base, does not represent
named-expert signed validation, and must not be interpreted as a causal model or clinical decision-support system.
The public payload excludes full-text passages, full source files, and patient-level data.

## Team identity

This resource is developed by the **Children’s Health Data Science Team, Children’s Hospital of Nanjing Medical University**
(南京医科大学附属儿童医院儿童健康数据科学团队).

- **NCH-CHDS**: institutional team mark — Nanjing Children’s Hospital · Children’s Health Data Science.
- **CHDS**: compact signature — Children’s Health Data Science.
- **EVID-KG**: resource mark — evidence knowledge graph.

The browser uses an inline three-node emblem to represent children’s health, data, and science. It is a team/resource
identity mark, not a clinical validation or diagnostic symbol.

## Contents

- `docs/`: self-contained GitHub Pages browser at the repository root and a static fallback network figure.
- `branding/`: local editable SVG identity assets for the team emblem, abbreviation, and full-name lockup.
- `data/`: public relation metadata, analysis tables, supplementary tables, and current release state.
- `figures/`: publication and supplementary figure assets for this release.
- `code/`: reproducibility and publication scripts for the current release line.
- `research/PEDAKI_V8_RELEASE_20260831/`: deterministic browser-payload builder.

## Rebuild the browser payload

```text
python research/PEDAKI_V8_RELEASE_20260831/build_public_graph.py
```

The repository license and immutable archive/DOI remain author-controlled release actions. Until author approval, do
not infer a reuse license from the public visibility of this repository.
