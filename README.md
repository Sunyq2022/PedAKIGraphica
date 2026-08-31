# PedAKIGraphica

Public release package for the PedAKI-Graphica computational evidence-resource study.

Interactive knowledge-graph browser: **[open the graph browser](./docs/)**

The browser presents the 225-relation review-adjudicated construction graph among 34 entities. This is a
machine-adjudicated intermediate graph; it is not a clinically validated knowledge base, does not represent
named-expert signed validation, and must not be interpreted as a causal model or clinical decision-support system.
The public payload excludes full-text passages, full source files, and patient-level data.

## Contents

- `docs/`: self-contained GitHub Pages browser at the repository root and a static fallback network figure.
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
