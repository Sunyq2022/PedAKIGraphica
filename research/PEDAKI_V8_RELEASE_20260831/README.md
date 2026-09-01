# PedAKI-Graphica V8 public release package

This directory is the public, source-linked release for the PedAKI-Graphica V8 computational evidence-resource study.

## Interactive knowledge-graph browser

The browser is served from `docs/index.html` and uses only local static assets. The stable entry point is the repository
Pages root URL.

## Scope and boundary

The browser contains the 225-relation pediatric AKI evidence graph among 34 entities. The relation table
contains identifiers, typed endpoints, predicates, evidence-layer metadata, source identifiers, population/domain
flags, split membership, and review confidence. It deliberately excludes full-text passages, full source files, and
patient-level data.

The graph preserves source-linked provenance and review metadata for evidence navigation and retrieval-method research.
It must not be interpreted as a causal model or clinical decision-support system.

## Reproduce the browser data

From the repository root:

```text
python research/PEDAKI_V8_RELEASE_20260831/build_public_graph.py
```

The script converts `data/relations_public.csv` into `docs/graph.json` using only the Python standard library.

## Included files

- `data/relations_public.csv`: public relation metadata without evidence-window text.
- `data/release_metadata.json`: release counts, source boundary, and checksums.
- `build_public_graph.py`: deterministic graph-data builder.
- `docs/index.html`: self-contained interactive browser.
- `docs/graph.json`: browser data generated from the public relation table.
- `docs/knowledge_graph_static.svg`: static fallback network figure from the current visual output.

## Licensing

The repository license and immutable archive/DOI remain author-controlled release actions. Until they are approved,
the files should be treated as an author-review release and not redistributed under an inferred license.
