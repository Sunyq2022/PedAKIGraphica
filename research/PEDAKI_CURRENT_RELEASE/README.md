# Public graph release workflow

This directory produces the source-text-free payload used by the GitHub Pages browser.

1. `prepare_public_release.py` aggregates governed evidence assertions into a compact public edge table. It retains identifiers and counts but excludes abstract/full-text snippets and local file paths.
2. `build_public_graph.py` deterministically rebuilds `docs/graph.json` and `docs/knowledge_graph_static.svg` from the public edge table.

The public graph is a computational research resource. Contract validation means that the original model output passed structural, source-identifier and verbatim-snippet checks in the private research workspace; it does not imply expert or clinical validation.
