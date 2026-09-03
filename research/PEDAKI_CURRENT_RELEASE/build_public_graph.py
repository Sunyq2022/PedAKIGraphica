"""Build the GitHub Pages payload and a source-text-free static SVG fallback."""
from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).resolve().parent / "data" / "public_edges.csv"
METADATA = Path(__file__).resolve().parent / "data" / "release_metadata.json"
OUTPUT = ROOT / "docs" / "graph.json"
STATIC = ROOT / "docs" / "knowledge_graph_static.svg"

COLORS = {
    "clinical_concept": "#55B5A6", "population": "#7B61B5", "care_setting": "#E9825B",
    "relation": "#6F9FC5", "guideline": "#D3A23F", "guideline_topic": "#9EBB72",
}


def build() -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            weight = int(row["assertion_count"] or 0)
            for key in ("source", "target"):
                node = nodes.setdefault(row[key], {
                    "id": row[key], "label": row[f"{key}_label"], "type": row[f"{key}_type"],
                    "degree": 0, "weighted_degree": 0,
                })
                node["degree"] += 1
                node["weighted_degree"] += weight
            edges.append({
                "id": row["edge_id"], "source": row["source"], "target": row["target"],
                "predicate": row["predicate"], "layer": row["layer"],
                "assertion_count": weight, "source_count": int(row["source_count"] or 0),
                "source_ids": [value for value in row["source_ids"].split(";") if value],
                "populations": [value for value in row["population"].split(";") if value],
                "care_settings": [value for value in row["care_setting"].split(";") if value],
            })
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    payload = {
        **metadata,
        "scope": f"{len(nodes):,} public summary nodes and {len(edges):,} aggregated relations",
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": sorted(edges, key=lambda edge: edge["id"]),
        "facets": {
            "node_types": dict(Counter(node["type"] for node in nodes.values())),
            "layers": dict(Counter(edge["layer"] for edge in edges)),
            "predicates": dict(Counter(edge["predicate"] for edge in edges)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_static_svg(payload)
    return payload


def write_static_svg(payload: dict) -> None:
    ranked = sorted(payload["nodes"], key=lambda node: (-node["weighted_degree"], node["id"]))[:140]
    ids = {node["id"] for node in ranked}
    edges = [edge for edge in payload["edges"] if edge["source"] in ids and edge["target"] in ids][:500]
    width, height, cx, cy = 1400, 920, 700, 470
    positions = {}
    ordered_types = ["clinical_concept", "relation", "population", "care_setting", "guideline", "guideline_topic"]
    by_type = {kind: [node for node in ranked if node["type"] == kind] for kind in ordered_types}
    radii = {"clinical_concept": 390, "relation": 220, "population": 300, "care_setting": 330, "guideline": 120, "guideline_topic": 270}
    for kind_index, kind in enumerate(ordered_types):
        group = by_type[kind]
        for index, node in enumerate(group):
            angle = 2 * math.pi * (index / max(1, len(group))) + kind_index * 0.31
            radius = radii[kind] + 28 * ((index % 3) - 1)
            positions[node["id"]] = (cx + radius * math.cos(angle), cy + 0.78 * radius * math.sin(angle))
    aki = next((node for node in ranked if node["label"] == "Acute kidney injury"), None)
    if aki:
        positions[aki["id"]] = (cx, cy)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="920" viewBox="0 0 1400 920">',
        '<rect width="1400" height="920" fill="#0b1220"/>',
        '<text x="44" y="54" fill="#f4f8ff" font-family="Arial" font-size="30" font-weight="700">PedAKI-CARE evidence network</text>',
        '<text x="44" y="82" fill="#a9bad0" font-family="Arial" font-size="15">Context-aware retrieval of evidence; source text excluded</text>',
    ]
    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        opacity = min(0.58, 0.12 + math.log1p(edge["assertion_count"]) / 18)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#7594b5" stroke-opacity="{opacity:.3f}" stroke-width="1"/>')
    max_weight = max(node["weighted_degree"] for node in ranked) or 1
    for node in ranked:
        x, y = positions[node["id"]]
        radius = 4 + 18 * math.sqrt(node["weighted_degree"] / max_weight)
        color = COLORS.get(node["type"], "#b7c8da")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" stroke="#f6fbff" stroke-width="0.8"/>')
        if node["weighted_degree"] >= ranked[min(34, len(ranked) - 1)]["weighted_degree"]:
            parts.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" fill="#edf5ff" font-family="Arial" font-size="11">{html.escape(node["label"])}</text>')
    legend_x = 44
    for index, kind in enumerate(ordered_types):
        x = legend_x + index * 205
        parts.append(f'<circle cx="{x}" cy="884" r="7" fill="{COLORS[kind]}"/>')
        parts.append(f'<text x="{x + 13}" y="889" fill="#c7d5e5" font-family="Arial" font-size="12">{html.escape(kind.replace("_", " "))}</text>')
    parts.append('</svg>')
    STATIC.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = build()
    print(json.dumps({"nodes": len(result["nodes"]), "edges": len(result["edges"]), "output": str(OUTPUT)}, indent=2))
