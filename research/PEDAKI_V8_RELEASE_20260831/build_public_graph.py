"""Build the static, public graph payload used by the GitHub Pages browser."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(__file__).resolve().parent / "data" / "relations_public.csv"
OUTPUT = ROOT / "docs" / "graph.json"


def node_type(identifier: str) -> str:
    return identifier.split(":", 1)[0] if ":" in identifier else "entity"


def build() -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row["subject"]
            target = row["object"]
            for identifier, label in ((source, row["subject_label"]), (target, row["object_label"])):
                nodes.setdefault(identifier, {
                    "id": identifier,
                    "label": label,
                    "type": node_type(identifier),
                    "degree": 0,
                })
                nodes[identifier]["degree"] = int(nodes[identifier]["degree"]) + 1
            edges.append({
                "id": row["sro_id"],
                "source": source,
                "target": target,
                "predicate": row["predicate"],
                "layer": row["primary_evidence_layer"],
                "source_count": int(row["independent_source_count"] or 0),
                "replication": row["replication_class"],
                "source_ids": row["supporting_source_ids"].split(";"),
                "domains": row["pediatric_domains"].split(";") if row["pediatric_domains"] else [],
                "population": row["population"],
                "stream": row["stream"],
                "split": row["split"],
                "review_basis": row["review_basis"],
                "confidence": float(row["review_confidence"] or 0),
            })
    edges.sort(key=lambda item: item["id"])
    ordered_nodes = sorted(nodes.values(), key=lambda item: item["id"])
    payload = {
        "release": "PedAKIGraphica public review-adjudicated intermediate graph",
        "scope": "225 construction relations among 34 entities",
        "status": "machine_adjudicated_intermediate_not_clinically_validated",
        "source_text_policy": "full text and evidence-window text excluded from this public browser payload",
        "nodes": ordered_nodes,
        "edges": edges,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = build()
    print(json.dumps({"output": str(OUTPUT), "nodes": len(result["nodes"]), "edges": len(result["edges"])}, indent=2))
