"""Create a source-text-free public graph table from governed PEDAKI assertions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "public_edges.csv"

AKI_NAMES = {
    "aki", "acute kidney injury", "acute kidney injury (aki)", "acute kidney injury（aki）",
    "急性肾损伤", "急性肾损伤 (aki)", "急性肾损伤（aki）", "儿科急性肾损伤",
}
ALIASES = {
    "death": "Mortality", "mortality": "Mortality", "死亡": "Mortality", "死亡率": "Mortality",
    "chronic kidney disease": "Chronic kidney disease", "ckd": "Chronic kidney disease",
    "慢性肾病": "Chronic kidney disease", "慢性肾脏病": "Chronic kidney disease",
    "serum creatinine": "Serum creatinine", "血清肌酐": "Serum creatinine",
    "sepsis": "Sepsis", "脓毒症": "Sepsis",
    "cardiac surgery": "Cardiac surgery", "heart surgery": "Cardiac surgery", "心脏手术": "Cardiac surgery",
    "kidney replacement therapy": "Kidney replacement therapy", "krt": "Kidney replacement therapy",
    "肾脏替代治疗": "Kidney replacement therapy",
    "continuous renal replacement therapy": "Continuous kidney replacement therapy",
    "crrt": "Continuous kidney replacement therapy",
    "腹膜透析": "Peritoneal dialysis", "peritoneal dialysis": "Peritoneal dialysis",
}
GENERIC = {
    "patient", "patients", "患者", "children", "child", "儿童", "儿科患者", "newborn", "新生儿",
    "study", "研究", "group", "组", "cohort", "population",
}
POPULATION_LABELS = {
    "非新生儿儿科": "Non-neonatal pediatric", "新生儿": "Neonatal",
    "儿科年龄未明": "Pediatric age unclear", "明确非儿科": "Explicitly non-pediatric",
    "原文未明": "Not specified", "未明确": "Not specified",
}
SETTING_LABELS = {
    "儿科重症监护": "PICU", "普通住院": "General inpatient", "心脏手术": "Cardiac surgery",
    "脓毒症": "Sepsis", "肾脏替代治疗": "Kidney replacement therapy",
    "肿瘤或移植": "Oncology or transplant", "肾毒性暴露": "Nephrotoxic exposure",
    "原文未明": "Not specified",
}
RELATION_LABELS = {
    "相关": "Association", "风险因素": "Risk factor", "管理或干预": "Management or intervention",
    "诊断或分期": "Diagnosis or staging", "结局": "Outcome", "预测": "Prediction",
    "监测": "Monitoring", "推荐或共识": "Recommendation or consensus", "未确定": "Uncertain relation",
}


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def canonical_concept(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip()).strip(" .,:;")
    folded = text.casefold()
    if folded in AKI_NAMES:
        return "Acute kidney injury"
    return ALIASES.get(folded, text)


def is_public_display_label(value: str) -> bool:
    """Keep only stable English labels in the public summary visualization."""
    return value.isascii() and bool(re.search(r"[A-Za-z]", value))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_source_ids(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["来源编号"]: row.get("医学文献数据库编号", "")
            for row in csv.DictReader(handle)
            if row.get("来源编号")
        }


def add_edge(store: dict, source: tuple[str, str, str], target: tuple[str, str, str], predicate: str,
             layer: str, source_id: str = "", population: str = "", setting: str = "") -> None:
    key = (source[0], target[0], predicate, layer)
    item = store.setdefault(key, {
        "source": source[0], "source_label": source[1], "source_type": source[2],
        "target": target[0], "target_label": target[1], "target_type": target[2],
        "predicate": predicate, "layer": layer, "assertion_count": 0,
        "source_ids": set(), "populations": set(), "care_settings": set(),
    })
    item["assertion_count"] += 1
    if source_id:
        item["source_ids"].add(source_id)
    if population:
        item["populations"].add(population)
    if setting:
        item["care_settings"].add(setting)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assertions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--guidelines", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-concepts", type=int, default=180)
    args = parser.parse_args()

    assertions = load_jsonl(args.assertions)
    source_to_pmid = load_source_ids(args.corpus)
    frequency: Counter[str] = Counter()
    normalized: list[tuple[dict, str, str]] = []
    for item in assertions:
        subject = canonical_concept(item.get("临床主体", ""))
        target = canonical_concept(item.get("临床客体", ""))
        normalized.append((item, subject, target))
        for concept in {subject, target}:
            if (
                concept
                and concept.casefold() not in GENERIC
                and len(concept) <= 100
                and is_public_display_label(concept)
            ):
                frequency[concept] += 1
    selected = {name for name, _ in frequency.most_common(args.max_concepts)} | {"Acute kidney injury"}

    edges: dict[tuple[str, str, str, str], dict] = {}
    concept_context: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {
        "population": Counter(), "setting": Counter(), "relation": Counter(),
    })
    for item, subject, target in normalized:
        population = POPULATION_LABELS.get(item.get("人口标签", ""), item.get("人口标签", "") or "Not specified")
        setting = SETTING_LABELS.get(item.get("场景标签", ""), item.get("场景标签", "") or "Not specified")
        relation = RELATION_LABELS.get(item.get("受控关系", ""), item.get("受控关系", "") or "Uncertain relation")
        if not is_public_display_label(population):
            population = "Other population context"
        if not is_public_display_label(setting):
            setting = "Other care setting"
        if not is_public_display_label(relation):
            relation = "Uncertain relation"
        pmid = source_to_pmid.get(item.get("来源编号", ""), "")
        internal_source_id = item.get("来源编号", "")
        public_source_id = (
            f"PMID:{pmid}"
            if pmid
            else stable_id("SOURCE", internal_source_id) if internal_source_id else ""
        )
        kept = [concept for concept in {subject, target} if concept in selected]
        for concept in kept:
            concept_context[concept]["population"][population] += 1
            concept_context[concept]["setting"][setting] += 1
            concept_context[concept]["relation"][relation] += 1
        if subject in selected and target in selected and subject != target:
            add_edge(
                edges,
                (stable_id("concept", subject), subject, "clinical_concept"),
                (stable_id("concept", target), target, "clinical_concept"),
                relation, "empirical_assertion", public_source_id, population, setting,
            )

    for concept, contexts in concept_context.items():
        concept_node = (stable_id("concept", concept), concept, "clinical_concept")
        for population, count in contexts["population"].most_common(3):
            if count >= 2:
                for _ in range(count):
                    add_edge(edges, concept_node, (stable_id("population", population), population, "population"),
                             "Observed in population", "population_context")
        for setting, count in contexts["setting"].most_common(3):
            if count >= 2:
                for _ in range(count):
                    add_edge(edges, concept_node, (stable_id("setting", setting), setting, "care_setting"),
                             "Observed in care setting", "care_setting_context")
        for relation, count in contexts["relation"].most_common(2):
            if count >= 2:
                for _ in range(count):
                    add_edge(edges, concept_node, (stable_id("relation", relation), relation, "relation"),
                             "Participates in relation", "relation_context")

    if args.guidelines and args.guidelines.exists():
        with args.guidelines.open(encoding="utf-8-sig", newline="") as handle:
            guideline_rows = list(csv.DictReader(handle))
        hub = ("guideline:layer", "Guideline evidence layer", "guideline")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in guideline_rows:
            grouped[row.get("主题", "Other")].append(row)
        for topic, rows in grouped.items():
            label = RELATION_LABELS.get(topic, SETTING_LABELS.get(topic, topic))
            if not is_public_display_label(label):
                label = "Other indexed topic"
            node = (stable_id("guideline_topic", label), label, "guideline_topic")
            unique_guides = {row.get("指南候选编号", "") for row in rows if row.get("指南候选编号")}
            for _ in rows:
                add_edge(edges, hub, node, "Indexed normative topic", "guideline_index")
            edges[(hub[0], node[0], "Indexed normative topic", "guideline_index")]["source_ids"].update(unique_guides)

    rows = []
    for key, item in edges.items():
        item["edge_id"] = stable_id("edge", "|".join(key))
        item["source_count"] = len(item["source_ids"])
        item["source_ids"] = ";".join(sorted(item["source_ids"])[:20])
        item["population"] = ";".join(sorted(item.pop("populations")))
        item["care_setting"] = ";".join(sorted(item.pop("care_settings")))
        rows.append(item)
    rows.sort(key=lambda row: (-row["assertion_count"], row["edge_id"]))
    fieldnames = [
        "edge_id", "source", "source_label", "source_type", "target", "target_label", "target_type",
        "predicate", "layer", "assertion_count", "source_count", "source_ids", "population", "care_setting",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"assertions": len(assertions), "selected_concepts": len(selected), "public_edges": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
