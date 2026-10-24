from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
CORPUS = PROJECT / "data" / "raw" / "ddi_corpus" / "extracted" / "DDICorpus"
OUT = PROJECT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

VALID_TYPES = {"drug", "brand", "group", "drug_n"}
VALID_LABELS = {"true", "false"}


def parse_offsets(raw, sentence_length):
    """Parse inclusive XML offsets, including discontinuous spans."""
    spans = []

    for segment in raw.split(";"):
        start_text, end_text = segment.strip().split("-", 1)
        start = int(start_text)
        end = int(end_text) + 1  # XML end offsets are inclusive.

        if start < 0 or end <= start or end > sentence_length:
            raise ValueError(
                f"Invalid character span {segment!r} for "
                f"sentence length {sentence_length}"
            )

        spans.append((start, end))

    return spans


def mark_pair(sentence, e1, e2):
    """Insert entity markers without changing the original sentence text."""
    spans = []
    entities = [(e1, "E1"), (e2, "E2")]

    for entity, marker in entities:
        entity_type = entity.get("type", "unknown").upper()
        offsets = parse_offsets(
            entity.get("charOffset", ""),
            len(sentence)
        )

        for start, end in offsets:
            spans.append(
                (start, 1, f"[{marker}_{entity_type}]")
            )
            spans.append((end, 0, f"[/{marker}]"))

    # At the same position, close the previous span before opening
    # the next one. This also supports adjacent entities.
    spans.sort(key=lambda item: (item[0], item[1]))

    result = []
    cursor = 0

    for position, _, marker in spans:
        if position < cursor:
            raise ValueError("Overlapping entity spans detected.")

        result.append(sentence[cursor:position])
        result.append(marker)
        cursor = position

    result.append(sentence[cursor:])
    return "".join(result)


def partition_of(path):
    parts = path.relative_to(CORPUS).parts
    if parts and parts[0].lower() in {"train", "test"}:
        return parts[0].lower()
    return None


def main():
    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus folder not found: {CORPUS}")

    xml_files = sorted(CORPUS.rglob("*.xml"))
    records = {"train": [], "test": []}
    summary = {
        "source": str(CORPUS),
        "files_found": len(xml_files),
        "partitions": {},
        "warnings": []
    }

    for path in xml_files:
        partition = partition_of(path)
        if partition is None:
            continue

        document = ET.parse(path).getroot()

        for sentence_element in document.iter("sentence"):
            sentence = sentence_element.get("text", "")
            entities = {
                entity.get("id"): entity
                for entity in sentence_element.findall("entity")
            }

            for pair in sentence_element.findall("pair"):
                label_text = pair.get("ddi", "").lower()

                if label_text not in VALID_LABELS:
                    summary["warnings"].append(
                        f"Unknown DDI label in {path.name}: {label_text!r}"
                    )
                    continue

                e1 = entities.get(pair.get("e1"))
                e2 = entities.get(pair.get("e2"))

                if e1 is None or e2 is None:
                    raise ValueError(
                        f"Unresolved pair reference in {path.name}: "
                        f"{pair.attrib}"
                    )

                type1 = e1.get("type", "unknown")
                type2 = e2.get("type", "unknown")

                if type1 not in VALID_TYPES or type2 not in VALID_TYPES:
                    summary["warnings"].append(
                        f"Unexpected entity type in {path.name}: "
                        f"{type1}, {type2}"
                    )

                try:
                    marked_text = mark_pair(sentence, e1, e2)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to mark pair in {path.name}, "
                        f"sentence {sentence_element.get('id')}: {exc}"
                    ) from exc

                records[partition].append({
                    "text": marked_text,
                    "sentence": sentence,
                    "entity1": e1.get("text", ""),
                    "entity1_type": type1,
                    "entity2": e2.get("text", ""),
                    "entity2_type": type2,
                    "label": 1 if label_text == "true" else 0,
                    "relation_type": pair.get("type"),
                    "document_id": document.get("id", ""),
                    "sentence_id": sentence_element.get("id", ""),
                    "pair_id": pair.get("id", ""),
                    "source_file": path.name,
                    "partition": partition
                })

    for partition, rows in records.items():
        if not rows:
            raise RuntimeError(f"No records extracted for {partition}.")

        output_file = OUT / f"ddi_{partition}.jsonl"

        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        labels = Counter(str(row["label"]) for row in rows)
        relations = Counter(
            row["relation_type"]
            for row in rows
            if row["label"] == 1
        )

        summary["partitions"][partition] = {
            "records": len(rows),
            "negative_label_0": labels.get("0", 0),
            "positive_label_1": labels.get("1", 0),
            "positive_relation_types": dict(relations),
            "output_file": str(output_file)
        }

    summary_path = OUT / "ddi_dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n===== DATASET PREPARATION COMPLETE =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n===== FIRST TRAINING EXAMPLE =====")
    print(json.dumps(records["train"][0], indent=2, ensure_ascii=False))

    print("\n===== OUTPUT FILES =====")
    print(OUT / "ddi_train.jsonl")
    print(OUT / "ddi_test.jsonl")
    print(summary_path)


if __name__ == "__main__":
    main()
