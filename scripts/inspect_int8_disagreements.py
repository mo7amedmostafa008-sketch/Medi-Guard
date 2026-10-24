from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
INT8_FILE = PROJECT / "models" / "mediguard-onnx" / "model.int8.onnx"
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
OUT_FILE = PROJECT / "results" / "int8_disagreements.json"

BATCH_SIZE = 16
MAX_LENGTH = 256

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def main():
    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not rows or any(r.get("partition") != "test" for r in rows):
        raise RuntimeError("Unexpected or empty test dataset.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()

    session = ort.InferenceSession(
        str(INT8_FILE), providers=["CPUExecutionProvider"]
    )
    input_names = [x.name for x in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    differences = []

    print(f"Inspecting {len(rows)} examples...")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        encoded = tokenizer(
            [r["text"] for r in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.inference_mode():
            original_logits = model(**encoded).logits.cpu().numpy()

        original_probs = softmax(original_logits)

        feed = {
            name: encoded[name].cpu().numpy()
            for name in input_names
        }
        int8_logits = session.run([output_name], feed)[0]
        int8_probs = softmax(int8_logits)

        for j, row in enumerate(batch):
            p = original_probs[j]
            q = int8_probs[j]
            original_class = int(np.argmax(p))
            int8_class = int(np.argmax(q))

            differences.append({
                "test_row_index": start + j,
                "true_label": int(row["label"]),
                "original_class": original_class,
                "int8_class": int8_class,
                "classes_match": original_class == int8_class,
                "original_positive_probability": float(p[1]),
                "int8_positive_probability": float(q[1]),
                "absolute_probability_difference": float(
                    np.max(np.abs(p - q))
                ),
                "text": row["text"],
            })

        done = min(start + len(batch), len(rows))
        if done % 512 < BATCH_SIZE or done == len(rows):
            print(f"Processed {done}/{len(rows)}")

    disagreements = [
        item for item in differences if not item["classes_match"]
    ]
    largest_probability_changes = sorted(
        differences,
        key=lambda item: item["absolute_probability_difference"],
        reverse=True,
    )[:20]

    result = {
        "test_examples": len(rows),
        "class_disagreement_count": len(disagreements),
        "class_agreement": (
            1 - len(disagreements) / len(rows)
        ),
        "disagreements": disagreements,
        "largest_probability_changes": largest_probability_changes,
        "note": (
            "Diagnostic analysis of quantization differences on the "
            "held-out DDI text benchmark; not clinical validation."
        ),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== DISAGREEMENT SUMMARY =====")
    print(json.dumps({
        "test_examples": result["test_examples"],
        "class_disagreement_count":
            result["class_disagreement_count"],
        "class_agreement": result["class_agreement"],
        "largest_probability_changes": [
            {
                k: item[k] for k in (
                    "test_row_index",
                    "true_label",
                    "original_class",
                    "int8_class",
                    "original_positive_probability",
                    "int8_positive_probability",
                    "absolute_probability_difference",
                )
            }
            for item in largest_probability_changes[:5]
        ],
    }, indent=2))

    print(f"\nFull diagnostic report: {OUT_FILE}")

if __name__ == "__main__":
    main()
