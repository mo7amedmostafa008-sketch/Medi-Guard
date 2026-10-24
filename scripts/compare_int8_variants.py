from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
ONNX_DIR = PROJECT / "models" / "mediguard-onnx"
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
OUT_FILE = PROJECT / "results" / "int8_variant_comparison.json"

BATCH_SIZE = 16
MAX_LENGTH = 256

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def metrics(y, probs):
    pred = (probs[:, 1] >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "positive_precision": float(precision_score(y, pred, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, zero_division=0)),
        "positive_f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, probs[:, 1])),
        "roc_auc": float(roc_auc_score(y, probs[:, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }

def main():
    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or any(r.get("partition") != "test" for r in rows):
        raise RuntimeError("Unexpected or empty test dataset.")

    texts = [r["text"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

    models = {
        "onnx_fp32": ONNX_DIR / "model.onnx",
        "int8_per_channel": ONNX_DIR / "model.int8.onnx",
        "int8_per_tensor": ONNX_DIR / "model.int8_pertensor.onnx",
    }

    sessions = {}
    for name, path in models.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing model: {path}")
        sessions[name] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

    all_probs = {name: [] for name in models}

    print(f"Evaluating {len(rows)} held-out examples...")
    for start in range(0, len(rows), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        for name, session in sessions.items():
            feed = {
                inp.name: encoded[inp.name]
                for inp in session.get_inputs()
            }
            logits = session.run(None, feed)[0]
            all_probs[name].append(softmax(logits))

        done = min(start + len(batch_texts), len(rows))
        if done % 512 < BATCH_SIZE or done == len(rows):
            print(f"Processed {done}/{len(rows)}")

    probs = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in all_probs.items()
    }

    reference_classes = np.argmax(probs["onnx_fp32"], axis=1)
    result = {
        "test_examples": len(rows),
        "threshold": 0.5,
        "metrics": {},
        "comparison_to_fp32": {},
        "model_sizes_bytes": {
            name: path.stat().st_size for name, path in models.items()
        },
        "note": (
            "Held-out DDI text benchmark only; not clinical validation "
            "or proof of patient-level medication safety."
        ),
    }

    for name, p in probs.items():
        result["metrics"][name] = metrics(y, p)
        classes = np.argmax(p, axis=1)
        result["comparison_to_fp32"][name] = {
            "class_agreement": float(np.mean(classes == reference_classes)),
            "class_disagreements": int(np.sum(classes != reference_classes)),
            "max_absolute_probability_difference": float(
                np.max(np.abs(p - probs["onnx_fp32"]))
            ),
            "mean_absolute_probability_difference": float(
                np.mean(np.abs(p - probs["onnx_fp32"]))
            ),
        }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n===== METRICS =====")
    print(json.dumps(result["metrics"], indent=2))
    print("\n===== AGREEMENT AND PROBABILITY DIFFERENCES =====")
    print(json.dumps(result["comparison_to_fp32"], indent=2))
    print(f"\nFull report: {OUT_FILE}")

if __name__ == "__main__":
    main()
