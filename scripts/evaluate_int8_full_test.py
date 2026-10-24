from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
INT8_FILE = PROJECT / "models" / "mediguard-onnx" / "model.int8.onnx"
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
OUT_FILE = PROJECT / "results" / "int8_full_test_metrics.json"

MAX_LENGTH = 256
BATCH_SIZE = 16

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def metrics(y_true, probs):
    preds = (probs[:, 1] >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "positive_precision": float(
            precision_score(y_true, preds, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(y_true, preds, zero_division=0)
        ),
        "positive_f1": float(
            f1_score(y_true, preds, zero_division=0)
        ),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probs[:, 1])
        ),
        "roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
        "confusion_matrix": confusion_matrix(
            y_true, preds, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true, preds, labels=[0, 1],
            target_names=["no_annotated_DDI", "annotated_DDI"],
            output_dict=True, zero_division=0
        ),
    }

def main():
    for path in (MODEL_DIR, INT8_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("Loading official test dataset...")
    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not rows or any(r.get("partition") != "test" for r in rows):
        raise RuntimeError("Test dataset is empty or has unexpected partitions.")

    texts = [r["text"] for r in rows]
    y_true = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)

    print(f"Test examples: {len(rows)}")
    print("Loading tokenizer and original model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()

    print("Loading INT8 ONNX model...")
    session = ort.InferenceSession(
        str(INT8_FILE), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    torch_prob_batches = []
    int8_prob_batches = []
    matching_classes = 0
    max_prob_diff = 0.0

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = model(**encoded).logits.cpu().numpy()

        torch_probs = softmax(logits)

        feed = {}
        for name in input_names:
            if name not in encoded:
                raise RuntimeError(
                    f"ONNX input '{name}' missing from tokenizer output."
                )
            feed[name] = encoded[name].cpu().numpy()

        int8_logits = session.run([output_name], feed)[0]
        int8_probs = softmax(int8_logits)

        matching_classes += int(
            np.sum(np.argmax(torch_probs, axis=1) ==
                   np.argmax(int8_probs, axis=1))
        )
        max_prob_diff = max(
            max_prob_diff,
            float(np.max(np.abs(torch_probs - int8_probs)))
        )

        torch_prob_batches.append(torch_probs)
        int8_prob_batches.append(int8_probs)

        done = min(start + len(batch_texts), len(texts))
        if done % 512 < BATCH_SIZE or done == len(texts):
            print(f"Evaluated {done}/{len(texts)}")

    torch_probs = np.concatenate(torch_prob_batches, axis=0)
    int8_probs = np.concatenate(int8_prob_batches, axis=0)

    result = {
        "dataset": "existing official DDI test split",
        "examples": len(rows),
        "threshold": 0.5,
        "original_pytorch": metrics(y_true, torch_probs),
        "quantized_int8": metrics(y_true, int8_probs),
        "pytorch_int8_class_agreement": matching_classes / len(rows),
        "matching_class_predictions": matching_classes,
        "maximum_probability_difference": max_prob_diff,
        "note": (
            "Benchmark comparison on annotated DDI text records. "
            "Not patient-level validation or proof of medication safety."
        ),
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n===== FULL TEST COMPARISON =====")
    print(json.dumps({
        "examples": result["examples"],
        "original_pytorch": result["original_pytorch"],
        "quantized_int8": result["quantized_int8"],
        "pytorch_int8_class_agreement":
            result["pytorch_int8_class_agreement"],
        "matching_class_predictions":
            result["matching_class_predictions"],
        "maximum_probability_difference":
            result["maximum_probability_difference"],
    }, indent=2))
    print(f"\nFull report saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()
