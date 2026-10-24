from pathlib import Path
from collections import Counter
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
RESULTS_DIR = PROJECT / "results"

BATCH_SIZE = 8
MAX_LENGTH = 256
THRESHOLD = 0.5

def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

class TestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

def make_collator(tokenizer):
    def collate(rows):
        encoded = tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(row["label"]) for row in rows],
            dtype=torch.long,
        )
        return encoded
    return collate

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check GPU setup.")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading official test set...")
    rows = load_jsonl(TEST_FILE)

    if not rows:
        raise RuntimeError("The official test set is empty.")

    if any(row.get("partition") != "test" for row in rows):
        raise RuntimeError("Unexpected partition found in test file.")

    labels = np.asarray([int(row["label"]) for row in rows])

    print("Test examples:", len(rows))
    print("Class counts:", dict(Counter(labels.tolist())))
    print("Loading saved best checkpoint...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    loader = DataLoader(
        TestDataset(rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collator(tokenizer),
        pin_memory=True,
    )

    probabilities = []

    print("Running official test evaluation...")
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            batch.pop("labels")
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                logits = model(**batch).logits

            batch_probs = torch.softmax(
                logits.float(), dim=1
            )[:, 1]

            probabilities.extend(
                batch_probs.cpu().numpy().tolist()
            )

            if step % 100 == 0 or step == len(loader):
                print(f"Processed batch {step}/{len(loader)}")

    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "evaluation": "official DDI Corpus test partition",
        "test_examples": int(len(rows)),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(labels, y_pred)),
        "positive_precision": float(
            precision_score(labels, y_pred, zero_division=0)
        ),
        "positive_recall": float(
            recall_score(labels, y_pred, zero_division=0)
        ),
        "positive_f1": float(
            f1_score(labels, y_pred, zero_division=0)
        ),
        "pr_auc_average_precision": float(
            average_precision_score(labels, y_prob)
        ),
        "roc_auc": float(roc_auc_score(labels, y_prob)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            y_pred,
            labels=[0, 1],
            target_names=["no_annotated_DDI", "annotated_DDI"],
            output_dict=True,
            zero_division=0,
        ),
        "caution": (
            "Text-based benchmark results; not patient-level clinical "
            "validation or evidence that a medication combination is safe."
        ),
    }

    metrics_path = RESULTS_DIR / "official_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_path = RESULTS_DIR / "official_test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row, probability, prediction in zip(
            rows, y_prob, y_pred
        ):
            output = {
                "pair_id": row.get("pair_id"),
                "document_id": row.get("document_id"),
                "sentence": row.get("sentence"),
                "entity1": row.get("entity1"),
                "entity1_type": row.get("entity1_type"),
                "entity2": row.get("entity2"),
                "entity2_type": row.get("entity2_type"),
                "true_label": int(row["label"]),
                "predicted_label": int(prediction),
                "predicted_ddi_probability": float(probability),
                "threshold": THRESHOLD,
                "relation_type": row.get("relation_type"),
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print("\n===== OFFICIAL TEST RESULTS =====")
    print(json.dumps(metrics, indent=2))
    print("\nMetrics saved to:", metrics_path)
    print("Predictions saved to:", predictions_path)

if __name__ == "__main__":
    main()
