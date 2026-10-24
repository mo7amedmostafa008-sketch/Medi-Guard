from pathlib import Path
from collections import Counter
import json
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import GroupShuffleSplit
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
TRAIN_FILE = PROJECT / "data" / "processed" / "ddi_train.jsonl"
MODEL_SOURCE = PROJECT / "models" / "biomedbert-base"
OUTPUT_DIR = PROJECT / "models" / "mediguard-ddi"
RESULTS_DIR = PROJECT / "results"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
EPOCHS = 3
LEARNING_RATE = 2e-5
VALIDATION_FRACTION = 0.10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "text": row["text"],
            "label": int(row["label"]),
        }

def make_collator(tokenizer):
    def collate(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded
    return collate

def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(**batch).logits

            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    y_true = np.asarray(all_labels, dtype=int)
    y_prob = np.asarray(all_probabilities, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["no_annotated_DDI", "annotated_DDI"],
            output_dict=True,
            zero_division=0,
        ),
        "validation_examples": int(len(y_true)),
        "validation_positive_count": int(y_true.sum()),
        "threshold": 0.5,
    }

def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Stop here and check the PyTorch GPU setup."
        )
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not MODEL_SOURCE.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_SOURCE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Loading training records...")

    rows = load_jsonl(TRAIN_FILE)
    if not rows:
        raise RuntimeError("The training JSONL file is empty.")

    labels = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([
        row["document_id"] or row["source_file"] for row in rows
    ])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
    )
    train_indices, val_indices = next(
        splitter.split(np.zeros(len(rows)), labels, groups)
    )

    train_rows = [rows[i] for i in train_indices]
    val_rows = [rows[i] for i in val_indices]

    train_labels = [int(row["label"]) for row in train_rows]
    counts = Counter(train_labels)
    if 0 not in counts or 1 not in counts:
        raise RuntimeError(f"Training split lacks a class: {counts}")

    print(f"All source training pairs: {len(rows):,}")
    print(f"Model-training pairs: {len(train_rows):,}")
    print(f"Validation pairs: {len(val_rows):,}")
    print("Training class counts:", dict(counts))
    print(
        "Validation class counts:",
        dict(Counter(int(row["label"]) for row in val_rows)),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SOURCE),
        local_files_only=True,
    )

    # Register entity markers as individual tokenizer tokens.
    marker_tokens = [
        "[E1_DRUG]", "[E1_BRAND]", "[E1_GROUP]", "[E1_DRUG_N]",
        "[E2_DRUG]", "[E2_BRAND]", "[E2_GROUP]", "[E2_DRUG_N]",
        "[/E1]", "[/E2]",
    ]
    tokenizer.add_special_tokens({
        "additional_special_tokens": marker_tokens
    })

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_SOURCE),
        num_labels=2,
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    collator = make_collator(tokenizer)
    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PairDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )

    # Inverse-frequency weights reduce the impact of class imbalance.
    total = len(train_labels)
    class_weights = torch.tensor(
        [
            total / (2 * counts[0]),
            total / (2 * counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRAD_ACCUMULATION
    )
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = int(0.1 * total_updates)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    print("Class weights:", class_weights.detach().cpu().tolist())
    print("Batch size:", BATCH_SIZE)
    print("Gradient accumulation:", GRAD_ACCUMULATION)
    print("Effective batch size:", BATCH_SIZE * GRAD_ACCUMULATION)
    print("Epochs:", EPOCHS)
    print("Starting fine-tuning...")

    history = []
    best_f1 = -1.0
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
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
                loss = loss_function(logits, labels_batch)
                loss = loss / GRAD_ACCUMULATION

            scaler.scale(loss).backward()
            total_loss += loss.item() * GRAD_ACCUMULATION

            if (
                step % GRAD_ACCUMULATION == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Running loss {total_loss / step:.4f}"
                )

        metrics = evaluate(model, val_loader, device)
        average_loss = total_loss / max(len(train_loader), 1)

        record = {
            "epoch": epoch,
            "average_training_loss": float(average_loss),
            "validation": metrics,
        }
        history.append(record)

        print(f"\n===== EPOCH {epoch} RESULTS =====")
        print("Training loss:", round(average_loss, 4))
        print(json.dumps(metrics, indent=2))

        if metrics["f1_positive"] > best_f1:
            best_f1 = metrics["f1_positive"]
            best_epoch = epoch

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print("Saved new best validation checkpoint.")

        with (RESULTS_DIR / "training_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=2)

    final_summary = {
        "model_source": str(MODEL_SOURCE),
        "saved_model": str(OUTPUT_DIR),
        "official_test_used_during_training": False,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "epochs_requested": EPOCHS,
        "best_epoch_by_validation_f1": best_epoch,
        "best_validation_f1": best_f1,
        "training_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "training_class_counts": dict(counts),
        "history_file": str(RESULTS_DIR / "training_history.json"),
    }

    summary_path = RESULTS_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== TRAINING COMPLETE =====")
    print(json.dumps(final_summary, indent=2))
    print("\nBest model saved to:", OUTPUT_DIR)
    print("Summary saved to:", summary_path)

if __name__ == "__main__":
    main()
