from pathlib import Path
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
ONNX_FILE = PROJECT / "models" / "mediguard-onnx" / "model.onnx"
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
RESULT_FILE = PROJECT / "results" / "onnx_validation.json"

MAX_LENGTH = 256
NUM_EXAMPLES = 64

def main():
    for path in (MODEL_DIR, ONNX_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    print("Checking ONNX graph...")
    graph = onnx.load(str(ONNX_FILE))
    onnx.checker.check_model(graph)
    print("ONNX graph check passed.")

    print("Loading tokenizer and original PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()

    print("Loading ONNX Runtime session...")
    session = ort.InferenceSession(
        str(ONNX_FILE),
        providers=["CPUExecutionProvider"],
    )

    input_names = [item.name for item in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    print("ONNX inputs:", input_names)
    print("ONNX output:", output_name)

    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not rows:
        raise RuntimeError("The test dataset is empty.")

    if any(row.get("partition") != "test" for row in rows):
        raise RuntimeError(
            "Unexpected partition found. Expected only official test records."
        )

    indices = np.linspace(
        0, len(rows) - 1, min(NUM_EXAMPLES, len(rows)), dtype=int
    )
    selected = [rows[i] for i in indices]

    max_logit_diff = 0.0
    max_prob_diff = 0.0
    matching_classes = 0
    comparisons = []

    print(f"Comparing {len(selected)} examples from the test set...")

    with torch.inference_mode():
        for i, row in enumerate(selected, start=1):
            encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )

            torch_logits = model(**encoded).logits.cpu().numpy()[0]
            torch_probs = torch.softmax(
                torch.tensor(torch_logits), dim=0
            ).numpy()

            feed = {}
            for name in input_names:
                if name not in encoded:
                    raise RuntimeError(
                        f"ONNX requires '{name}', but tokenizer did not "
                        f"produce it. Available: {list(encoded.keys())}"
                    )
                feed[name] = encoded[name].cpu().numpy()

            onnx_logits = session.run([output_name], feed)[0][0]

            def softmax(values):
                values = values - np.max(values)
                exp_values = np.exp(values)
                return exp_values / exp_values.sum()

            onnx_probs = softmax(onnx_logits)

            logit_diff = float(
                np.max(np.abs(torch_logits - onnx_logits))
            )
            prob_diff = float(
                np.max(np.abs(torch_probs - onnx_probs))
            )

            torch_class = int(np.argmax(torch_probs))
            onnx_class = int(np.argmax(onnx_probs))
            same = torch_class == onnx_class

            max_logit_diff = max(max_logit_diff, logit_diff)
            max_prob_diff = max(max_prob_diff, prob_diff)
            matching_classes += int(same)

            comparisons.append({
                "example": i,
                "true_label": int(row["label"]),
                "pytorch_class": torch_class,
                "onnx_class": onnx_class,
                "classes_match": same,
                "max_logit_absolute_difference": logit_diff,
                "max_probability_absolute_difference": prob_diff,
            })

            if i % 16 == 0 or i == len(selected):
                print(f"Compared {i}/{len(selected)}")

    result = {
        "onnx_checker_passed": True,
        "examples_compared": len(selected),
        "matching_predicted_classes": matching_classes,
        "class_agreement": matching_classes / len(selected),
        "max_absolute_logit_difference": max_logit_diff,
        "max_absolute_probability_difference": max_prob_diff,
        "all_classes_match": matching_classes == len(selected),
        "comparisons": comparisons,
        "note": (
            "Export-equivalence smoke test only; not clinical validation "
            "or proof that a medication regimen is safe."
        ),
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\n===== ONNX VALIDATION RESULTS =====")
    print(json.dumps(
        {key: value for key, value in result.items()
         if key != "comparisons"},
        indent=2,
    ))
    print(f"Detailed report saved to: {RESULT_FILE}")

if __name__ == "__main__":
    main()
