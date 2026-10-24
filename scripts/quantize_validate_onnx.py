from pathlib import Path
import json
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
MODEL_DIR = PROJECT / "models" / "mediguard-ddi"
ONNX_FILE = PROJECT / "models" / "mediguard-onnx" / "model.onnx"
INT8_FILE = PROJECT / "models" / "mediguard-onnx" / "model.int8.onnx"
TEST_FILE = PROJECT / "data" / "processed" / "ddi_test.jsonl"
RESULT_FILE = PROJECT / "results" / "onnx_int8_validation.json"

MAX_LENGTH = 256
NUM_EXAMPLES = 64

def softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()

def main():
    for path in (MODEL_DIR, ONNX_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    print("Checking original ONNX model...")
    onnx.checker.check_model(onnx.load(str(ONNX_FILE)))
    print("Original ONNX graph is valid.")

    print("\nQuantizing model weights to INT8...")
    print("The original ONNX file will not be overwritten.")

    start = time.time()
    quantize_dynamic(
        model_input=str(ONNX_FILE),
        model_output=str(INT8_FILE),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )

    print(f"Quantization completed in {time.time() - start:.1f} seconds.")

    print("\nChecking INT8 ONNX graph...")
    onnx.checker.check_model(onnx.load(str(INT8_FILE)))
    print("INT8 graph check passed.")

    print("\nLoading original PyTorch model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()

    print("Loading quantized ONNX model...")
    session = ort.InferenceSession(
        str(INT8_FILE),
        providers=["CPUExecutionProvider"],
    )
    input_names = [item.name for item in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    rows = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not rows or any(row.get("partition") != "test" for row in rows):
        raise RuntimeError(
            "Test file is empty or contains unexpected partitions."
        )

    indices = np.linspace(
        0, len(rows) - 1, min(NUM_EXAMPLES, len(rows)), dtype=int
    )
    selected = [rows[i] for i in indices]

    matches = 0
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    comparisons = []

    print(f"\nComparing INT8 with PyTorch on {len(selected)} examples...")

    with torch.inference_mode():
        for i, row in enumerate(selected, start=1):
            encoded = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )

            original_logits = model(**encoded).logits.cpu().numpy()[0]
            original_probs = softmax(original_logits)

            feed = {}
            for name in input_names:
                if name not in encoded:
                    raise RuntimeError(
                        f"Required ONNX input '{name}' is missing from "
                        f"tokenizer output: {list(encoded.keys())}"
                    )
                feed[name] = encoded[name].cpu().numpy()

            int8_logits = session.run([output_name], feed)[0][0]
            int8_probs = softmax(int8_logits)

            original_class = int(np.argmax(original_probs))
            int8_class = int(np.argmax(int8_probs))
            same = original_class == int8_class

            logit_diff = float(
                np.max(np.abs(original_logits - int8_logits))
            )
            prob_diff = float(
                np.max(np.abs(original_probs - int8_probs))
            )

            matches += int(same)
            max_logit_diff = max(max_logit_diff, logit_diff)
            max_prob_diff = max(max_prob_diff, prob_diff)

            comparisons.append({
                "example": i,
                "true_label": int(row["label"]),
                "pytorch_class": original_class,
                "int8_class": int8_class,
                "classes_match": same,
                "max_logit_absolute_difference": logit_diff,
                "max_probability_absolute_difference": prob_diff,
            })

            if i % 16 == 0 or i == len(selected):
                print(f"Compared {i}/{len(selected)}")

    original_size = ONNX_FILE.stat().st_size
    int8_size = INT8_FILE.stat().st_size

    result = {
        "quantization": "dynamic INT8 weights",
        "onnx_checker_passed": True,
        "examples_compared": len(selected),
        "matching_predicted_classes": matches,
        "class_agreement": matches / len(selected),
        "all_classes_match": matches == len(selected),
        "max_absolute_logit_difference": max_logit_diff,
        "max_absolute_probability_difference": max_prob_diff,
        "original_onnx_size_bytes": original_size,
        "int8_onnx_size_bytes": int8_size,
        "size_reduction_percent": (
            (original_size - int8_size) / original_size * 100
        ),
        "comparisons": comparisons,
        "note": (
            "Sampled technical comparison only. Not full benchmark "
            "reevaluation, clinical validation, or proof of medication safety."
        ),
    }

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n===== INT8 VALIDATION RESULTS =====")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "comparisons"},
        indent=2,
    ))
    print(f"\nDetailed report: {RESULT_FILE}")
    print(f"INT8 model: {INT8_FILE}")

if __name__ == "__main__":
    main()
