from pathlib import Path
import json
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

PROJECT = Path(r"D:\extracurriculars\Medi-Guard")
SOURCE = PROJECT / "models" / "mediguard-onnx" / "model.onnx"
OUTPUT = PROJECT / "models" / "mediguard-onnx" / "model.int8_pertensor.onnx"
REPORT = PROJECT / "results" / "int8_pertensor_export.json"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Missing source model: {SOURCE}")

print("Checking original ONNX model...")
onnx.checker.check_model(str(SOURCE))

print("Creating per-tensor dynamic INT8 model...")
quantize_dynamic(
    model_input=str(SOURCE),
    model_output=str(OUTPUT),
    weight_type=QuantType.QInt8,
    per_channel=False,
    op_types_to_quantize=["MatMul"],
)

print("Checking quantized model...")
onnx.checker.check_model(str(OUTPUT))

report = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "source_bytes": SOURCE.stat().st_size,
    "output_bytes": OUTPUT.stat().st_size,
    "size_reduction_fraction": (
        1 - OUTPUT.stat().st_size / SOURCE.stat().st_size
    ),
    "quantization": "dynamic INT8, per-tensor, MatMul",
    "onnx_check": "passed",
    "clinical_validation": False,
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
