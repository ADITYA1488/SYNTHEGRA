import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "test_dataset.csv",
    "finetuned_model.jsonl",
    "exp001_metrics.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter/tokenizer.json",
    "adapter/tokenizer_config.json",
    "adapter/training_metrics.json",
]

errors = []
warnings = []


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        errors.append(message)


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Could not read valid JSON from {path}: {exc}")
        print(f"[FAIL] Valid JSON: {path}")
        return None


print("SYNTHEGRA | EXP-001 Artifact Validation")
print("=" * 48)

for relative_path in REQUIRED_FILES:
    path = ROOT / relative_path
    check(
        path.is_file() and path.stat().st_size > 0,
        f"Artifact exists and is non-empty: {relative_path}",
    )

metrics_path = ROOT / "exp001_metrics.json"
metrics = load_json(metrics_path) if metrics_path.is_file() else None

if metrics is not None:
    check(isinstance(metrics, dict), "Experiment metrics are a JSON object")

    if isinstance(metrics, dict):
        total = metrics.get("total_examples")
        parseable = metrics.get("parseable_predictions")

        check(
            isinstance(total, int) and total > 0,
            "total_examples is a positive integer",
        )
        check(
            isinstance(parseable, int)
            and isinstance(total, int)
            and 0 <= parseable <= total,
            "parseable_predictions is within the experiment total",
        )

prediction_path = ROOT / "finetuned_model.jsonl"

if prediction_path.is_file():
    valid_records = 0
    invalid_records = 0

    try:
        with prediction_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        valid_records += 1
                    else:
                        invalid_records += 1
                        print(f"[WARN] Non-object JSON at line {line_number}")
                except json.JSONDecodeError:
                    invalid_records += 1
                    print(f"[WARN] Invalid JSON at line {line_number}")

        check(invalid_records == 0, "All prediction lines contain valid JSON objects")
        check(
            metrics is not None
            and valid_records == metrics.get("total_examples"),
            "Prediction record count matches EXP-001 total_examples",
        )
    except OSError as exc:
        errors.append(f"Could not read predictions: {exc}")
        print(f"[FAIL] Read prediction file: {exc}")

adapter_config_path = ROOT / "adapter" / "adapter_config.json"
if adapter_config_path.is_file():
    adapter_config = load_json(adapter_config_path)
    if adapter_config is not None:
        check(
            isinstance(adapter_config, dict),
            "Adapter configuration is a JSON object",
        )

training_metrics_path = ROOT / "adapter" / "training_metrics.json"
if training_metrics_path.is_file():
    training_metrics = load_json(training_metrics_path)
    if training_metrics is not None:
        check(
            isinstance(training_metrics, dict),
            "Training metrics are a JSON object",
        )

print("\n" + "=" * 48)
if errors:
    print(f"VALIDATION FAILED: {len(errors)} issue(s) found.")
    sys.exit(1)

print("VALIDATION PASSED: EXP-001 artifacts are present and readable.")
