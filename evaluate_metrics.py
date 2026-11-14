import json
import re
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

PREDICTIONS = PROJECT_DIR / "finetuned_model.jsonl"
OUTPUT = PROJECT_DIR / "exp001_metrics.json"

def normalize(value):
    return str(value).strip().lower()

def parse_prediction(raw):
    raw = str(raw).strip()

    try:
        return json.loads(raw), True
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), False
        except json.JSONDecodeError:
            pass

    return None, False

def placeholders(text):
    return set(re.findall(
        r"\[[^\[\]]+\]|\{\{[^{}]+\}\}|\{[^{}]+\}",
        str(text)
    ))

with PREDICTIONS.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

total = len(records)
json_valid = 0
schema_valid = 0
category_correct = 0
intent_correct = 0
response_exact = 0
placeholder_total = 0
placeholder_preserved = 0
parseable = 0

for record in records:
    predicted, strict_valid = parse_prediction(record["prediction"])

    if strict_valid:
        json_valid += 1

    if predicted is None:
        continue

    parseable += 1

    required = {"category", "intent", "response"}
    if isinstance(predicted, dict) and set(predicted.keys()) == required:
        schema_valid += 1

    if not isinstance(predicted, dict):
        continue

    if normalize(predicted.get("category", "")) == normalize(
        record["expected_category"]
    ):
        category_correct += 1

    if normalize(predicted.get("intent", "")) == normalize(
        record["expected_intent"]
    ):
        intent_correct += 1

    expected_response = str(record["expected_response"])
    predicted_response = str(predicted.get("response", ""))

    if normalize(expected_response) == normalize(predicted_response):
        response_exact += 1

    expected_placeholders = placeholders(expected_response)

    if expected_placeholders:
        placeholder_total += len(expected_placeholders)
        placeholder_preserved += sum(
            token in predicted_response
            for token in expected_placeholders
        )

def percentage(n, d):
    return round(100 * n / d, 2) if d else 0.0

metrics = {
    "total_examples": total,
    "parseable_predictions": parseable,
    "json_valid_count": json_valid,
    "json_validity_pct": percentage(json_valid, total),
    "schema_valid_count": schema_valid,
    "schema_validity_pct": percentage(schema_valid, total),
    "category_correct": category_correct,
    "category_accuracy_pct": percentage(category_correct, total),
    "intent_correct": intent_correct,
    "intent_accuracy_pct": percentage(intent_correct, total),
    "response_exact_matches": response_exact,
    "response_exact_match_pct": percentage(response_exact, total),
    "placeholder_preserved": placeholder_preserved,
    "placeholder_total": placeholder_total,
    "placeholder_preservation_pct": percentage(
        placeholder_preserved, placeholder_total
    )
}

OUTPUT.write_text(
    json.dumps(metrics, indent=4),
    encoding="utf-8"
)

print("\n========== EXP-001 EVALUATION ==========")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved metrics:", OUTPUT)
