import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

BASELINE = PROJECT_DIR / "baseline_predictions.jsonl"
FINETUNED = PROJECT_DIR / "finetuned_model.jsonl"

BASELINE_METRICS = PROJECT_DIR / "baseline_metrics.json"
FINETUNED_METRICS = PROJECT_DIR / "finetuned_comparison_metrics.json"
COMPARISON = PROJECT_DIR / "exp001_comparison.json"


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


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    ids = [str(r["id"]) for r in records]

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {path.name}")

    return {str(r["id"]): r for r in records}


def evaluate(records):
    total = len(records)
    json_valid = 0
    schema_valid = 0
    category_correct = 0
    intent_correct = 0
    response_exact = 0
    placeholder_total = 0
    placeholder_preserved = 0
    parseable = 0

    for record in records.values():
        predicted, strict_valid = parse_prediction(
            record["prediction"]
        )

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

    return {
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


def main():
    print("Loading baseline predictions...")
    baseline = load_records(BASELINE)

    print("Loading fine-tuned predictions...")
    finetuned = load_records(FINETUNED)

    if len(baseline) != 4016 or len(finetuned) != 4016:
        raise ValueError("Expected exactly 4016 predictions per model.")

    if set(baseline) != set(finetuned):
        raise ValueError("Baseline and fine-tuned IDs do not match.")

    expected_fields = [
        "instruction",
        "expected_category",
        "expected_intent",
        "expected_response"
    ]

    for example_id in baseline:
        for field in expected_fields:
            if baseline[example_id].get(field) != finetuned[example_id].get(field):
                raise ValueError(
                    f"Dataset mismatch for ID {example_id}, field {field}"
                )

    print("Dataset alignment verified: 4016 matching examples.")

    baseline_metrics = evaluate(baseline)
    finetuned_metrics = evaluate(finetuned)

    BASELINE_METRICS.write_text(
        json.dumps(baseline_metrics, indent=4),
        encoding="utf-8"
    )

    FINETUNED_METRICS.write_text(
        json.dumps(finetuned_metrics, indent=4),
        encoding="utf-8"
    )

    metric_pairs = {
        "json_validity_pct": "JSON validity",
        "schema_validity_pct": "Schema validity",
        "category_accuracy_pct": "Category accuracy",
        "intent_accuracy_pct": "Intent accuracy",
        "response_exact_match_pct": "Response exact match",
        "placeholder_preservation_pct": "Placeholder preservation"
    }

    comparison = {
        "test_examples": 4016,
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "differences_percentage_points": {}
    }

    print("\n========== BASELINE VS FINETUNED ==========")
    print(f"{'Metric':<32} {'Baseline':>12} {'Fine-tuned':>12} {'Change':>12}")

    for key, label in metric_pairs.items():
        base = baseline_metrics[key]
        fine = finetuned_metrics[key]
        difference = round(fine - base, 2)

        comparison["differences_percentage_points"][key] = difference

        print(
            f"{label:<32} {base:>11.2f}% "
            f"{fine:>11.2f}% {difference:>+11.2f} pp"
        )

    COMPARISON.write_text(
        json.dumps(comparison, indent=4),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(BASELINE_METRICS)
    print(FINETUNED_METRICS)
    print(COMPARISON)
    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()
