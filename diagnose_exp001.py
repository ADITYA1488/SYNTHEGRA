import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
path = PROJECT_DIR / "finetuned_model.jsonl"

def placeholders(text):
    return set(re.findall(
        r"\[[^\[\]]+\]|\{\{[^{}]+\}\}|\{[^{}]+\}",
        str(text)
    ))

invalid = []
missing = []

with path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        r = json.loads(line)

        try:
            p = json.loads(r["prediction"])
            if not isinstance(p, dict) or set(p) != {
                "category", "intent", "response"
            }:
                invalid.append(r)
                continue
        except Exception:
            invalid.append(r)
            continue

        expected = placeholders(r["expected_response"])
        actual = placeholders(p.get("response", ""))

        lost = expected - actual
        if lost:
            missing.append((r, lost))

print("Malformed/schema-invalid outputs:", len(invalid))
print("Examples with missing placeholders:", len(missing))

print("\n========== INVALID OUTPUT EXAMPLES ==========")
for r in invalid[:5]:
    print("\nID:", r["id"])
    print("Instruction:", r["instruction"][:250])
    print("Prediction:", r["prediction"][:500])

print("\n========== PLACEHOLDER FAILURE EXAMPLES ==========")
for r, lost in missing[:10]:
    print("\nID:", r["id"])
    print("Missing:", sorted(lost))
    print("Expected:", r["expected_response"][:400])
    print("Predicted:", r["prediction"][:500])
