import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
path = PROJECT_DIR / "finetuned_model.jsonl"

def extract_placeholders(text):
    return set(re.findall(r"\{\{[^{}]+\}\}", str(text)))

total = 0
fully_preserved = 0
partially_preserved = 0
none_preserved = 0

for line in path.open(encoding="utf-8"):
    if not line.strip():
        continue

    r = json.loads(line)
    expected = extract_placeholders(r["expected_response"])

    if not expected:
        continue

    total += 1

    try:
        prediction = json.loads(r["prediction"])
        response = prediction.get("response", "")
    except Exception:
        response = ""

    actual = extract_placeholders(response)

    if expected.issubset(actual):
        fully_preserved += 1
    elif expected.intersection(actual):
        partially_preserved += 1
    else:
        none_preserved += 1

print("Examples containing placeholders:", total)
print("All placeholders preserved:", fully_preserved)
print("Partially preserved:", partially_preserved)
print("No placeholders preserved:", none_preserved)
print(
    "Strict example-level preservation:",
    round(fully_preserved / total * 100, 2) if total else 0, "%"
)
