import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"

SCRIPTS = [
    ("Core metrics", "evaluate_metrics.py"),
    ("Failure diagnosis", "diagnose_exp001.py"),
    ("Strict placeholder evaluation", "strict_placeholder_eval.py"),
]


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = REPORT_DIR / f"exp001_evaluation_{timestamp}.txt"

    sections = [
        "SYNTHEGRA - EXP-001 UNIFIED EVALUATION",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "=" * 65,
    ]

    failures = []

    for label, script in SCRIPTS:
        script_path = ROOT / script

        sections.extend([
            "",
            f"## {label}",
            f"Script: {script}",
            "-" * 65,
        ])

        if not script_path.is_file():
            message = f"Script not found: {script}"
            sections.append(message)
            failures.append(message)
            continue

        print(f"\nRunning {script}...")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            output = result.stdout.strip()
            errors = result.stderr.strip()

            sections.append(output or "(No standard output)")

            if errors:
                sections.extend(["", "STDERR:", errors])

            sections.extend(["", f"Exit code: {result.returncode}"])

            if result.returncode != 0:
                failures.append(f"{script} exited with code {result.returncode}")
                print(f"[FAIL] {script}")
            else:
                print(f"[PASS] {script}")

        except OSError as exc:
            message = f"Could not execute {script}: {exc}"
            sections.append(message)
            failures.append(message)
            print(f"[FAIL] {message}")

    sections.extend([
        "",
        "=" * 65,
        "FINAL STATUS: " + ("FAILED" if failures else "SUCCESS"),
    ])

    if failures:
        sections.extend(["", "Failures:"])
        sections.extend(f"- {failure}" for failure in failures)

    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")

    print("\n" + "=" * 55)
    print("Evaluation report:", report_path)
    print("Final status:", "FAILED" if failures else "SUCCESS")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
