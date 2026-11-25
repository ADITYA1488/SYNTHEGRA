# SYNTHEGRA

An LLM engineering and experimentation platform focused on fine-tuning, evaluation, inference engineering, and deployment.

## EXP-001: Customer Support LLM

### Objective
Fine-tune an instruction-following language model to generate structured customer-support classifications and responses.

### Model
- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning: LoRA
- Rank: 8
- Alpha: 16
- Dropout: 0.05
- Decoding: Greedy
- Maximum generation tokens: 384

### Dataset
- Source: bitext/bitext-customer-support-llm-chatbot-training-dataset
- Original rows: 26,872
- Test examples: 4,016
- Split: Group-aware stratified
- Random state: 42

### Initial Fine-tuned Model Results

| Metric | Result |
|---|---:|
| JSON validity | 98.75% |
| Category accuracy | 98.68% |
| Intent accuracy | 98.58% |
| Placeholder token preservation | 75.82% |
| Strict example-level placeholder preservation | 71.39% |

Baseline comparison and broader response-quality evaluation are pending.

## Implemented
- EXP-001 LoRA fine-tuning completed
- Inference over 4,016 test examples completed
- Prediction generation
- JSON and schema validation
- Category and intent evaluation
- Placeholder preservation evaluation
- Error diagnosis scripts

## Planned
- Recover original training source
- Base-model versus fine-tuned-model comparison
- Inference optimization and benchmarking
- Structured output validation and recovery
- FastAPI serving
- Docker deployment
- CI/CD
- Observability and experiment tracking

## Setup

Create a virtual environment:

    python -m venv .venv

Activate on Windows:

    .venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Reconstruct the test dataset:

    python prepare_test.py

Evaluate existing predictions:

    python evaluate_metrics.py
    python diagnose_exp001.py
    python strict_placeholder_eval.py

## Reproducibility

The original training source code is not yet included in this checkpoint. The base model is downloaded separately from Hugging Face.

Check the licensing and redistribution terms of the model, adapter, and dataset before publishing their associated artifacts.

## License

License selection is pending review of third-party component licenses.

## Artifact Validation

Run the preflight validator before evaluating or sharing EXP-001 artifacts:

    python validate_artifacts.py
