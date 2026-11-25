import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_DIR = Path(__file__).resolve().parent

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_SOURCE = PROJECT_DIR / "adapter"
DATASET = PROJECT_DIR / "test_dataset.csv"
OUTPUT = PROJECT_DIR / "baseline_predictions.jsonl"

MAX_NEW_TOKENS = 384
CHECKPOINT_INTERVAL = 25

SYSTEM_PROMPT = """You are an enterprise customer support assistant.
Identify the customer's category and intent and generate a helpful response.
Return valid JSON containing exactly category, intent, response.
Do not invent customer-specific details. Preserve placeholders."""

def main():
    assert torch.cuda.is_available(), "CUDA GPU not available"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading test dataset...")

    df = pd.read_csv(DATASET).fillna("")
    print("Test examples:", len(df))
    assert len(df) == 4016, "Unexpected test dataset size"

    completed = set()

    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    print("Already completed:", len(completed))
    print("Remaining:", len(df) - len(completed))

    if len(completed) == len(df):
        print("Baseline predictions already complete.")
        return

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)

    print("\nLoading BASE MODEL without LoRA adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )
    model.eval()

    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )

    count = 0

    with OUTPUT.open("a", encoding="utf-8") as out:
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="BASELINE"
        ):
            example_id = str(idx)

            if example_id in completed:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row["instruction"])}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to("cuda:0")

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    )
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True
            )

            record = {
                "id": example_id,
                "instruction": str(row["instruction"]),
                "expected_category": str(row["category"]),
                "expected_intent": str(row["intent"]),
                "expected_response": str(row["response"]),
                "prediction": prediction
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % CHECKPOINT_INTERVAL == 0:
                out.flush()
                print(
                    f"\nCheckpoint: {len(completed) + count}/{len(df)}"
                )

        out.flush()

    print("\nBASELINE GENERATION COMPLETE")
    print("Predictions:", OUTPUT)
    print("Total records:", len(completed) + count)
    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB"
    )

if __name__ == "__main__":
    main()
