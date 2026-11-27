
import gc
import json
import time
import statistics
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from contextlib import nullcontext

PROJECT_DIR = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = PROJECT_DIR / "adapter"
DATASET = PROJECT_DIR / "test_dataset.csv"
OUTPUT = PROJECT_DIR / "exp002b_results.json"

MAX_NEW_TOKENS = 128
NUM_TRIALS = 4
SAMPLE_INDICES = [0, 100, 500, 1000, 2000]

SYSTEM_PROMPT = """You are an enterprise customer support assistant.
Identify the customer's category and intent and generate a helpful response.
Return valid JSON containing exactly category, intent, response.
Do not invent customer-specific details. Preserve placeholders."""

def synchronize():
    torch.cuda.synchronize()

def build_prompt(tokenizer, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(instruction)}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def load_model(mode):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    if mode in ("peft_disabled", "peft_enabled", "merged"):
        model = PeftModel.from_pretrained(
            model,
            ADAPTER,
            is_trainable=False
        )

    if mode == "merged":
        model = model.merge_and_unload()

    model.eval()
    return model

def adapter_context(model, mode):
    if mode == "peft_disabled":
        return model.disable_adapter()

    return nullcontext()

def benchmark_configuration(mode, tokenizer, prompts, trial):
    print(f"\nTrial {trial}: {mode}")

    gc.collect()
    torch.cuda.empty_cache()

    synchronize()
    load_start = time.perf_counter()

    model = load_model(mode)

    synchronize()
    load_seconds = time.perf_counter() - load_start

    # Warm-up
    warmup = tokenizer(
        prompts[0], return_tensors="pt"
    ).to("cuda:0")

    with torch.inference_mode(), adapter_context(model, mode):
        model.generate(
            **warmup,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    synchronize()

    latencies = []
    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt"
        ).to("cuda:0")

        input_length = inputs["input_ids"].shape[1]

        synchronize()
        start = time.perf_counter()

        with torch.inference_mode(), adapter_context(model, mode):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        synchronize()
        elapsed = time.perf_counter() - start

        generated = outputs.shape[1] - input_length
        total_tokens += generated
        latencies.append(elapsed)

        print(
            f"  Sample {i + 1}: "
            f"{generated} tokens, {elapsed:.3f}s"
        )

    total_seconds = sum(latencies)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "trial": trial,
        "mode": mode,
        "model_loading_seconds": round(load_seconds, 4),
        "total_output_tokens": total_tokens,
        "total_generation_seconds": round(total_seconds, 4),
        "average_latency_seconds": round(
            statistics.mean(latencies), 4
        ),
        "generation_tokens_per_second": round(
            total_tokens / total_seconds, 4
        ),
        "peak_gpu_memory_gb": round(peak_memory, 4)
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result

def summarize(results):
    summary = {}

    metrics = [
        "model_loading_seconds",
        "average_latency_seconds",
        "generation_tokens_per_second",
        "peak_gpu_memory_gb"
    ]

    for mode, trials in results.items():
        summary[mode] = {}

        for metric in metrics:
            values = [item[metric] for item in trials]
            summary[mode][metric] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(
                    statistics.stdev(values), 4
                ) if len(values) > 1 else 0
            }

    return summary

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    df = pd.read_csv(DATASET).fillna("")
    assert len(df) == 4016

    prompts = [
        build_prompt(tokenizer, df.iloc[i]["instruction"])
        for i in SAMPLE_INDICES
    ]

    modes = [
        "base",
        "peft_disabled",
        "peft_enabled",
        "merged"
    ]

    results = {mode: [] for mode in modes}

    # Alternate order to reduce systematic order bias.
    orders = [
        modes,
        list(reversed(modes)),
        modes,
    list(reversed(modes))
    ]

    for trial, order in enumerate(orders, start=1):
        print(f"\n========== TRIAL {trial} ==========")

        for mode in order:
            result = benchmark_configuration(
                mode, tokenizer, prompts, trial
            )
            results[mode].append(result)

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "FP16",
        "max_new_tokens": MAX_NEW_TOKENS,
        "sample_indices": SAMPLE_INDICES,
        "trials_per_configuration": NUM_TRIALS,
        "raw_results": results,
        "summary": summarize(results)
    }

    OUTPUT.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\n========== SUMMARY ==========")
    for mode, metrics in output["summary"].items():
        print(f"\n{mode}")
        for metric, values in metrics.items():
            print(
                f"  {metric}: "
                f"{values['mean']} +/- {values['std']}"
            )

    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()
