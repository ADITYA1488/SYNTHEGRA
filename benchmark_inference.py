
import gc
import json
import time
import statistics
import torch
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parent

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = PROJECT_DIR / "adapter"
DATASET = PROJECT_DIR / "test_dataset.csv"
OUTPUT = PROJECT_DIR / "benchmark_results.json"

SYSTEM_PROMPT = """You are an enterprise customer support assistant.
Identify the customer's category and intent and generate a helpful response.
Return valid JSON containing exactly category, intent, response.
Do not invent customer-specific details. Preserve placeholders."""

MAX_NEW_TOKENS = 128
NUM_SAMPLES = 5
NUM_TRIALS = 3

PAD_TOKEN_ID = None


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


def load_model(use_adapter):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    if use_adapter:
        model = PeftModel.from_pretrained(
            model,
            ADAPTER,
            is_trainable=False
        )

    model.eval()
    return model


def benchmark_one_run(name, tokenizer, prompts, use_adapter, trial):
    print(f"\nTrial {trial}: {name}")

    gc.collect()
    torch.cuda.empty_cache()

    synchronize()
    load_start = time.perf_counter()

    model = load_model(use_adapter)

    synchronize()
    load_time = time.perf_counter() - load_start

    # Warm-up excluded from measured generation.
    warmup_inputs = tokenizer(
        prompts[0],
        return_tensors="pt"
    ).to("cuda:0")

    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID
        )

    synchronize()

    # Measure first-token latency on the first prompt.
    ttft_inputs = tokenizer(
        prompts[0],
        return_tensors="pt"
    ).to("cuda:0")

    synchronize()
    ttft_start = time.perf_counter()

    with torch.inference_mode():
        model.generate(
            **ttft_inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=PAD_TOKEN_ID
        )

    synchronize()
    ttft = time.perf_counter() - ttft_start

    latencies = []
    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()

    for index, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda:0")

        input_length = inputs["input_ids"].shape[1]

        synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=PAD_TOKEN_ID
            )

        synchronize()
        elapsed = time.perf_counter() - start

        output_tokens = outputs.shape[1] - input_length

        latencies.append(elapsed)
        total_tokens += output_tokens

        print(
            f"  Sample {index + 1}: "
            f"{output_tokens} tokens, {elapsed:.2f}s"
        )

    total_time = sum(latencies)

    result = {
        "trial": trial,
        "model": name,
        "model_loading_seconds": round(load_time, 3),
        "time_to_first_token_seconds": round(ttft, 4),
        "total_output_tokens": total_tokens,
        "total_generation_seconds": round(total_time, 3),
        "average_latency_seconds": round(
            total_time / len(prompts), 3
        ),
        "generation_tokens_per_second": round(
            total_tokens / total_time, 2
        ) if total_time else 0,
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
    }

    del model
    del warmup_inputs
    del ttft_inputs

    gc.collect()
    torch.cuda.empty_cache()

    return result


def summarize(results):
    keys = [
        "model_loading_seconds",
        "time_to_first_token_seconds",
        "average_latency_seconds",
        "generation_tokens_per_second",
        "peak_gpu_memory_gb"
    ]

    summary = {}

    for model_name, model_results in results.items():
        summary[model_name] = {}

        for key in keys:
            values = [r[key] for r in model_results]

            summary[model_name][key] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(
                    statistics.stdev(values), 4
                ) if len(values) > 1 else 0
            }

    return summary


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)

    df = pd.read_csv(DATASET).fillna("")

    assert len(df) == 4016, "Unexpected test dataset size"

    sample_indices = [0, 100, 500, 1000, 2000]

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    global PAD_TOKEN_ID
    PAD_TOKEN_ID = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    prompts = [
        build_prompt(tokenizer, df.iloc[i]["instruction"])
        for i in sample_indices
    ]

    print("\nBenchmark samples:", sample_indices)
    print("Trials per model:", NUM_TRIALS)

    raw_results = {
        "baseline": [],
        "finetuned": []
    }

    model_configs = {
        "baseline": {
            "name": "Base Qwen2.5-0.5B",
            "use_adapter": False
        },
        "finetuned": {
            "name": "Qwen2.5-0.5B + EXP-001 LoRA",
            "use_adapter": True
        }
    }

    # Alternate execution order to reduce order-related bias.
    execution_orders = [
        ["baseline", "finetuned"],
        ["finetuned", "baseline"],
        ["baseline", "finetuned"]
    ]

    for trial, order in enumerate(execution_orders, start=1):
        print(f"\n========== TRIAL {trial} ==========")

        for model_key in order:
            config = model_configs[model_key]

            result = benchmark_one_run(
                config["name"],
                tokenizer,
                prompts,
                config["use_adapter"],
                trial
            )

            raw_results[model_key].append(result)

    summary = summarize(raw_results)

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "precision": "FP16",
        "max_new_tokens": MAX_NEW_TOKENS,
        "benchmark_samples": NUM_SAMPLES,
        "trials_per_model": NUM_TRIALS,
        "sample_indices": sample_indices,
        "raw_results": raw_results,
        "summary": summary
    }

    OUTPUT.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )

    print("\n========== FINAL BENCHMARK ==========")

    for key, label in [
        ("time_to_first_token_seconds", "TTFT (seconds)"),
        ("average_latency_seconds", "Average latency (seconds)"),
        ("generation_tokens_per_second", "Generation speed (tok/s)"),
        ("peak_gpu_memory_gb", "Peak GPU memory (GB)")
    ]:
        print(f"\n{label}")

        for model_key in ["baseline", "finetuned"]:
            values = summary[model_key][key]

            print(
                f"  {model_key}: "
                f"{values['mean']} +/- {values['std']}"
            )

    print("\nSaved benchmark:", OUTPUT)


if __name__ == "__main__":
    main()
