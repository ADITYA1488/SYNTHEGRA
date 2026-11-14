import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parent

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = PROJECT_DIR / "adapter"

print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

print("Loading base model in FP16...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True
)

print("Attaching EXP-001 adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
    is_trainable=False
)

model.eval()

print("\nModel loaded successfully!")
print("GPU memory allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

messages = [
    {
        "role": "system",
        "content": "You are an enterprise customer support assistant. Identify the customer's category and intent and generate a helpful response. Return valid JSON containing exactly category, intent, response. Do not invent customer-specific details. Preserve placeholders."
    },
    {
        "role": "user",
        "content": "I want to cancel my subscription and get a refund."
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("\nGenerating test response...")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print("\nEXP-001 TEST OUTPUT:\n")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
