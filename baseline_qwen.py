"""
Step 2b. Zero-shot Qwen2.5-7B-Instruct baseline, run in Colab with a
GPU runtime (A100 or T4). This is the "before fine-tuning" arm --
you need this number to later show what LoRA actually bought you.

pip install: transformers accelerate bitsandbytes torch
"""
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """You extract structured data from insurance claim narratives.
Given a claim text, output ONLY a JSON object matching this shape (use null for
anything not present in the text -- do not guess or fabricate values):

{
  "header": {"claim_id": ..., "report_date": ..., "incident_date": ..., "reported_by": ..., "channel": ...},
  "policy_details": {"policy_number": ..., "policyholder_name": ..., "coverage_type": ..., "effective_date": ..., "expiration_date": ...} or null,
  "insured_objects": [...] or null,
  "incident_description": {"incident_type": ..., "location_type": ..., "estimated_damage_amount": ...} or null
}

Output raw JSON only, no markdown fences, no commentary."""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)


def extract(claim_text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": claim_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=False, temperature=None, top_p=None
        )
    latency = time.time() - start

    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    return {
        "prediction": parsed,
        "latency_s": round(latency, 3),
        "completion_tokens": len(gen_tokens),
        "raw_valid_json": bool(parsed),
        "raw_output": raw,
    }


if __name__ == "__main__":
    with open("artifacts/dataset_full.json") as f:
        examples = json.load(f)

    results = []
    for i, ex in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] extracting...")
        r = extract(ex["claim_text"])
        r["ground_truth"] = ex["ground_truth"]
        results.append(r)

    with open("artifacts/qwen_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    valid_rate = sum(r["raw_valid_json"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print(f"JSON validity rate: {valid_rate:.2%}")
    print(f"Avg latency/request: {avg_latency:.2f}s")
