"""
Step 2a. Zero-shot GPT-4o baseline on the 30-example benchmark.
Requires OPENAI_API_KEY set in the environment. This is the frontier
reference point your whole cost/accuracy comparison is anchored to,
so get real numbers from it before touching Qwen or fine-tuning.
"""
import json
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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


def extract(claim_text: str) -> dict:
    start = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": claim_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    latency = time.time() - start
    raw = resp.choices[0].message.content
    usage = resp.usage
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "prediction": parsed,
        "latency_s": round(latency, 3),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "raw_valid_json": bool(parsed),
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

    with open("artifacts/gpt4o_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    valid_rate = sum(r["raw_valid_json"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results)
    print(f"JSON validity rate: {valid_rate:.2%}")
    print(f"Avg latency/request: {avg_latency:.2f}s")
    print(f"Total tokens across {len(results)} requests: {total_tokens}")
    # GPT-4o pricing changes -- pull current $/1M tokens from OpenAI's pricing
    # page at report-writing time rather than hardcoding a rate here.
