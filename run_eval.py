"""
Step 3. Run after both baseline_gpt4o.py and baseline_qwen.py have
produced their results.json files. Produces the actual numbers table
for the report -- this is your "Evidence of Progress" section.
"""
import json
from score_utils import score_example, aggregate, bootstrap_ci

def evaluate(path: str, label: str):
    with open(path) as f:
        results = json.load(f)
    scores = [score_example(r["prediction"], r["ground_truth"]) for r in results]
    agg = aggregate(scores)
    lo, hi = bootstrap_ci(scores)
    valid_rate = sum(r["raw_valid_json"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print(f"\n=== {label} ===")
    print(f"n = {agg['n']}")
    print(f"JSON validity rate: {valid_rate:.2%}")
    print(f"Field-level precision: {agg['field_precision']:.3f}")
    print(f"Field-level recall:    {agg['field_recall']:.3f}")
    print(f"Field-level F1:        {agg['field_f1']:.3f}  (95% CI: [{lo}, {hi}])")
    print(f"Output exact-match rate: {agg['output_exact_match_rate']:.2%}")
    print(f"Avg latency/request: {avg_latency:.2f}s")
    return {"label": label, **agg, "f1_ci": [lo, hi], "json_validity": round(valid_rate, 4),
            "avg_latency_s": round(avg_latency, 3)}


if __name__ == "__main__":
    rows = []
    rows.append(evaluate("artifacts/gpt4o_baseline_results.json", "GPT-4o (zero-shot)"))
    rows.append(evaluate("artifacts/qwen_baseline_results.json", "Qwen2.5-7B-Instruct (zero-shot, no fine-tuning)"))

    with open("artifacts/results_table.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved artifacts/results_table.json")
