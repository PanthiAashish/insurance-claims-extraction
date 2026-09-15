"""
Step 1 to run in Colab. Pulls the real, fixed eval set and confirms
what we're actually working with before anything else happens.
Verified once already (30 rows, train split only) -- run this yourself
too so you have your own timestamped evidence for the report.
"""
from datasets import load_dataset
import json
from score_utils import parse_ground_truth

ds = load_dataset("Cleanlab/insurance-claims-extraction", split="train")
print(f"n_examples = {len(ds)}")
print(f"columns = {ds.column_names}")

examples = []
for row in ds:
    gt = parse_ground_truth(row["ground_truth"])
    examples.append({"claim_text": row["claim_text"], "ground_truth": gt})

# quick field-coverage audit: how often is each top-level section present vs None
from collections import Counter
section_counts = Counter()
for ex in examples:
    for section, val in ex["ground_truth"].items():
        section_counts[section] += 0 if val is None else 1
print("Top-level section non-null counts (out of", len(examples), "):")
for k, v in section_counts.items():
    print(f"  {k}: {v}")

with open("artifacts/dataset_full.json", "w") as f:
    json.dump(examples, f, indent=2, default=str)

print("Saved artifacts/dataset_full.json -- this is your evidence the benchmark")
print("is real, pulled, and parsed correctly. Screenshot this output for the report.")
