# Insurance Claims Extraction

Evaluating whether a parameter-efficient fine-tuned small language model (Qwen2.5-7B-Instruct) can match a frontier-scale model (GPT-4o) on structured insurance claims extraction. Full project proposal: [`project.md`](project.md).

## Pipeline

Run in order (step 4 requires a GPU runtime, e.g. Google Colab):

1. **`explore_dataset.py`** — Loads the Cleanlab `insurance-claims-extraction` benchmark, confirms dataset size and schema, and audits field coverage across examples.
2. **`baseline_gpt4o.py`** — Runs zero-shot extraction with GPT-4o on the full benchmark. Requires `OPENAI_API_KEY` set in the environment.
3. **`baseline_qwen.py`** — Runs zero-shot extraction with Qwen2.5-7B-Instruct (4-bit quantized), prior to any fine-tuning. See `qwen_baseline_colab.ipynb` for a ready-to-run Colab notebook.
4. **`run_eval.py`** — Computes field-level precision, recall, and F1 (with bootstrap confidence intervals), JSON validity rate, and latency for each model, and writes the results table.

Dependencies: `datasets`, `openai`, `transformers`, `accelerate`, `bitsandbytes`, `torch`.

## Results (zero-shot baselines)

| Metric | GPT-4o | Qwen2.5-7B-Instruct |
|---|---|---|
| JSON validity | 100.0% | 96.7% |
| Field precision | 0.649 | 0.594 |
| Field recall | 0.537 | 0.470 |
| Field F1 (95% CI) | 0.588 [0.543, 0.626] | 0.525 [0.478, 0.570] |
| Avg. latency / request | 4.90s | 27.91s |

Fine-tuning results will be added once the LoRA training stage is complete.

## Repository structure

```
project.md                   Full project proposal
score_utils.py                Field-level scoring against ground truth
explore_dataset.py             Benchmark loading and schema audit
baseline_gpt4o.py              Zero-shot GPT-4o baseline
baseline_qwen.py               Zero-shot Qwen2.5-7B baseline
qwen_baseline_colab.ipynb      Colab notebook for the Qwen baseline
run_eval.py                    Evaluation and results table generation
artifacts/                     Saved datasets and per-model results
```

## Methodology notes

- **Sample size.** The Cleanlab benchmark's `train` split is its only split, containing 30 examples. All point estimates are reported alongside 95% bootstrap confidence intervals to reflect this.
- **Null handling.** Several ground-truth examples have entire sections (`policy_details`, `insured_objects`) set to `null`. The scorer treats a correctly predicted absence as a true negative, a hallucinated value where the truth is null as a false positive, and a missed value as a false negative.
- **Pricing.** Token-based cost comparisons should use current API pricing at the time of analysis, as rates are subject to change.
