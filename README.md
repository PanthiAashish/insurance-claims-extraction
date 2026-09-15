# Insurance Claims Extraction — Baseline Pipeline

Run this in order, in a Colab notebook (GPU runtime for step 2b):

1. `pip install datasets openai transformers accelerate bitsandbytes torch`
2. `python explore_dataset.py`
   - Confirms the real Cleanlab benchmark loads, prints n and column names,
     audits how often each top-level section (header/policy_details/
     insured_objects/incident_description) is non-null across the 30 rows.
   - Screenshot this terminal output for the report's Evidence section.
3. `OPENAI_API_KEY=sk-... python baseline_gpt4o.py`
   - Zero-shot GPT-4o on all 30 examples. Records latency and token counts
     per request so the cost/latency comparison in your proposal has real
     numbers behind it.
4. `python baseline_qwen.py` (GPU runtime)
   - Zero-shot Qwen2.5-7B-Instruct, 4-bit, no fine-tuning yet. This is the
     "before" number that fine-tuning has to beat to mean anything.
5. `python run_eval.py`
   - Field-level precision/recall/F1 with 95% bootstrap CIs (necessary
     given n=30), JSON validity rate, output exact-match rate, latency,
     for both models side by side.

## Known issues to write up honestly in the report

- **n=30.** The Cleanlab benchmark's `train` split is the only split and
  has 30 rows. Point estimates on 30 examples are noisy -- report the
  bootstrap CI alongside every F1 number, not just the point estimate.
- **Schema fields aren't fully nailed down from the HF preview** (some
  sub-fields were truncated in the viewer). `explore_dataset.py`'s section
  audit is how you actually confirm the schema once you load the parquet
  yourself -- don't hardcode field names before that.
- **Null ground truth is common**: several examples have `policy_details`
  or `insured_objects` set to `None` entirely. The scorer in
  `score_utils.py` treats "correctly predicting absence" as a true
  negative, "hallucinating a value where truth is null" as a false
  positive, and "missing a value that exists" as a false negative --
  say this explicitly in the report so a null-heavy benchmark doesn't
  make your metrics look inflated for free.
- GPT-4o's per-token pricing changes — pull the current rate from
  OpenAI's pricing page at report-writing time rather than hardcoding one.
