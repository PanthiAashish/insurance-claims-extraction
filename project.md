Project: Can Small, Fine-Tuned Language Models Match Large LLMs at Extracting Insurance Claims Data?

Aashish Panthi — Proposal approved September 1, 2026

Introduction

Insurance claims intake requires converting unstructured claim narratives (emails, call transcripts, claim forms) into structured, decision-ready records (policy numbers, incident dates, damage descriptions, injury flags) before a claim can be routed for processing. This project investigates whether a small language model, fine-tuned on domain-specific insurance claims data, can match a much larger general-purpose LLM on this structured extraction task.

The problem matters because LLM inference spending has become one of the largest and fastest-growing technology costs for companies deploying AI at scale, yet most production systems default to routing every request through a frontier-scale model regardless of task complexity. Insurance claims intake is a strong test case because it demands precise, objectively verifiable structured output rather than open-ended judgment, so model performance can be measured directly against ground truth.

Research question: Does parameter-efficient fine-tuning of a small (7-billion-parameter) open-weight model close the performance gap with a frontier-scale API model on field-level claims extraction accuracy, structured-output validity, and inference cost and latency — and if so, by how much?

Detailed Plan

Three-way comparison on the same held-out test set:

Small open-weight base model (Qwen2.5-7B-Instruct, Apache-2.0), no task-specific training
The same model after parameter-efficient fine-tuning (LoRA/QLoRA)
Large general-purpose LLM via API (GPT-4o)

Fine-tuning: LoRA/QLoRA using Hugging Face transformers, peft, and bitsandbytes, with Unsloth as a candidate for faster/lower-memory training if needed. Base model quantized to 4-bit for training. Training on Google Colab (Pro tier, A100/T4), with a request for supplemental GPU access through Fisk University if needed.

Training data: Sourced from publicly available insurance claims datasets where they exist. Since real claim narratives aren't publicly available at fine-tuning volume (privacy restrictions), synthetic claim narratives will be generated/augmented via LLM prompting, structured to mirror realistic field distributions and edge cases (missing fields, ambiguous incident dates, multi-claimant narratives), without overlapping the evaluation set.

Fallback: If sufficient insurance claims data and a matching benchmark can't be assembled, pivot to a comparable structured-extraction domain (e.g., medical intake notes or legal contract clause extraction) with the same properties, keeping the three-way comparison design unchanged.

Evaluation: Cleanlab's publicly released Insurance Claims Extraction benchmark — an independently curated, fixed test set none of the models are trained on. Metrics: field-level exact-match accuracy, JSON schema validity rate, precision/recall/F1 per field type, and inference cost and latency per 1,000 requests for each condition.

Success criterion: How much of the accuracy/validity gap between the base small model and GPT-4o is closed by fine-tuning, weighed against GPT-4o's cost/latency advantage or disadvantage per request.

Deliverables: Public code repository (fine-tuning pipeline, evaluation harness, synthetic data generation scripts), a written report with full metrics tables and analysis, and the fine-tuned model checkpoint shared (e.g., via Hugging Face Hub) for independent reproducibility.

If time permits: Measure how extraction accuracy scales with fine-tuning data size; build a live demo/comparison dashboard; add a downstream claim acceptance/rejection classification stage on top of the extracted fields.
