"""
Field-level scoring for nested JSON extraction against the Cleanlab
insurance-claims-extraction ground truth format.

Design choice: we do NOT hardcode the schema (header/policy_details/...)
because the exact leaf-field set wasn't fully visible before actually
loading the dataset. Instead we flatten both the prediction and the
ground truth into dotted-path -> value dicts, then compare key-by-key.
This is robust to whatever the real schema turns out to be once you
pull the dataset in Colab, and it still gives clean field-level P/R/F1.

Null-handling matters here: several ground-truth rows have entire
sub-objects (e.g. policy_details) set to None. We treat "predicted
None where truth is None" as a correct field, "predicted a value
where truth is None" as a false positive (hallucinated field), and
"predicted None where truth has a value" as a false negative (missed
field), same as a wrong-value mismatch.
"""

from __future__ import annotations
import ast
import json
from typing import Any


def parse_ground_truth(raw: str) -> dict:
    """The HF viewer shows ground_truth as a Python-dict-literal string
    (single quotes, None instead of null). ast.literal_eval handles
    that; json.loads will fail on it, so don't use json.loads here."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return json.loads(raw)  # fallback if it's actually valid JSON


def flatten(obj: Any, prefix: str = "") -> dict:
    """Flatten nested dicts/lists into {dotted.path: value}.
    Lists are flattened by index: insured_objects.0.type
    A None at a branch point collapses to a single leaf so a totally
    missing sub-object is one field, not zero."""
    flat = {}
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            flat.update(flatten(v, path))
    elif isinstance(obj, list) and obj:
        for i, v in enumerate(obj):
            path = f"{prefix}.{i}"
            flat.update(flatten(v, path))
    else:
        # leaf: empty dict, empty list, None, or a scalar
        flat[prefix or "root"] = obj if obj not in ({}, []) else None
    return flat


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower()
    return v


def score_example(pred: dict, truth: dict) -> dict:
    """Returns per-field-key comparison plus aggregate counts for one example."""
    flat_pred = flatten(pred)
    flat_truth = flatten(truth)
    all_keys = set(flat_pred) | set(flat_truth)

    tp = fp = fn = tn = 0
    mismatches = []

    for key in all_keys:
        p = normalize_value(flat_pred.get(key))
        t = normalize_value(flat_truth.get(key))
        if t is None and p is None:
            tn += 1
        elif t is None and p is not None:
            fp += 1
            mismatches.append((key, "hallucinated", p, t))
        elif t is not None and p is None:
            fn += 1
            mismatches.append((key, "missed", p, t))
        elif p == t:
            tp += 1
        else:
            fn += 1  # wrong value counts against recall of the correct field
            fp += 1  # and as a spurious wrong value
            mismatches.append((key, "wrong_value", p, t))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "mismatches": mismatches,
            "exact_match": len(mismatches) == 0}


def aggregate(example_scores: list[dict]) -> dict:
    tp = sum(s["tp"] for s in example_scores)
    fp = sum(s["fp"] for s in example_scores)
    fn = sum(s["fn"] for s in example_scores)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact_match_rate = sum(s["exact_match"] for s in example_scores) / len(example_scores)
    return {
        "n": len(example_scores),
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "output_exact_match_rate": round(exact_match_rate, 4),
    }


def bootstrap_ci(example_scores: list[dict], n_boot: int = 2000, seed: int = 0):
    """95% CI on field_f1 via bootstrap resampling of the 30 examples.
    With n=30, report this alongside every point estimate -- point
    estimates alone are misleading at this sample size."""
    import random
    rng = random.Random(seed)
    n = len(example_scores)
    f1s = []
    for _ in range(n_boot):
        sample = [example_scores[rng.randrange(n)] for _ in range(n)]
        f1s.append(aggregate(sample)["field_f1"])
    f1s.sort()
    lo = f1s[int(0.025 * n_boot)]
    hi = f1s[int(0.975 * n_boot)]
    return round(lo, 4), round(hi, 4)
