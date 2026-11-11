#!/usr/bin/env python3
"""Compute evaluation metrics from generated reports vs. the ground-truth manifest.

Usage:
    python scripts/eval_metrics.py --reports ./reports --dataset ./sample_submissions

Reads similarity_matrix.json + per_document reports + dataset_manifest.json and
prints/writes the confusion data used in EVALUATION.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(reports_dir: Path, dataset_dir: Path) -> dict:
    matrix = json.loads((reports_dir / "similarity_matrix.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    docs = {}
    for f in (reports_dir / "per_document").glob("*.json"):
        rep = json.loads(f.read_text(encoding="utf-8"))
        docs[rep["document_id"]] = rep
    return {"matrix": matrix, "manifest": manifest, "docs": docs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports"))
    ap.add_argument("--dataset", default=str(ROOT / "sample_submissions"))
    ap.add_argument("--pair_threshold", type=float, default=20.0,
                    help="Pair flagged when max(coverage either direction) >= this %%")
    ap.add_argument("--out", default=str(ROOT / "reports" / "evaluation_metrics.json"))
    args = ap.parse_args()

    data = load(Path(args.reports), Path(args.dataset))
    matrix, manifest, docs = data["matrix"]["matrix"], data["manifest"], data["docs"]
    thr = args.pair_threshold

    ground_truth_pairs = {
        (r["query"], r["source"])
        for r in manifest["plagiarism_relations"]
        if r["signal"].startswith("peer")
    }
    gt_norm = {frozenset(p) for p in ground_truth_pairs}

    all_files = sorted(matrix.keys())
    # Files with no extractable text cannot participate in pairwise evaluation.
    evaluable = [f for f in all_files if docs.get(f, {}).get("processing", {}).get("chunks", 0) > 0]

    tp, fp, fn, tn = [], [], [], []
    fp_details = {}
    for a, b in itertools.combinations(evaluable, 2):
        cov = max(matrix.get(a, {}).get(b, 0.0), matrix.get(b, {}).get(a, 0.0))
        detected = cov >= thr
        is_gt = frozenset((a, b)) in gt_norm
        if detected and is_gt:
            tp.append((a, b, cov))
        elif detected and not is_gt:
            fp.append((a, b, cov))
            fp_details[f"{a} <-> {b}"] = cov
        elif not detected and is_gt:
            fn.append((a, b, cov))
        else:
            tn.append((a, b))

    n_pos, n_neg = len(tp) + len(fn), len(fp) + len(tn)
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0.0
    recall = len(tp) / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # AI detection: ground-truth AI doc should rank #1 by AI probability.
    ai_files = [f for f, d in docs.items() if d["scores"]["ai_generated_probability"] > 0]
    ai_rank = sorted(
        ai_files, key=lambda f: docs[f]["scores"]["ai_generated_probability"], reverse=True
    )
    ai_gt = manifest["ai_generated"][0]["file"]
    ai_top = ai_rank[0] if ai_rank else None

    # External detection.
    ext_gt = next(
        r for r in manifest["plagiarism_relations"] if r["signal"] == "external"
    )
    ext_doc = ext_gt["query"]
    ext_score = docs[ext_doc]["scores"]["external_similarity_percentage"]
    ext_detected = ext_score >= 50.0
    ext_others = {
        f: d["scores"]["external_similarity_percentage"]
        for f, d in docs.items()
        if f != ext_doc and d["scores"]["external_similarity_percentage"] > 0
    }

    # Degraded handling.
    degraded = {g["file"]: docs.get(g["file"], {}).get("processing", {}) for g in manifest["degraded"]}

    result = {
        "pair_threshold_pct": thr,
        "evaluable_documents": len(evaluable),
        "pairs_evaluated": len(evaluable) * (len(evaluable) - 1) // 2,
        "positives": n_pos,
        "negatives": n_neg,
        "tp": [list(t) for t in tp],
        "fp": [list(t) for t in fp],
        "fn": [list(t) for t in fn],
        "tn_count": len(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_margins": fp_details,
        "ai_detection": {
            "ground_truth": ai_gt,
            "top_scored": ai_top,
            "ranking": [(f, docs[f]["scores"]["ai_generated_probability"]) for f in ai_rank[:5]],
            "correct": ai_top == ai_gt,
            "margin": round(
                docs[ai_gt]["scores"]["ai_generated_probability"]
                - (docs[ai_rank[1]]["scores"]["ai_generated_probability"] if len(ai_rank) > 1 else 0.0),
                1,
            ),
        },
        "external_detection": {
            "ground_truth": ext_doc,
            "score": ext_score,
            "detected_at_50pct": ext_detected,
            "bleed_into_others": ext_others,
        },
        "degraded_handling": degraded,
        "benign_originals_max_pair_cov": {
            f: max(
                [max(matrix.get(f, {}).get(o, 0.0), matrix.get(o, {}).get(f, 0.0))
                 for o in evaluable if o != f and frozenset((f, o)) not in gt_norm] or [0.0]
            )
            for f in manifest["independent_originals"]
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
