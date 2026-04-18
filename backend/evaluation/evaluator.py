"""
evaluation/evaluator.py
=======================
Lightweight evaluation pipeline for the Enterprise Knowledge RAG system.

Measures retrieval precision, keyword coverage, confidence sanity, and latency
by calling the *exact same internal functions* used by the live API -- no HTTP
calls, no logic duplication.

Usage
-----
    cd "Enterprise Knowledge RAG"          # project root
    python backend/evaluation/evaluator.py <kb_id>

    # or via environment variable:
    set EVAL_KB_ID=<your-kb-id>
    python backend/evaluation/evaluator.py

How relevant_doc_ids works
--------------------------
Real document IDs in the system are UUIDs.  The test cases use symbolic labels
(e.g. "hr", "wfh") which are matched as case-insensitive substrings of the
chunk's `source` field (filename).  Examples:
    "hr"      matches  "hr_policy.pdf", "Annual_HR_Handbook.txt"
    "wfh"     matches  "wfh_guidelines.pdf", "remote_work_WFH.docx"
    "finance" matches  "Finance_Q3_Report.pdf"

Update the TEST_CASES list to reflect the actual filenames in your KB.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# ── 1. Set CWD to project root BEFORE importing backend modules ───────────────
# settings.INDEX_DIR = "backend/data/indexes" is relative to the project root.
# This must happen before any app import so the path resolves correctly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../Enterprise Knowledge RAG
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

# ── 2. Silence noisy internal loggers so evaluator output stays readable ──────
for _noisy in ("app.kb.manager", "app.retrieval.rerank",
               "app.generation.answer_generator", "api.routes.query",
               "sentence_transformers", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# ── 3. Import production pipeline functions (no duplication) ──────────────────
from app.core.schemas import QueryRequest                   # noqa: E402
from api.routes.query import (                              # noqa: E402
    _retrieve_and_rerank,
    _get_generator,
    _compute_confidence,
    _maybe_rewrite_query,
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST DATASET
# ─────────────────────────────────────────────────────────────────────────────
# relevant_doc_ids  -- case-insensitive substrings of the chunk's source filename
# expected_keywords -- words that should appear in the generated answer

TEST_CASES = [
    {
        "query": "What is the annual leave policy?",
        "expected_keywords": ["leave", "days", "policy"],
        "relevant_doc_ids": ["hr"],          # matches any file with "hr" in name
    },
    {
        "query": "Work from home rules",
        "expected_keywords": ["home", "remote"],
        "relevant_doc_ids": ["wfh", "remote"],  # matches wfh or remote work files
    },
]

# ── Evaluation parameters ─────────────────────────────────────────────────────
TOP_K = 3          # chunks passed to reranker
ANSWER_PREVIEW = 120   # characters shown in output


# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(reranked: list, relevant_doc_ids: list[str]) -> tuple[float, int, int]:
    """
    Fraction of retrieved chunks whose source filename matches at least one
    entry in relevant_doc_ids (case-insensitive substring match).

    Returns (precision, relevant_count, total_count).
    """
    if not reranked:
        return 0.0, 0, 0
    relevant = 0
    for chunk in reranked:
        source = chunk.get("source", "").lower()
        if any(rid.lower() in source for rid in relevant_doc_ids):
            relevant += 1
    total = len(reranked)
    return round(relevant / total, 3), relevant, total


def keyword_coverage(answer: str, expected_keywords: list[str]) -> tuple[float, int, int]:
    """
    Fraction of expected_keywords that appear (case-insensitive) in the answer.

    Returns (coverage, matched_count, total_count).
    """
    if not expected_keywords:
        return 1.0, 0, 0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(matched / len(expected_keywords), 3), matched, len(expected_keywords)


def confidence_warning(confidence: float, kw_cov: float) -> Optional[str]:
    """
    Flag anomalous confidence / coverage combinations.
    Returns a warning string or None.
    """
    if confidence > 0.7 and kw_cov < 0.3:
        return "HIGH confidence but LOW keyword coverage -- answer may be hallucinated"
    if confidence < 0.3 and kw_cov > 0.7:
        return "LOW confidence but HIGH keyword coverage -- answer may be correct but retrieval scored poorly"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_W = 66   # output column width

def _header(text: str) -> None:
    print("\n" + "=" * _W)
    print(f"  {text}")
    print("=" * _W)

def _divider() -> None:
    print("-" * _W)

def _row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(kb_id: str) -> None:
    _header(f"RAG EVALUATION PIPELINE")
    _row("KB ID", kb_id)
    _row("Test cases", str(len(TEST_CASES)))
    _row("top_k", str(TOP_K))
    _row("Project root", str(_PROJECT_ROOT))
    _divider()

    results_summary = []

    for idx, case in enumerate(TEST_CASES, start=1):
        query   = case["query"]
        exp_kws = case["expected_keywords"]
        rel_ids = case["relevant_doc_ids"]

        print(f"\n[{idx}/{len(TEST_CASES)}] {query}")
        _divider()

        # ── Retrieval ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        retrieval_query = _maybe_rewrite_query(query)
        req = QueryRequest(query=query, top_k=TOP_K)

        try:
            stage_results, reranked = _retrieve_and_rerank(
                kb_id, query, retrieval_query, req
            )
        except ValueError as exc:
            _row("ERROR", f"Retrieval failed -- {exc}")
            _row("Hint", "Is the KB indexed? Does it contain documents?")
            results_summary.append({
                "query": query, "precision": 0.0,
                "kw_coverage": 0.0, "latency_ms": 0,
                "confidence": 0.0, "conf_warning": None, "retrieval_error": True,
            })
            continue
        except Exception as exc:
            _row("ERROR", f"Unexpected retrieval error -- {exc}")
            results_summary.append({
                "query": query, "precision": 0.0,
                "kw_coverage": 0.0, "latency_ms": 0,
                "confidence": 0.0, "conf_warning": None, "retrieval_error": True,
            })
            continue

        t_retrieval_ms = int((time.perf_counter() - t0) * 1000)

        # ── Generation ─────────────────────────────────────────────────────
        answer = ""
        try:
            context = [c["text"] for c in reranked]
            answer = _get_generator().generate(query=query, retrieved_chunks=context)
        except Exception as exc:
            answer = f"[generation error: {exc}]"

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # ── Metrics ────────────────────────────────────────────────────────
        confidence               = _compute_confidence(reranked, answer)
        prec, rel_n, total_n     = precision_at_k(reranked, rel_ids)
        kw_cov, kw_n, kw_total   = keyword_coverage(answer, exp_kws)
        warning = confidence_warning(confidence, kw_cov)

        # Retrieved doc filenames (deduplicated, order-preserving)
        seen: set = set()
        retrieved_sources = []
        for c in reranked:
            src = c.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                retrieved_sources.append(src)

        # ── Per-case output ────────────────────────────────────────────────
        preview = (answer[:ANSWER_PREVIEW] + "...") if len(answer) > ANSWER_PREVIEW else answer
        _row("Answer (preview)", preview)
        _row("Confidence", f"{confidence:.3f}")
        _row("Latency", f"{latency_ms} ms  (retrieval: {t_retrieval_ms} ms)")
        _row("Retrieved sources", ", ".join(retrieved_sources) or "none")
        print()
        _row("Precision@k",
             f"{prec:.3f}  ({rel_n}/{total_n} chunks from relevant docs)")
        _row("Keyword coverage",
             f"{kw_cov:.3f}  ({kw_n}/{kw_total} found: {', '.join(exp_kws)})")
        _row("Confidence warning", warning if warning else "none")

        results_summary.append({
            "query":           query,
            "precision":       prec,
            "kw_coverage":     kw_cov,
            "latency_ms":      latency_ms,
            "confidence":      confidence,
            "conf_warning":    warning,
            "retrieval_error": False,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results_summary:
        _header("SUMMARY -- no results collected")
        return

    n = len(results_summary)
    avg_precision  = round(sum(r["precision"]   for r in results_summary) / n, 3)
    avg_kw_cov     = round(sum(r["kw_coverage"] for r in results_summary) / n, 3)
    avg_latency    = round(sum(r["latency_ms"]  for r in results_summary) / n, 1)
    # Confidence warnings = anomalous confidence/coverage pairs on successful retrievals
    # Retrieval errors   = cases where the KB had no matching chunks at all
    conf_warnings_count     = sum(1 for r in results_summary if r["conf_warning"])
    retrieval_errors_count  = sum(1 for r in results_summary if r["retrieval_error"])

    _header("SUMMARY")
    _row("avg_precision",        f"{avg_precision}")
    _row("avg_keyword_coverage", f"{avg_kw_cov}")
    _row("avg_latency_ms",       f"{avg_latency}")
    _row("confidence_warnings",  f"{conf_warnings_count}")
    _row("retrieval_errors",     f"{retrieval_errors_count}  (KB empty or not indexed?)")
    _divider()

    # Machine-readable block for CI/CD parsing
    print("\n  JSON summary:")
    import json
    print("  " + json.dumps({
        "avg_precision":        avg_precision,
        "avg_keyword_coverage": avg_kw_cov,
        "avg_latency_ms":       avg_latency,
        "confidence_warnings":  conf_warnings_count,
        "retrieval_errors":     retrieval_errors_count,
    }, indent=4).replace("\n", "\n  "))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Accept kb_id from CLI arg or environment variable
    kb_id = (sys.argv[1] if len(sys.argv) > 1
             else os.environ.get("EVAL_KB_ID", "").strip())

    if not kb_id:
        print("ERROR: KB ID is required.")
        print()
        print("Usage:")
        print("  python backend/evaluation/evaluator.py <kb_id>")
        print()
        print("  Or set the environment variable:")
        print("  EVAL_KB_ID=<kb_id> python backend/evaluation/evaluator.py")
        sys.exit(1)

    run_evaluation(kb_id)
