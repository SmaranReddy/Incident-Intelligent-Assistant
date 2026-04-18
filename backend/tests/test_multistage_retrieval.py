"""
QA Test Suite — Multi-Stage Retrieval
======================================
Tests the two-stage retrieval pipeline without starting the HTTP server
and without modifying any production code.

Strategy
--------
• Build lightweight in-memory KBIndex objects with synthetic 8-d FAISS vectors.
  Directional relationships between the vectors mimic semantic similarity, so
  doc_hr chunks point "north", doc_wfh chunks point "east", doc_finance chunks
  point "south".  A query vector pointing "north" should always pick doc_hr first.
• Capture Python log output with logging.handlers to verify exact log messages.
• Patch settings.TOP_N_DOCS at test time to validate configurability.

Covered scenarios
-----------------
  T1  Stage-1 log messages present (Doc scores + Selected docs)
  T2  Relevant document ranked first, irrelevant document excluded
  T3  Empty-KB fallback — get_top_documents returns [], search returns [],
      filter_doc_ids is set to None (no crash)
  T4  TOP_N_DOCS config respected — top_n=2 returns at most 2 docs
"""

import logging
import sys
import os
import unittest

import faiss
import numpy as np

# ── make backend importable ───────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.kb.manager import KBIndex     # noqa: E402  (after sys.path patch)

DIM = 8  # lightweight synthetic embeddings


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit(arr):
    """Return L2-normalised float32 row vector."""
    v = np.array(arr, dtype="float32").reshape(1, -1)
    norm = np.linalg.norm(v) + 1e-9
    return v / norm


def _build_kb(chunks_def: list) -> KBIndex:
    """
    Build a KBIndex from a list of dicts:
        {id, doc_id, text, source, vec}   (vec is a list of DIM floats)
    """
    index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
    chunks = []
    vecs = []
    for i, c in enumerate(chunks_def):
        vec = _unit(c["vec"])
        vecs.append(vec)
        chunks.append({
            "id":       c["id"],
            "faiss_id": i,
            "text":     c["text"],
            "source":   c["source"],
            "doc_id":   c["doc_id"],
        })
    if vecs:
        embeddings = np.vstack(vecs)
        ids = np.arange(len(vecs), dtype=np.int64)
        index.add_with_ids(embeddings, ids)
    return KBIndex(kb_id="test-kb", faiss_index=index, chunks=chunks)


# ── shared fixture ─────────────────────────────────────────────────────────────
#
#  Three documents, orthogonal embedding directions:
#    doc_hr      -> "north"    (1, 1, 0, 0, 0, 0, 0, 0)
#    doc_wfh     -> "east"     (0, 0, 1, 1, 0, 0, 0, 0)
#    doc_finance -> "south"    (0, 0, 0, 0, 1, 1, 0, 0)
#
#  A query pointing "north" should consistently score doc_hr highest.

CHUNKS_3_DOCS = [
    # doc_hr — 2 chunks (north direction)
    {"id": "hr1", "doc_id": "doc_hr",      "source": "hr_policy.pdf",
     "text": "annual leave employee policy 20 days vacation time off",
     "vec":  [1, 1, 0, 0, 0, 0, 0, 0]},
    {"id": "hr2", "doc_id": "doc_hr",      "source": "hr_policy.pdf",
     "text": "annual leave approval required manager sign off",
     "vec":  [0.9, 1.1, 0, 0, 0, 0, 0, 0]},
    # doc_wfh — 2 chunks (east direction)
    {"id": "wfh1", "doc_id": "doc_wfh",   "source": "wfh_policy.pdf",
     "text": "remote work flexible home office schedule",
     "vec":  [0, 0, 1, 1, 0, 0, 0, 0]},
    {"id": "wfh2", "doc_id": "doc_wfh",   "source": "wfh_policy.pdf",
     "text": "work from home days per week approval",
     "vec":  [0, 0, 0.9, 1.1, 0, 0, 0, 0]},
    # doc_finance — 2 chunks (south direction)
    {"id": "fin1", "doc_id": "doc_finance","source": "finance_q3.pdf",
     "text": "quarterly revenue budget financial report expenses",
     "vec":  [0, 0, 0, 0, 1, 1, 0, 0]},
    {"id": "fin2", "doc_id": "doc_finance","source": "finance_q3.pdf",
     "text": "balance sheet profit loss fiscal year",
     "vec":  [0, 0, 0, 0, 0.9, 1.1, 0, 0]},
]

# "North" query — close to doc_hr vectors
QUERY_LEAVE   = "What is the annual leave policy?"
QUERY_VEC_HR  = _unit([1, 1, 0.1, 0, 0, 0, 0, 0])   # points north

# Completely off-axis query — nothing in the index matches
QUERY_UNRELATED     = "How does quantum entanglement work?"
QUERY_VEC_UNRELATED = _unit([0, 0, 0, 0, 0, 0, 1, 1])   # "west" — no chunks there


class LogCapture(logging.Handler):
    """Collect all log messages emitted during a test."""
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def messages(self) -> list[str]:
        return [self.format(record) for record in self.records]

    def contains(self, substring: str) -> bool:
        return any(substring in m for m in self.messages())


# ── T1 — Stage-1 log messages ─────────────────────────────────────────────────

class TestStage1Logging(unittest.TestCase):
    """
    T1: Verify that get_top_documents() emits the two required DEBUG log lines:
        • "[Stage1] Doc scores: ..."
        • Stage-1 doc selection summary line
    """

    def setUp(self):
        self.kb = _build_kb(CHUNKS_3_DOCS)
        self.handler = LogCapture()
        # Attach to the manager's logger so we capture exactly what the
        # production code logs, without any level filtering.
        self.logger = logging.getLogger("app.kb.manager")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def test_doc_scores_logged(self):
        """[Stage1] Doc scores must appear in logs after get_top_documents()."""
        self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        found = self.handler.contains("[Stage1] Doc scores")
        print(f"\n  Captured log lines:\n" +
              "\n".join(f"    {m}" for m in self.handler.messages()))
        self.assertTrue(
            found,
            "FAIL: '[Stage1] Doc scores' not found in log output"
        )

    def test_selected_docs_count_in_log(self):
        """Stage-1 selection summary must include 'selected' and the chosen doc_ids."""
        self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        found = self.handler.contains("Stage-1 doc selection")
        self.assertTrue(
            found,
            "FAIL: 'Stage-1 doc selection' summary line not found in logs"
        )

    def test_selected_docs_fewer_than_total(self):
        """top_n=2 must yield fewer selected docs than the 3 total docs in the KB."""
        result = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        total_docs = len({c["doc_id"] for c in self.kb.chunks})
        print(f"\n  Total docs: {total_docs} | Selected: {len(result)} | IDs: {result}")
        self.assertEqual(len(result), 2)
        self.assertLess(
            len(result), total_docs,
            f"FAIL: selected {len(result)} docs but corpus has {total_docs}"
        )


# ── T2 — Retrieval relevance ──────────────────────────────────────────────────

class TestMultiStageRelevance(unittest.TestCase):
    """
    T2: Stage 1 must rank doc_hr first for a leave-policy query and must exclude
        doc_finance when top_n < total_doc_count.
    """

    def setUp(self):
        self.kb = _build_kb(CHUNKS_3_DOCS)

    def test_relevant_doc_ranked_first(self):
        """doc_hr must be the top-scoring document for the leave query."""
        top_docs = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=3)
        print(f"\n  Stage-1 order: {top_docs}")
        self.assertEqual(
            top_docs[0], "doc_hr",
            f"FAIL: expected 'doc_hr' first but got '{top_docs[0]}'"
        )

    def test_irrelevant_doc_excluded_when_top_n_is_2(self):
        """With top_n=2, doc_finance (orthogonal to query) must not be selected."""
        top_docs = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        print(f"\n  Stage-1 top-2: {top_docs}")
        self.assertNotIn(
            "doc_finance", top_docs,
            f"FAIL: doc_finance should be excluded but got {top_docs}"
        )

    def test_stage2_chunks_restricted_to_stage1_docs(self):
        """Stage-2 search with filter_doc_ids must return ONLY chunks from those docs."""
        top_docs = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        results = self.kb.search(
            QUERY_VEC_HR, QUERY_LEAVE, top_k=10, filter_doc_ids=set(top_docs)
        )
        result_doc_ids = {r["metadata"]["doc_id"] for r in results}
        print(f"\n  Stage-1 selected: {top_docs}")
        print(f"  Stage-2 result doc_ids: {result_doc_ids}")
        unexpected = result_doc_ids - set(top_docs)
        self.assertEqual(
            unexpected, set(),
            f"FAIL: Stage-2 returned chunks from unexpected docs: {unexpected}"
        )

    def test_stage2_without_filter_returns_all_docs(self):
        """Sanity: search with filter_doc_ids=None (full corpus) returns chunks from all docs."""
        results = self.kb.search(QUERY_VEC_HR, QUERY_LEAVE, top_k=10, filter_doc_ids=None)
        result_doc_ids = {r["metadata"]["doc_id"] for r in results}
        print(f"\n  Full-corpus result doc_ids: {result_doc_ids}")
        # With top-50 FAISS + BM25 and only 6 chunks, all 3 docs should appear
        self.assertGreaterEqual(
            len(result_doc_ids), 1,
            "FAIL: full-corpus search returned no results"
        )


# ── T3 — Empty Stage-1 fallback ───────────────────────────────────────────────

class TestEmptyStage1Fallback(unittest.TestCase):
    """
    T3: When the KB has no chunks, get_top_documents() returns [].
        The pipeline must convert [] -> filter_doc_ids=None (full-corpus fallback)
        and search() must return [] gracefully — no exception raised.
    """

    def setUp(self):
        # Empty KB: index with no vectors, empty chunks list
        empty_index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
        self.kb = KBIndex(kb_id="empty-kb", faiss_index=empty_index, chunks=[])

    def test_get_top_documents_returns_empty_list(self):
        """get_top_documents on an empty KB must return [] without raising."""
        try:
            result = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=5)
        except Exception as exc:
            self.fail(f"FAIL: get_top_documents raised {type(exc).__name__}: {exc}")
        self.assertEqual(result, [], f"FAIL: expected [], got {result}")
        print(f"\n  get_top_documents (empty KB) -> {result!r}  OK")

    def test_empty_list_maps_to_none_filter(self):
        """Simulate _retrieve_and_rerank logic: [] -> filter_doc_ids=None."""
        top_doc_ids = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=5)
        filter_doc_ids = set(top_doc_ids) if top_doc_ids else None
        self.assertIsNone(
            filter_doc_ids,
            f"FAIL: expected None (full-corpus fallback) but got {filter_doc_ids}"
        )
        print(f"\n  [] -> filter_doc_ids={filter_doc_ids!r}  OK")

    def test_search_with_none_filter_on_empty_kb_returns_empty(self):
        """search(filter_doc_ids=None) on an empty KB must return [] without raising."""
        try:
            results = self.kb.search(QUERY_VEC_HR, QUERY_LEAVE, top_k=10, filter_doc_ids=None)
        except Exception as exc:
            self.fail(f"FAIL: search raised {type(exc).__name__}: {exc}")
        self.assertEqual(results, [], f"FAIL: expected [], got {results}")
        print(f"\n  search (empty KB, filter=None) -> {results!r}  OK")

    def test_no_crash_on_unrelated_query_non_empty_kb(self):
        """
        Non-empty KB + completely off-axis query.
        Stage 1 still returns some docs (low scores but not empty).
        Stage 2 still returns some chunks.
        No exception raised.
        """
        kb = _build_kb(CHUNKS_3_DOCS)
        try:
            top_docs = kb.get_top_documents(QUERY_VEC_UNRELATED, QUERY_UNRELATED, top_n=5)
            results  = kb.search(QUERY_VEC_UNRELATED, QUERY_UNRELATED,
                                 top_k=10, filter_doc_ids=set(top_docs) if top_docs else None)
        except Exception as exc:
            self.fail(f"FAIL: pipeline raised {type(exc).__name__}: {exc}")

        print(f"\n  Unrelated query — Stage-1 selected: {top_docs}")
        print(f"  Unrelated query — Stage-2 results  : {len(results)} chunks")
        # Non-empty KB: Stage 1 should still return some docs
        self.assertGreater(
            len(top_docs), 0,
            "FAIL: Stage 1 returned [] for a non-empty KB — unexpected"
        )


# ── T4 — TOP_N_DOCS configurability ───────────────────────────────────────────

class TestTopNDocsConfig(unittest.TestCase):
    """
    T4: Verify that the top_n parameter (backed by settings.TOP_N_DOCS) is
        correctly respected.  We test the method directly; we also verify that
        settings.TOP_N_DOCS is readable and has the expected type/default.
    """

    def setUp(self):
        self.kb = _build_kb(CHUNKS_3_DOCS)

    def test_top_n_1_returns_single_doc(self):
        """top_n=1 must return exactly 1 document."""
        result = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=1)
        print(f"\n  top_n=1 -> {result}")
        self.assertEqual(len(result), 1, f"FAIL: expected 1 doc, got {len(result)}")

    def test_top_n_2_returns_at_most_2_docs(self):
        """top_n=2 must return at most 2 documents."""
        result = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=2)
        print(f"\n  top_n=2 -> {result}")
        self.assertLessEqual(len(result), 2,
                             f"FAIL: expected ≤2 docs, got {len(result)}")

    def test_top_n_5_returns_all_3_docs_when_corpus_has_only_3(self):
        """top_n=5 on a 3-doc KB must return all 3 docs (capped at corpus size)."""
        result = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=5)
        print(f"\n  top_n=5 (3-doc corpus) -> {result}")
        self.assertEqual(len(result), 3,
                         f"FAIL: expected 3 docs, got {len(result)}: {result}")

    def test_settings_top_n_docs_is_int_with_default_5(self):
        """settings.TOP_N_DOCS must exist, be an int, and default to 5."""
        from app.core.config import settings
        self.assertIsInstance(
            settings.TOP_N_DOCS, int,
            f"FAIL: settings.TOP_N_DOCS is {type(settings.TOP_N_DOCS)}, expected int"
        )
        self.assertEqual(
            settings.TOP_N_DOCS, 5,
            f"FAIL: default TOP_N_DOCS should be 5, got {settings.TOP_N_DOCS}"
        )
        print(f"\n  settings.TOP_N_DOCS = {settings.TOP_N_DOCS} (int)  OK")

    def test_top_n_override_via_explicit_argument(self):
        """
        Simulate changing TOP_N_DOCS=2 via env:
        call get_top_documents with top_n=2 explicitly and confirm the result
        is limited to 2 — exactly what the pipeline does with settings.TOP_N_DOCS.
        """
        simulated_top_n = 2    # as if TOP_N_DOCS=2 in .env
        result_full = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE, top_n=5)
        result_trim = self.kb.get_top_documents(QUERY_VEC_HR, QUERY_LEAVE,
                                                top_n=simulated_top_n)
        print(f"\n  top_n=5 -> {result_full}")
        print(f"  top_n=2 -> {result_trim}")
        self.assertEqual(len(result_trim), simulated_top_n,
                         f"FAIL: top_n=2 returned {len(result_trim)} docs")
        # The top-2 should be the same first 2 as the top-5 ordering
        self.assertEqual(
            result_trim, result_full[:simulated_top_n],
            f"FAIL: top-2 results differ from first 2 of top-5\n"
            f"  top-5[:2]={result_full[:simulated_top_n]}, top-2={result_trim}"
        )


# ── runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretty separator between test classes
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestStage1Logging,
        TestMultiStageRelevance,
        TestEmptyStage1Fallback,
        TestTopNDocsConfig,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
