"""Query routes — synchronous and streaming.

Production improvements applied:
  Two-stage retrieval —
      Stage 1: get_top_documents() scores all chunks, aggregates to doc level,
               returns top-N doc_ids (default N=5).
      Stage 2: search(filter_doc_ids=...) re-scores only chunks from those
               documents, feeding a clean, document-filtered candidate set to
               the reranker.
      When the user explicitly provides doc_ids, Stage 1 is skipped and the
      user selection is used directly as the filter_doc_ids for Stage 2.
  Task 3  — Metadata pre-filtering passed into KBIndex.search()
  Task 5  — Optional query rewriting before retrieval
  Task 6  — Improved confidence: penalise fallback answers, boost multi-chunk
             agreement, apply answer-length factor
  Task 7  — Structured observability logging (chunks, scores, latency, cache)
  Task 8  — Cache hit/miss logging on top of existing TTL cache
"""

import asyncio
import json
import re
import time
import traceback
import uuid
import logging
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, check_rate_limit, require_scope
from app.cache.simple_cache import SimpleCache
from app.core.config import settings
from app.core.database import get_db
from app.core.models import KnowledgeBase, QueryLog
from app.core.schemas import QueryRequest, QueryResponse, SimilarIncident, SourceChunk, StructuredInfo
from app.generation.answer_generator import AnswerGenerator
from app.kb.manager import kb_manager
from app.retrieval.rerank import Reranker
from app.retrieval.structured_retrieval import (
    StructuredIncidentStore,
    enrich_chunks_with_structured,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kb/{kb_id}", tags=["Query"])

_reranker: Optional[Reranker] = None
_generator: Optional[AnswerGenerator] = None
_cache = SimpleCache()
_structured_store: Optional[StructuredIncidentStore] = None

# Phrases that indicate the LLM fell back to "I don't know" (Task 6)
_FALLBACK_PHRASES = (
    "i don't have enough information",
    "i cannot answer",
    "not present in the context",
    "not in the context",
    "not mentioned in the context",
    "error generating",
)

# ---------------------------------------------------------------------------
# Grounding check — prevents generation when chunks are unrelated to the query
# ---------------------------------------------------------------------------

# Answer returned when grounding fails (no LLM is called)
_GROUNDING_FALLBACK = "I don't have enough information in the selected documents."

# Minimum pre-rerank hybrid score for the top result.
# Post-rerank scores are normalised so the top chunk is always 1.0 — useless
# for a threshold check.  Pre-rerank hybrid scores are in [0, 1] and reflect
# true corpus-wide similarity, so only those are checked here.
_GROUNDING_SCORE_MIN = 0.25

# Common words filtered out before keyword matching so short/generic queries
# like "what is the policy" don't trivially pass on words like "the" or "is".
_GROUNDING_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "what", "how", "why", "when", "where", "which", "who",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "not",
    "tell", "me", "about", "explain", "describe", "can", "its", "our",
    "their", "this", "that", "these", "those", "get", "give", "say",
})


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator(api_key=settings.GROQ_API_KEY)
    return _generator


def _get_structured_store() -> StructuredIncidentStore:
    """Singleton structured incident store — loaded on first access."""
    global _structured_store
    if _structured_store is None:
        _structured_store = StructuredIncidentStore()
        _structured_store.load(settings.STRUCTURED_INCIDENTS_DIR)
    return _structured_store


async def _get_kb_or_404(kb_id: str, org_id: str, db: AsyncSession) -> KnowledgeBase:
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.org_id != org_id:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _compute_confidence(chunks: list, answer: str = "") -> float:
    """
    Improved confidence estimation (Task 6).

    Rules:
      1. Fallback / error responses → confidence 0.
      2. Weighted average of top-3 reranked scores (0.6 / 0.3 / 0.1).
      3. Boost by 10% when multiple chunks strongly agree (avg > 0.6).
      4. Length factor: very short answers are penalised (< 20 words).
    """
    if not chunks:
        return 0.0

    # Rule 1 — penalise fallback responses
    if answer:
        lower = answer.lower()
        if any(phrase in lower for phrase in _FALLBACK_PHRASES):
            return 0.0

    # Rule 2 — weighted average of top-3 scores
    weights = [0.6, 0.3, 0.1]
    total_w, total_s = 0.0, 0.0
    for chunk, w in zip(chunks[:3], weights):
        total_s += chunk.get("score", 0) * w
        total_w += w
    base = total_s / total_w if total_w else 0.0

    # Rule 3 — multi-chunk agreement boost
    if len(chunks) >= 2:
        avg = sum(c.get("score", 0) for c in chunks[:3]) / min(3, len(chunks))
        if avg > 0.6:
            base = min(1.0, base * 1.1)

    # Rule 4 — answer length factor
    if answer:
        word_count = len(answer.split())
        length_factor = min(1.0, word_count / 20.0)
        base = base * (0.7 + 0.3 * length_factor)

    return round(base, 3)


def _maybe_rewrite_query(query: str) -> str:
    """Run query rewriting if enabled in config (Task 5)."""
    if not settings.ENABLE_QUERY_REWRITING:
        return query
    rewritten = _get_generator().rewrite_query(query)
    if rewritten != query:
        logger.info("Query rewritten | original=%r | rewritten=%r", query, rewritten)
    return rewritten


def _check_grounding(query: str, results: list, reranked: list) -> tuple[bool, str]:
    """
    Decide whether the retrieved chunks are sufficiently grounded in the query
    to justify calling the LLM.  Returns (is_grounded, reason_string).

    Two heuristics — OR logic: passing either one is enough.

    Heuristic 1 — keyword presence:
        Strip stopwords from the query.  If at least one remaining token
        appears (case-insensitive) in the combined text of the reranked
        chunks, the content is considered relevant.

    Heuristic 2 — similarity score:
        The top entry in `results` carries the raw pre-rerank hybrid score
        (FAISS + BM25 fusion, normalised to [0, 1] corpus-wide).
        If this score >= _GROUNDING_SCORE_MIN the corpus has at least some
        signal for the query.
        NOTE: post-rerank scores are always 1.0 for the top chunk — only
        pre-rerank scores in `results` are meaningful for this check.

    Both failing means the chunks are unrelated to the query; the LLM should
    not be called.
    """
    # Extract meaningful query tokens (length > 2, not a stopword)
    query_tokens = {
        t for t in re.findall(r"[a-zA-Z]+", query.lower())
        if len(t) > 2 and t not in _GROUNDING_STOP_WORDS
    }

    # Heuristic 1 — keyword overlap with any retrieved chunk
    if query_tokens and reranked:
        combined_text = " ".join(c["text"].lower() for c in reranked)
        chunk_tokens  = set(re.findall(r"[a-zA-Z]+", combined_text))
        matched       = query_tokens & chunk_tokens
        if matched:
            return True, f"keyword hit: {sorted(matched)[:3]}"

    # Heuristic 2 — top pre-rerank hybrid score above threshold
    if results:
        top_score = results[0]["score"]
        if top_score >= _GROUNDING_SCORE_MIN:
            return True, f"score {top_score:.3f} >= {_GROUNDING_SCORE_MIN}"

    # Both heuristics failed — build an informative reason string
    fail_parts: list[str] = []
    if query_tokens:
        fail_parts.append("no keyword overlap with chunks")
    else:
        fail_parts.append("query has no meaningful tokens after stopword removal")
    if results:
        fail_parts.append(f"top score {results[0]['score']:.3f} < {_GROUNDING_SCORE_MIN}")
    else:
        fail_parts.append("no results")

    return False, " | ".join(fail_parts)


# ---------------------------------------------------------------------------
# Shared retrieval helper — used by both sync and streaming endpoints
# ---------------------------------------------------------------------------

def get_top_documents(query: str, kb_id: str, top_n: int = settings.TOP_N_DOCS) -> List[str]:
    """
    Stage 1 of two-stage retrieval.

    Scores every chunk in the KB, aggregates to document level (max chunk
    score per doc), and returns the top_n most relevant doc_ids.

    Args:
        query:  The retrieval query (may be a rewritten version of the user query).
        kb_id:  Knowledge base identifier.
        top_n:  Maximum number of documents to select.

    Returns:
        Ordered list of doc_ids, best-scoring first.
    """
    query_embedding = kb_manager.embed_query(query)
    kb_index = kb_manager.get(kb_id)
    return kb_index.get_top_documents(query_embedding, query, top_n=top_n)


def _retrieve_and_rerank(kb_id: str, query: str, retrieval_query: str, body: QueryRequest):
    """
    Two-stage retrieval → re-rank.

    Stage 1 (document selection):
        If the user provided explicit doc_ids, respect them directly.
        Otherwise call get_top_documents() to find the top-N most relevant
        documents — fully deterministic, score-based, no LLM involved.

    Stage 2 (chunk retrieval):
        Run hybrid search restricted to the Stage-1 doc set.

    Returns (results, reranked) or raises ValueError if nothing found.
    """
    query_embedding = kb_manager.embed_query(retrieval_query)
    kb_index = kb_manager.get(kb_id)

    # ── Stage 1: document-level filtering ─────────────────────────────
    if body.doc_ids:
        # User explicitly selected documents — skip Stage 1
        filter_doc_ids: Optional[set] = set(body.doc_ids)
        logger.debug(
            "Stage-1 skipped (user-supplied) | kb=%s | doc_ids=%s",
            kb_id, list(filter_doc_ids),
        )
    else:
        top_doc_ids = kb_index.get_top_documents(
            query_embedding, retrieval_query, top_n=settings.TOP_N_DOCS
        )
        logger.debug("[Stage1] Selected docs: %s", top_doc_ids)
        # Task 2: empty Stage-1 output → fall back to full-corpus search
        filter_doc_ids = set(top_doc_ids) if top_doc_ids else None
        logger.info(
            "Stage-1 complete | kb=%s | top_docs=%s",
            kb_id, top_doc_ids,
        )

    # ── Stage 2: chunk-level retrieval within selected documents ───────
    results = kb_index.search(
        query_embedding,
        retrieval_query,
        top_k=10,
        filter_doc_ids=filter_doc_ids,
    )

    # Task 7: log retrieved candidates
    logger.debug(
        "Stage-2 chunks | kb=%s | count=%d | top_scores=%s",
        kb_id,
        len(results),
        [round(r["score"], 4) for r in results[:5]],
    )

    if not results:
        raise ValueError("No matching chunks found")

    # ── Structured incident layer ────────────────────────────────────────
    # Run structured search in parallel with vector results (both in-memory).
    # Enriches matching chunks and optionally injects synthetic ones.
    store = _get_structured_store()
    structured_hits = store.search(query, top_k=10)

    if structured_hits:
        logger.debug(
            "Structured hits | kb=%s | top=%s",
            kb_id,
            [(inc["incident_id"], round(s, 3)) for inc, s in structured_hits[:3]],
        )
        results, chunk_structured_scores = enrich_chunks_with_structured(
            chunks=results,
            store=store,
            structured_hits=structured_hits,
            inject_limit=settings.STRUCTURED_INJECT_LIMIT,
            inject_min_score=settings.STRUCTURED_INJECT_MIN_SCORE,
        )
        enriched_count = sum(1 for c in results if "structured" in c)
        logger.info(
            "Structured enrichment | kb=%s | enriched=%d | injected=%d",
            kb_id,
            enriched_count,
            sum(1 for c in results if c.get("_is_synthetic")),
        )
    else:
        chunk_structured_scores = {}

    reranked = _get_reranker().rerank(
        query,
        results,
        top_k=body.top_k,
        structured_scores=chunk_structured_scores,
    )

    # Task 7: log final selected chunks
    logger.debug(
        "Reranked chunks | kb=%s | selected=%d | scores=%s | sources=%s",
        kb_id,
        len(reranked),
        [round(c["score"], 4) for c in reranked],
        [c.get("source", "")[:40] for c in reranked],
    )

    return results, reranked


# ---------------------------------------------------------------------------
# Incident intelligence — builds structured insights from reranked chunks
# ---------------------------------------------------------------------------

_STRUCTURED_REQUIRED_FIELDS = (
    "incident_id", "service", "root_cause", "resolution",
    "services_affected", "structured_score",
)


def _safe_structured_info(s: Optional[dict], chunk_id: str = "") -> Optional[StructuredInfo]:
    """Build StructuredInfo from a raw dict, filling missing required fields with safe defaults.

    StructuredInfo has no field defaults, so passing an incomplete dict via **unpacking
    raises a Pydantic ValidationError → 500. This helper prevents that.
    """
    if not s or not isinstance(s, dict):
        return None

    missing = [f for f in _STRUCTURED_REQUIRED_FIELDS if not s.get(f)]
    if missing:
        logger.warning(
            "Structured chunk has missing/empty fields | chunk_id=%s | missing=%s",
            chunk_id or s.get("incident_id", "?"), missing,
        )

    try:
        return StructuredInfo(
            incident_id=str(s.get("incident_id") or ""),
            service=str(s.get("service") or ""),
            root_cause=str(s.get("root_cause") or ""),
            resolution=str(s.get("resolution") or ""),
            services_affected=[str(x) for x in s.get("services_affected") or []],
            structured_score=float(s.get("structured_score") or 0.0),
        )
    except Exception as exc:
        logger.warning(
            "StructuredInfo validation failed | chunk_id=%s | error=%s",
            chunk_id or s.get("incident_id", "?"), exc,
        )
        return None


_EMPTY_INSIGHTS: dict = {
    "likely_root_cause": None,
    "suggested_fix": None,
    "similar_incidents": [],
    "affected_services": [],
}


def _build_incident_insights(reranked: list) -> dict:
    try:
        if not reranked:
            return _EMPTY_INSIGHTS.copy()

        invalid_count = 0
        structured = []
        for c in reranked:
            raw = c.get("structured")
            if not raw:
                continue
            info = _safe_structured_info(raw, chunk_id=c.get("id", ""))
            if info is None:
                invalid_count += 1
            else:
                structured.append(c)

        if invalid_count:
            logger.warning(
                "Dropped %d structured chunk(s) due to invalid/incomplete data", invalid_count
            )

        if not structured:
            return _EMPTY_INSIGHTS.copy()

        top_chunks = sorted(structured, key=lambda c: c.get("score", 0), reverse=True)[:5]

        root_cause_scores: dict[str, float] = {}
        root_cause_resolution: dict[str, str] = {}

        for chunk in top_chunks:
            s = chunk.get("structured") or {}
            rc = s.get("root_cause")
            if not rc:
                continue
            root_cause_scores[rc] = root_cause_scores.get(rc, 0.0) + chunk.get("score", 0)
            if rc not in root_cause_resolution and s.get("resolution"):
                root_cause_resolution[rc] = s["resolution"]

        if root_cause_scores:
            best_root_cause = max(root_cause_scores, key=root_cause_scores.get)
            suggested_fix = root_cause_resolution.get(best_root_cause)
        else:
            best_root_cause = None
            suggested_fix = None

        affected_services = list({
            svc
            for c in structured
            for svc in (c.get("structured") or {}).get("services_affected", [])
            if isinstance(svc, str)
        })

        similar_incidents = [
            SimilarIncident(
                incident_id=c["structured"].get("incident_id", ""),
                service=c["structured"].get("service", ""),
                score=round(c.get("score", 0), 2),
            )
            for c in structured[:5]
            if (c.get("structured") or {}).get("incident_id") and (c.get("structured") or {}).get("service")
        ]

        return {
            "likely_root_cause": best_root_cause,
            "suggested_fix": suggested_fix,
            "similar_incidents": [s.model_dump() for s in similar_incidents],
            "affected_services": affected_services,
        }
    except Exception:
        logger.exception("_build_incident_insights failed — returning empty insights")
        return _EMPTY_INSIGHTS.copy()


# ---------------------------------------------------------------------------
# Synchronous query endpoint
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def query_kb(
    kb_id: str,
    body: QueryRequest,
    principal: Principal = Depends(require_scope("query")),
    db: AsyncSession = Depends(get_db),
):
    check_rate_limit(principal)

    kb = await _get_kb_or_404(kb_id, principal.org_id, db)

    if kb.index_status == "empty":
        raise HTTPException(
            status_code=422,
            detail="This knowledge base has no documents yet. Upload documents first.",
        )

    # Task 8: Cache check — keyed by kb_id + original query + selected doc_ids
    cache_key = f"{kb_id}:{body.query}"
    if body.doc_ids:
        cache_key += ":" + ",".join(sorted(body.doc_ids))
    cached = _cache.get(cache_key)
    if cached is not None:
        # Task 7: log cache hit
        logger.info(
            "Cache HIT | kb=%s | query=%r | key=%s",
            kb_id, body.query[:80], cache_key[:60],
        )
        cached["request_id"] = str(uuid.uuid4())
        cached["cache_hit"] = True
        return QueryResponse(**cached)

    # Task 7: log cache miss
    logger.info("Cache MISS | kb=%s | query=%r", kb_id, body.query[:80])

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    logger.info("Received query | kb=%s | query=%r", kb_id, body.query[:80])

    try:
        # Task 5: optional query rewriting (uses retrieval_query for embedding/search)
        retrieval_query = _maybe_rewrite_query(body.query)

        # Retrieve + re-rank
        try:
            results, reranked = _retrieve_and_rerank(kb_id, body.query, retrieval_query, body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        logger.info("Retrieved %d chunks after reranking | kb=%s", len(reranked), kb_id)

        t_retrieval_ms = int((time.perf_counter() - t0) * 1000)

        # Grounding check — skip LLM entirely if chunks are unrelated to the query
        is_grounded, ground_reason = _check_grounding(body.query, results, reranked)
        if not is_grounded:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "Grounding FAILED | kb=%s | query=%r | reason=%s | latency_ms=%d",
                kb_id, body.query[:80], ground_reason, latency_ms,
            )
            return QueryResponse(
                request_id=request_id,
                answer=_GROUNDING_FALLBACK,
                sources=[],
                confidence=0.0,
                latency_ms=latency_ms,
                cache_hit=False,
                model_used=settings.LLM_MODEL,
            )

        logger.debug("Grounding OK | kb=%s | reason=%s", kb_id, ground_reason)

        # Generate answer using original query (not rewritten) for user-facing accuracy
        context = [c["text"] for c in reranked if c.get("text")]
        answer = _get_generator().generate(query=body.query, retrieved_chunks=context)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        # Task 6: improved confidence uses the generated answer
        confidence = _compute_confidence(reranked, answer)

        sources = [
            SourceChunk(
                id=c.get("id", ""),
                text=(c.get("text") or "")[:300],
                source=c.get("source", ""),
                score=c.get("score", 0.0),
                structured=_safe_structured_info(c.get("structured"), chunk_id=c.get("id", "")),
            )
            for c in reranked
        ]

        # Task 8: store in cache
        cacheable = {
            "request_id": request_id,
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "confidence": confidence,
            "latency_ms": latency_ms,
            "cache_hit": False,
            "model_used": settings.LLM_MODEL,
            **_build_incident_insights(reranked),
        }
        _cache.set(cache_key, cacheable)

        # Task 7: structured observability log
        logger.info(
            "Query complete | kb=%s | query=%r | retrieval_ms=%d | total_ms=%d | "
            "retrieved=%d | reranked=%d | confidence=%.3f | cache_hit=False",
            kb_id, body.query[:80], t_retrieval_ms, latency_ms,
            len(results), len(reranked), confidence,
        )

        # Log to DB
        log = QueryLog(
            org_id=principal.org_id,
            kb_id=kb_id,
            user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            question=body.query,
            answer=answer,
            chunk_ids=[c.get("id", "") for c in reranked],
            confidence=confidence,
            model_used=settings.LLM_MODEL,
            latency_ms=latency_ms,
            cache_hit=False,
        )
        db.add(log)
        await db.commit()

        return QueryResponse(**cacheable)

    except HTTPException:
        raise
    except Exception:
        print("\n=== QUERY HANDLER TRACEBACK ===")
        traceback.print_exc()
        print("================================\n")
        logger.exception("Query failed | kb=%s | query=%r", kb_id, body.query[:80])
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return QueryResponse(
            request_id=request_id,
            answer="I don't have enough information to answer this question.",
            sources=[],
            confidence=0.0,
            latency_ms=latency_ms,
            cache_hit=False,
            model_used=settings.LLM_MODEL,
            likely_root_cause=None,
            suggested_fix=None,
            similar_incidents=[],
            affected_services=[],
        )


# ---------------------------------------------------------------------------
# Streaming query endpoint (SSE)
# ---------------------------------------------------------------------------

@router.post("/stream")
async def stream_kb(
    kb_id: str,
    body: QueryRequest,
    principal: Principal = Depends(require_scope("query")),
    db: AsyncSession = Depends(get_db),
):
    check_rate_limit(principal)

    kb = await _get_kb_or_404(kb_id, principal.org_id, db)

    if kb.index_status == "empty":
        raise HTTPException(
            status_code=422,
            detail="This knowledge base has no documents yet.",
        )

    t0 = time.perf_counter()

    # Task 5: optional query rewriting
    retrieval_query = _maybe_rewrite_query(body.query)

    try:
        results, reranked = _retrieve_and_rerank(kb_id, body.query, retrieval_query, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    t_retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # Grounding check — stream fallback immediately if chunks are unrelated
    is_grounded, ground_reason = _check_grounding(body.query, results, reranked)
    if not is_grounded:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Grounding FAILED (stream) | kb=%s | query=%r | reason=%s | latency_ms=%d",
            kb_id, body.query[:80], ground_reason, latency_ms,
        )

        async def _grounding_fallback() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'token': _GROUNDING_FALLBACK})}\n\n"
            await asyncio.sleep(0)
            yield f"data: {json.dumps({'done': True, 'sources': [], 'confidence': 0.0, 'cache_hit': False})}\n\n"
            await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_grounding_fallback(), media_type="text/event-stream")

    logger.debug("Grounding OK (stream) | kb=%s | reason=%s", kb_id, ground_reason)

    context = [c["text"] for c in reranked]

    async def _stream() -> AsyncIterator[str]:
        full_answer = ""
        try:
            for token in _get_generator().stream_generate(query=body.query, retrieved_chunks=context):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            logger.error("Streaming generation error: %s", e)
            yield f"data: {json.dumps({'error': 'Error generating response'})}\n\n"
            await asyncio.sleep(0)
            return

        latency_ms = int((time.perf_counter() - t0) * 1000)
        # Task 6: confidence uses the full streamed answer
        confidence = _compute_confidence(reranked, full_answer)

        sources = [
            {
                "id": c["id"],
                "text": c["text"][:300],
                "source": c.get("source", ""),
                "score": c["score"],
                "structured": c.get("structured"),
            }
            for c in reranked
        ]

        yield f"data: {json.dumps({'done': True, 'sources': sources, 'confidence': confidence, 'cache_hit': False})}\n\n"
        await asyncio.sleep(0)
        yield "data: [DONE]\n\n"
        await asyncio.sleep(0)

        # Task 7: stream observability log
        logger.info(
            "Stream complete | kb=%s | query=%r | retrieval_ms=%d | total_ms=%d | "
            "retrieved=%d | reranked=%d | confidence=%.3f",
            kb_id, body.query[:80], t_retrieval_ms, latency_ms,
            len(results), len(reranked), confidence,
        )

        log = QueryLog(
            org_id=principal.org_id,
            kb_id=kb_id,
            user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            question=body.query,
            answer=full_answer,
            chunk_ids=[c["id"] for c in reranked],
            confidence=confidence,
            model_used=settings.LLM_MODEL,
            latency_ms=latency_ms,
            cache_hit=False,
        )
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as bg_db:
            bg_db.add(log)
            await bg_db.commit()

    return StreamingResponse(_stream(), media_type="text/event-stream")
