"""Analytics and usage metrics routes."""

import statistics
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, require_role
from app.core.database import get_db
from app.core.models import Document, Feedback, KnowledgeBase, QueryLog
from app.core.schemas import FeedbackSummary, LatencyStats, UsageSummary

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/usage", response_model=UsageSummary)
async def usage_summary(
    kb_id: Optional[str] = Query(default=None),
    principal: Principal = Depends(require_role("view_metrics")),
    db: AsyncSession = Depends(get_db),
):
    # Total queries
    q = select(func.count(QueryLog.id)).where(QueryLog.org_id == principal.org_id)
    if kb_id:
        q = q.where(QueryLog.kb_id == kb_id)
    total_queries = (await db.execute(q)).scalar() or 0

    # Total docs
    doc_q = select(func.count(Document.id)).where(Document.org_id == principal.org_id)
    if kb_id:
        doc_q = doc_q.where(Document.kb_id == kb_id)
    total_docs = (await db.execute(doc_q)).scalar() or 0

    # Total chunks
    kb_q = select(func.sum(KnowledgeBase.chunk_count)).where(KnowledgeBase.org_id == principal.org_id)
    total_chunks = (await db.execute(kb_q)).scalar() or 0

    # Avg latency
    lat_q = select(QueryLog.latency_ms).where(
        QueryLog.org_id == principal.org_id,
        QueryLog.latency_ms.isnot(None),
    )
    if kb_id:
        lat_q = lat_q.where(QueryLog.kb_id == kb_id)
    latencies = (await db.execute(lat_q)).scalars().all()
    avg_latency = round(statistics.mean(latencies), 1) if latencies else None

    # Cache hit rate
    cache_q = select(QueryLog.cache_hit).where(QueryLog.org_id == principal.org_id)
    if kb_id:
        cache_q = cache_q.where(QueryLog.kb_id == kb_id)
    cache_hits = (await db.execute(cache_q)).scalars().all()
    cache_hit_rate = round(sum(cache_hits) / len(cache_hits), 3) if cache_hits else None

    return UsageSummary(
        total_queries=total_queries,
        total_documents=total_docs,
        total_chunks=int(total_chunks),
        avg_latency_ms=avg_latency,
        cache_hit_rate=cache_hit_rate,
    )


@router.get("/feedback", response_model=FeedbackSummary)
async def feedback_summary(
    kb_id: Optional[str] = Query(default=None),
    principal: Principal = Depends(require_role("view_metrics")),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(Feedback)
        .join(QueryLog, Feedback.query_log_id == QueryLog.id)
        .where(Feedback.org_id == principal.org_id)
    )
    if kb_id:
        q = q.where(QueryLog.kb_id == kb_id)

    rows = (await db.execute(q)).scalars().all()

    thumbs_up = sum(1 for r in rows if r.rating == 1)
    thumbs_down = sum(1 for r in rows if r.rating == -1)
    total = len(rows)
    satisfaction = round(thumbs_up / total, 3) if total else 0.0

    return FeedbackSummary(
        total_feedback=total,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
        satisfaction_rate=satisfaction,
    )


@router.get("/latency", response_model=LatencyStats)
async def latency_stats(
    kb_id: Optional[str] = Query(default=None),
    principal: Principal = Depends(require_role("view_metrics")),
    db: AsyncSession = Depends(get_db),
):
    q = select(QueryLog.latency_ms).where(
        QueryLog.org_id == principal.org_id,
        QueryLog.latency_ms.isnot(None),
    )
    if kb_id:
        q = q.where(QueryLog.kb_id == kb_id)

    latencies = sorted((await db.execute(q)).scalars().all())

    if not latencies:
        return LatencyStats(p50_ms=None, p95_ms=None, p99_ms=None)

    def _percentile(data, p):
        idx = int(len(data) * p / 100)
        return float(data[min(idx, len(data) - 1)])

    return LatencyStats(
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


@router.get("/queries/recent")
async def recent_queries(
    kb_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, le=100),
    principal: Principal = Depends(require_role("view_metrics")),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(QueryLog)
        .where(QueryLog.org_id == principal.org_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    if kb_id:
        q = q.where(QueryLog.kb_id == kb_id)

    logs = (await db.execute(q)).scalars().all()
    return [
        {
            "id": log.id,
            "question": log.question,
            "confidence": log.confidence,
            "latency_ms": log.latency_ms,
            "cache_hit": log.cache_hit,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
