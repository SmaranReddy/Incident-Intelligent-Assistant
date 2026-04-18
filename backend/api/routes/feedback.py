"""User feedback on query responses."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, get_current_principal
from app.core.database import get_db
from app.core.models import Feedback, QueryLog
from app.core.schemas import FeedbackCreate, FeedbackOut

router = APIRouter(tags=["Feedback"])


@router.post("/query/{query_log_id}/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    query_log_id: str,
    body: FeedbackCreate,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    # Verify query log belongs to this org
    log = await db.get(QueryLog, query_log_id)
    if not log or log.org_id != principal.org_id:
        raise HTTPException(status_code=404, detail="Query log not found")

    # Prevent duplicate feedback from same user
    if principal.user_id:
        from sqlalchemy import select
        result = await db.execute(
            select(Feedback).where(
                Feedback.query_log_id == query_log_id,
                Feedback.user_id == principal.user_id,
            )
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Feedback already submitted for this query")

    feedback = Feedback(
        query_log_id=query_log_id,
        org_id=principal.org_id,
        user_id=principal.user_id,
        rating=body.rating,
        comment=body.comment,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)

    return feedback
