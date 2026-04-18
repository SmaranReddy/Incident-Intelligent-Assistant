"""Knowledge base CRUD routes."""

import re
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, get_current_principal, require_role
from app.core.database import get_db
from app.core.models import KnowledgeBase
from app.core.schemas import KBCreate, KBOut, KBUpdate
from app.kb.manager import kb_manager

router = APIRouter(prefix="/kb", tags=["Knowledge Bases"])


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


async def _get_kb_or_404(kb_id: str, org_id: str, db: AsyncSession) -> KnowledgeBase:
    """Fetch KB and verify it belongs to the principal's org."""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.org_id != org_id:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


# ---------------------------------------------------------------------------
# List KBs
# ---------------------------------------------------------------------------

@router.get("", response_model=List[KBOut])
async def list_kbs(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.org_id == principal.org_id)
    )
    return result.scalars().all()


# ---------------------------------------------------------------------------
# Create KB
# ---------------------------------------------------------------------------

@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
async def create_kb(
    body: KBCreate,
    principal: Principal = Depends(require_role("manage_kb")),
    db: AsyncSession = Depends(get_db),
):
    slug = _slug(body.name)

    # Ensure slug unique within org
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.org_id == principal.org_id,
            KnowledgeBase.slug == slug,
        )
    )
    if result.scalar_one_or_none():
        slug = slug + "-2"

    kb = KnowledgeBase(
        org_id=principal.org_id,
        name=body.name,
        slug=slug,
        description=body.description,
        created_by=principal.user_id,
        index_status="empty",
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


# ---------------------------------------------------------------------------
# Get KB
# ---------------------------------------------------------------------------

@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(
    kb_id: str,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    return await _get_kb_or_404(kb_id, principal.org_id, db)


# ---------------------------------------------------------------------------
# Update KB
# ---------------------------------------------------------------------------

@router.patch("/{kb_id}", response_model=KBOut)
async def update_kb(
    kb_id: str,
    body: KBUpdate,
    principal: Principal = Depends(require_role("manage_kb")),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, principal.org_id, db)

    if body.name is not None:
        kb.name = body.name
        kb.slug = _slug(body.name)
    if body.description is not None:
        kb.description = body.description

    await db.commit()
    await db.refresh(kb)
    return kb


# ---------------------------------------------------------------------------
# Delete KB
# ---------------------------------------------------------------------------

@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kb(
    kb_id: str,
    principal: Principal = Depends(require_role("manage_kb")),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, principal.org_id, db)

    # Remove FAISS + BM25 indexes from disk
    kb_manager.delete_kb(kb_id)

    await db.delete(kb)
    await db.commit()
