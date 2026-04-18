"""Document upload, listing, status, and deletion routes."""

import os
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, get_current_principal, require_role
from app.core.config import settings
from app.core.database import get_db
from app.core.models import Document, KnowledgeBase
from app.core.schemas import DocumentListOut, DocumentOut
from app.ingestion.parsers import SUPPORTED_TYPES
from app.ingestion.pipeline import run_ingestion
from app.kb.manager import kb_manager

router = APIRouter(prefix="/kb/{kb_id}/documents", tags=["Documents"])

MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


async def _get_kb_or_404(kb_id: str, org_id: str, db: AsyncSession) -> KnowledgeBase:
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb or kb.org_id != org_id:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


def _detect_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext if ext in SUPPORTED_TYPES else "txt"


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

@router.get("", response_model=DocumentListOut)
async def list_documents(
    kb_id: str,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    await _get_kb_or_404(kb_id, principal.org_id, db)

    result = await db.execute(
        select(Document)
        .where(Document.kb_id == kb_id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    docs = result.scalars().all()

    count_result = await db.execute(
        select(Document).where(Document.kb_id == kb_id)
    )
    total = len(count_result.scalars().all())

    return DocumentListOut(documents=list(docs), total=total)


# ---------------------------------------------------------------------------
# Upload document
# ---------------------------------------------------------------------------

@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    principal: Principal = Depends(require_role("ingest")),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_kb_or_404(kb_id, principal.org_id, db)

    # Validate file size
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    file_type = _detect_type(file.filename or "upload.txt")

    # Save raw file to local storage
    org_upload_dir = os.path.join(settings.UPLOAD_DIR, principal.org_id, kb_id)
    os.makedirs(org_upload_dir, exist_ok=True)

    # Create DB record first to get document ID
    doc = Document(
        kb_id=kb_id,
        org_id=principal.org_id,
        filename=file.filename or "upload",
        storage_path="",    # filled in below
        file_type=file_type,
        file_size_bytes=len(raw),
        status="pending",
        uploaded_by=principal.user_id,
    )
    db.add(doc)
    await db.flush()   # get doc.id

    storage_path = os.path.join(org_upload_dir, f"{doc.id}_{file.filename}")
    with open(storage_path, "wb") as f_out:
        f_out.write(raw)

    doc.storage_path = storage_path
    await db.commit()
    await db.refresh(doc)

    # Queue background ingestion
    background_tasks.add_task(run_ingestion, doc.id)

    return doc


# ---------------------------------------------------------------------------
# Get document status
# ---------------------------------------------------------------------------

@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    kb_id: str,
    doc_id: str,
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    await _get_kb_or_404(kb_id, principal.org_id, db)
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------------

@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    kb_id: str,
    doc_id: str,
    principal: Principal = Depends(require_role("manage_kb")),
    db: AsyncSession = Depends(get_db),
):
    await _get_kb_or_404(kb_id, principal.org_id, db)
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove raw file from disk
    if doc.storage_path and os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)

    # Remove chunks from FAISS index
    kb_manager.remove_doc_chunks(kb_id, doc.id)

    await db.delete(doc)

    # Update KB aggregate counts
    kb = await db.get(KnowledgeBase, kb_id)
    if kb:
        kb.doc_count = max(0, (kb.doc_count or 1) - 1)
        kb.chunk_count = max(0, (kb.chunk_count or doc.chunk_count) - doc.chunk_count)

    await db.commit()
