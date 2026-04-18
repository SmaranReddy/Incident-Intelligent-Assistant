"""
Ingestion pipeline orchestrator.

Called from the FastAPI BackgroundTasks when a document is uploaded.
Flow: download raw file → parse → chunk+embed → index → update DB status.
"""

import asyncio
import functools
import logging
import os
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.models import Document, KnowledgeBase
from app.ingestion.parsers import get_parser
from app.ingestion.indexer import index_document

logger = logging.getLogger(__name__)


async def run_ingestion(document_id: str) -> None:
    """
    Entry point called by BackgroundTasks.
    Opens its own DB session (background tasks run outside the request session).
    """
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, document_id)
        if not doc:
            logger.error("Document not found: %s", document_id)
            return

        await _set_status(db, doc, "processing")

        try:
            # 1. Read raw file from local storage
            if not os.path.exists(doc.storage_path):
                raise FileNotFoundError(f"File not found at: {doc.storage_path}")

            with open(doc.storage_path, "rb") as f:
                raw_bytes = f.read()

            # 2. Parse to pages
            parser = get_parser(doc.file_type)
            pages = parser.parse(raw_bytes, filename=doc.filename)

            if not pages:
                raise ValueError("Parser returned no content — file may be empty or unreadable")

            # 3. Chunk + embed + index — run in a thread pool so the async event
            #    loop is not blocked by the CPU-heavy embedding step.
            loop = asyncio.get_event_loop()
            chunk_count = await loop.run_in_executor(
                None,
                functools.partial(
                    index_document,
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    filename=doc.filename,
                    pages=pages,
                ),
            )

            # 4. Update document record
            doc.status = "indexed"
            doc.chunk_count = chunk_count
            doc.indexed_at = datetime.utcnow()

            # 5. Update KB aggregate counts
            kb = await db.get(KnowledgeBase, doc.kb_id)
            if kb:
                kb.doc_count = (kb.doc_count or 0) + 1
                kb.chunk_count = (kb.chunk_count or 0) + chunk_count
                kb.index_status = "ready"

            await db.commit()
            logger.info(
                "Document indexed: doc_id=%s kb_id=%s chunks=%d",
                document_id, doc.kb_id, chunk_count,
            )

        except Exception as exc:
            logger.exception("Ingestion failed for doc_id=%s: %s", document_id, exc)
            try:
                doc.status = "failed"
                doc.error_message = str(exc)[:500]
                await db.commit()
            except Exception:
                pass


async def _set_status(db: AsyncSession, doc: Document, status: str) -> None:
    doc.status = status
    await db.commit()
