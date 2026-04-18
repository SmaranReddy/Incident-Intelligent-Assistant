"""
Chunk text pages and add them to a KB's FAISS + BM25 index.

Chunking strategy (Task 4):
  - Target size : CHUNK_SIZE_TOKENS  (default 400, range 200-500)
  - Overlap     : CHUNK_OVERLAP_TOKENS (default 50, range 20-50)
  - Minimum size: MIN_CHUNK_TOKENS   (default 50) — tiny trailing chunks
                  are merged into the previous one instead of being kept
                  as noise.
  - Metadata    : each chunk records doc_id, page_num, section, and its
                  sequential position within the page (chunk_index).
"""

import logging
import uuid
from typing import List

import tiktoken

from app.ingestion.parsers.base import ParsedPage
from app.kb.manager import kb_manager
from app.core.config import settings

logger = logging.getLogger(__name__)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _chunk_text(
    text: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
) -> List[str]:
    """
    Split *text* into token-bounded chunks with overlap.

    Chunks shorter than *min_tokens* are merged into the preceding chunk
    (up to *max_tokens*) to avoid polluting the index with micro-fragments.
    """
    tokens = _tokenizer.encode(text)
    if not tokens:
        return []

    step = max(1, max_tokens - overlap)

    # Build raw token windows
    raw: List[List[int]] = []
    for i in range(0, len(tokens), step):
        window = tokens[i: i + max_tokens]
        if len(window) >= 3:          # skip degenerate 1-2 token windows
            raw.append(window)

    if not raw:
        return []

    # Merge trailing micro-chunks into the previous chunk (Task 4 — avoid noise)
    merged: List[List[int]] = [raw[0]]
    for window in raw[1:]:
        if len(window) < min_tokens:
            # Absorb into the previous chunk, capped at max_tokens
            combined = (merged[-1] + window)[:max_tokens]
            merged[-1] = combined
        else:
            merged.append(window)

    return [_tokenizer.decode(t) for t in merged]


def index_document(
    kb_id: str,
    doc_id: str,
    filename: str,
    pages: List[ParsedPage],
) -> int:
    """
    Chunk pages, embed, and add to KB index.
    Returns the number of chunks created.
    """
    raw_chunks = []
    global_chunk_idx = 0   # monotonic position across all pages

    for page in pages:
        text_chunks = _chunk_text(
            page.text,
            max_tokens=settings.CHUNK_SIZE_TOKENS,
            overlap=settings.CHUNK_OVERLAP_TOKENS,
            min_tokens=settings.MIN_CHUNK_TOKENS,
        )
        for local_idx, chunk_text in enumerate(text_chunks):
            raw_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "source": filename,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "page_num": page.page_num,
                "section": page.section,
                "chunk_index": global_chunk_idx,   # global position (Task 4 metadata)
                "page_chunk_index": local_idx,      # position within page
            })
            global_chunk_idx += 1

    if not raw_chunks:
        logger.warning(
            "No chunks produced for doc_id=%s (file=%s). "
            "The document may be empty or contain only whitespace.",
            doc_id, filename,
        )
        return 0

    logger.info(
        "Chunked doc_id=%s into %d chunks (max_tokens=%d, overlap=%d, min_tokens=%d)",
        doc_id, len(raw_chunks),
        settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS, settings.MIN_CHUNK_TOKENS,
    )

    texts = [c["text"] for c in raw_chunks]
    embeddings = kb_manager.embed_texts(texts)
    kb_manager.add_chunks(kb_id, raw_chunks, embeddings)

    return len(raw_chunks)
