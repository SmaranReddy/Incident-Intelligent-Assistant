"""
Per-KB index manager.

Each knowledge base gets its own isolated FAISS index and BM25 index stored at:
    backend/data/indexes/{kb_id}/faiss.index
    backend/data/indexes/{kb_id}/chunks.json
    backend/data/indexes/{kb_id}/meta.json   ← next available faiss_id counter

The FAISS index is an IndexIDMap wrapping IndexFlatL2, which enables
ID-based deletion so that removing a document also removes its chunks
from the vector index.

The manager keeps a hot cache of loaded indexes (LRU by access).
Ingestion invalidates the cache entry so the next query reloads from disk.

Two-stage retrieval:
    Stage 1 — KBIndex.get_top_documents()
        Score every chunk via _compute_fused_scores(), aggregate to doc level
        (max chunk score per doc), return top N doc_ids.
    Stage 2 — KBIndex.search(filter_doc_ids=...)
        Re-run _compute_fused_scores() restricted to the Stage-1 doc set,
        return top-k chunks for the reranker.

Both stages share _compute_fused_scores() so the scoring logic is defined once.
"""

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "to", "of", "in", "on",
    "for", "with", "does", "do", "did", "what", "why", "how",
    "when", "where", "required", "necessary",
}


def _tokenize(text: str) -> List[str]:
    import re
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [t for t in tokens if t not in STOPWORDS]


def _norm_scores(d: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a score dict to [0, 1]. Handles empty and flat inputs."""
    if not d:
        return d
    vals = np.array(list(d.values()))
    mn, mx = vals.min(), vals.max()
    if mx - mn == 0:
        # All identical: 1.0 if there is signal, 0.0 if all zeros
        return {k: (1.0 if mx > 0 else 0.0) for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


# ---------------------------------------------------------------------------
# KBIndex — wraps FAISS + BM25 + chunk metadata for one KB
# ---------------------------------------------------------------------------

@dataclass
class KBIndex:
    kb_id: str
    faiss_index: faiss.Index
    chunks: List[dict]                   # list of {id, faiss_id, text, source, ...}
    chunk_map: Dict[str, dict] = field(default_factory=dict)
    bm25: Optional[BM25Okapi] = None

    def __post_init__(self):
        self.chunk_map = {str(c["id"]): c for c in self.chunks}
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    # ------------------------------------------------------------------
    # Shared scoring kernel — used by both Stage 1 and Stage 2
    # ------------------------------------------------------------------

    def _compute_fused_scores(
        self,
        query_embedding: np.ndarray,
        query: str,
        eligible: Optional[set] = None,
    ) -> Dict[str, float]:
        """
        Compute normalised, fused (70% vector + 30% BM25) scores for all
        candidate chunks.

        Args:
            eligible: Optional set of chunk IDs to include in results.
                      None means all chunks are considered (Stage 1).
                      A populated set restricts to selected-doc chunks (Stage 2).

        Returns:
            {chunk_id: fused_score} — only for chunks that pass the eligible
            filter and appear in at least one of FAISS or BM25 top-50.
        """
        # --- Vector retrieval (top-50 FAISS candidates) ---
        n_candidates = min(50, len(self.chunks))
        raw_scores, indices = self.faiss_index.search(query_embedding, n_candidates)
        vector_scores: Dict[str, float] = {}
        for score, faiss_id in zip(raw_scores[0], indices[0]):
            if faiss_id < 0:
                continue
            chunk = next(
                (c for c in self.chunks if c.get("faiss_id") == int(faiss_id)), None
            )
            if chunk:
                cid = str(chunk["id"])
                if eligible is None or cid in eligible:
                    vector_scores[cid] = float(1 / (1 + score))

        # --- BM25 retrieval (top-50 BM25 candidates) ---
        bm25_scores: Dict[str, float] = {}
        if self.bm25:
            tokenized_query = _tokenize(query)
            bm25_raw = self.bm25.get_scores(tokenized_query)
            top_bm25 = np.argsort(bm25_raw)[::-1][:50]
            for idx in top_bm25:
                cid = str(self.chunks[idx]["id"])
                if eligible is None or cid in eligible:
                    bm25_scores[cid] = float(bm25_raw[idx])

        # --- Normalise and fuse: 70% vector + 30% BM25 ---
        vector_scores = _norm_scores(vector_scores)
        bm25_scores   = _norm_scores(bm25_scores)

        alpha = 0.7
        candidate_ids = set(vector_scores) | set(bm25_scores)
        return {
            cid: alpha * vector_scores.get(cid, 0.0)
                 + (1 - alpha) * bm25_scores.get(cid, 0.0)
            for cid in candidate_ids
        }

    # ------------------------------------------------------------------
    # Stage 1 — document-level selection
    # ------------------------------------------------------------------

    def get_top_documents(
        self,
        query_embedding: np.ndarray,
        query: str,
        top_n: int = 5,
    ) -> List[str]:
        """
        Return the IDs of the top_n most relevant documents.

        Strategy: score every chunk via _compute_fused_scores (no filter),
        then aggregate chunk scores to document level using the *maximum*
        chunk score for each document.  Maximum is preferred over average
        because a single highly relevant chunk is enough to justify including
        the document in Stage 2.

        Returns:
            Ordered list of doc_ids (best first), length ≤ top_n.
        """
        if not self.chunks:
            return []

        fused = self._compute_fused_scores(query_embedding, query, eligible=None)

        # Aggregate: max chunk score per document
        doc_scores: Dict[str, float] = {}
        for cid, score in fused.items():
            chunk = self.chunk_map.get(cid)
            if not chunk:
                continue
            doc_id = chunk.get("doc_id", "")
            if doc_id:
                if score > doc_scores.get(doc_id, -1.0):
                    doc_scores[doc_id] = score

        logger.debug("[Stage1] Doc scores: %s", {k: round(v, 4) for k, v in doc_scores.items()})

        top_docs = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)[:top_n]
        logger.debug(
            "Stage-1 doc selection | kb=%s | top_n=%d | selected=%s | scores=%s",
            self.kb_id,
            top_n,
            top_docs,
            [round(doc_scores[d], 4) for d in top_docs],
        )
        return top_docs

    # ------------------------------------------------------------------
    # Stage 2 — chunk-level retrieval within selected documents
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        query: str,
        top_k: int = 10,
        filter_doc_ids: Optional[set] = None,
    ) -> List[dict]:
        """
        Hybrid retrieval: 70% FAISS vector score + 30% BM25 score.

        Args:
            filter_doc_ids: When provided (populated by Stage 1 or by the
                user's explicit doc selection), only chunks from those
                documents are scored.  This is the pre-filter introduced in
                the previous round's Task 3.
        """
        if not self.chunks:
            return []

        # Build eligible chunk-ID set for the chosen documents
        if filter_doc_ids:
            eligible: Optional[set] = {
                str(c["id"]) for c in self.chunks
                if c.get("doc_id") in filter_doc_ids
            }
            if not eligible:            # filter produced nothing — fall back
                eligible = None
        else:
            eligible = None

        fused = self._compute_fused_scores(query_embedding, query, eligible=eligible)

        # Sort and materialise top_k results
        results: List[dict] = []
        for cid in sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]:
            chunk = self.chunk_map.get(cid)
            if chunk:
                results.append({
                    "id": cid,
                    "text": chunk["text"],
                    "source": chunk.get("source", ""),
                    "score": round(fused[cid], 4),
                    "metadata": chunk,
                })

        return results


# ---------------------------------------------------------------------------
# KBIndexManager — singleton, loaded once at startup
# ---------------------------------------------------------------------------

class KBIndexManager:
    MAX_CACHED = 20   # maximum number of KB indexes to keep in memory

    def __init__(self):
        self._cache: OrderedDict[str, KBIndex] = OrderedDict()
        self._embed_model: Optional[SentenceTransformer] = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
            self._embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._embed_model

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], show_progress_bar=False)
        return np.array(vec).astype("float32")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.embed_model.encode(texts, show_progress_bar=True, batch_size=64)
        return np.array(vecs).astype("float32")

    # --- path helpers ---

    def _index_dir(self, kb_id: str) -> str:
        path = os.path.join(settings.INDEX_DIR, kb_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _faiss_path(self, kb_id: str) -> str:
        return os.path.join(self._index_dir(kb_id), "faiss.index")

    def _chunks_path(self, kb_id: str) -> str:
        return os.path.join(self._index_dir(kb_id), "chunks.json")

    def _meta_path(self, kb_id: str) -> str:
        return os.path.join(self._index_dir(kb_id), "meta.json")

    def _load_next_id(self, kb_id: str) -> int:
        """Return the next available integer faiss_id for this KB."""
        path = self._meta_path(kb_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f).get("next_faiss_id", 0)
        return 0

    def _save_next_id(self, kb_id: str, next_id: int) -> None:
        path = self._meta_path(kb_id)
        with open(path, "w") as f:
            json.dump({"next_faiss_id": next_id}, f)

    def _new_index(self) -> faiss.Index:
        """Create a fresh IndexIDMap wrapping IndexFlatL2."""
        return faiss.IndexIDMap(faiss.IndexFlatL2(settings.EMBEDDING_DIM))

    # --- cache management ---

    def _evict_if_needed(self):
        while len(self._cache) >= self.MAX_CACHED:
            evicted_id, _ = self._cache.popitem(last=False)
            logger.info("Evicted KB index from cache: %s", evicted_id)

    def invalidate(self, kb_id: str):
        """Call after ingestion so the next query loads fresh data from disk."""
        self._cache.pop(kb_id, None)

    # --- load / create ---

    def get(self, kb_id: str) -> KBIndex:
        """Return cached KBIndex or load from disk. Creates empty index if none exists."""
        if kb_id in self._cache:
            self._cache.move_to_end(kb_id)
            return self._cache[kb_id]

        self._evict_if_needed()

        faiss_path = self._faiss_path(kb_id)
        chunks_path = self._chunks_path(kb_id)

        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            logger.info("Loading KB index from disk: %s", kb_id)
            index = faiss.read_index(faiss_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            logger.info("Creating empty KB index: %s", kb_id)
            index = self._new_index()
            chunks = []

        kb_index = KBIndex(kb_id=kb_id, faiss_index=index, chunks=chunks)
        self._cache[kb_id] = kb_index
        return kb_index

    def add_chunks(self, kb_id: str, new_chunks: List[dict], embeddings: np.ndarray):
        """
        Add new chunks + embeddings to a KB's index.
        Each chunk is assigned a unique integer faiss_id for later deletion.
        Persists to disk and invalidates cache so next get() reloads fresh.
        """
        faiss_path = self._faiss_path(kb_id)
        chunks_path = self._chunks_path(kb_id)

        # Load or create existing index
        if os.path.exists(faiss_path) and os.path.exists(chunks_path):
            index = faiss.read_index(faiss_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
        else:
            index = self._new_index()
            existing_chunks = []

        # Assign integer faiss_ids starting from the stored counter
        next_id = self._load_next_id(kb_id)
        ids = np.arange(next_id, next_id + len(new_chunks), dtype=np.int64)
        for chunk, fid in zip(new_chunks, ids):
            chunk["faiss_id"] = int(fid)

        # Add embeddings with explicit IDs
        index.add_with_ids(embeddings, ids)

        # Merge and persist
        all_chunks = existing_chunks + new_chunks
        faiss.write_index(index, faiss_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        self._save_next_id(kb_id, next_id + len(new_chunks))
        logger.info("KB %s: added %d chunks, total=%d", kb_id, len(new_chunks), len(all_chunks))

        self.invalidate(kb_id)

    def remove_doc_chunks(self, kb_id: str, doc_id: str) -> int:
        """
        Remove all chunks belonging to doc_id from the FAISS index and chunks.json.
        Returns the number of chunks removed.
        """
        faiss_path = self._faiss_path(kb_id)
        chunks_path = self._chunks_path(kb_id)

        if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
            return 0

        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)

        to_remove = [c for c in all_chunks if c.get("doc_id") == doc_id]
        if not to_remove:
            return 0

        faiss_ids = np.array([c["faiss_id"] for c in to_remove], dtype=np.int64)
        remaining = [c for c in all_chunks if c.get("doc_id") != doc_id]

        index = faiss.read_index(faiss_path)
        index.remove_ids(faiss_ids)

        faiss.write_index(index, faiss_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(remaining, f, ensure_ascii=False)

        logger.info("KB %s: removed %d chunks for doc_id=%s", kb_id, len(to_remove), doc_id)
        self.invalidate(kb_id)
        return len(to_remove)

    def delete_kb(self, kb_id: str):
        """Remove a KB's index from disk and cache."""
        import shutil
        self.invalidate(kb_id)
        index_dir = os.path.join(settings.INDEX_DIR, kb_id)
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            logger.info("Deleted KB index directory: %s", kb_id)


# Singleton — imported everywhere
kb_manager = KBIndexManager()
