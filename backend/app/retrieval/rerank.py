"""
Lightweight re-ranker using TF-IDF cosine similarity.

Default strategy (no structured data):
  final_score = 0.6 * hybrid_score + 0.4 * tfidf_cosine_score

When a chunk has a structured incident score (from StructuredIncidentStore):
  final_score = 0.6 * hybrid_score + 0.4 * structured_score

The structured score replaces TF-IDF cosine for chunks that are linked to a
known incident, encoding causal and service-level signal rather than lexical
similarity.

Falls back to normalised keyword-overlap if sklearn is unavailable.
Selects top_k=3 by default (re-rank from top-10 hybrid candidates).
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int = 3,
        structured_scores: Optional[Dict[str, float]] = None,
    ) -> list:
        """
        Re-rank chunks by combining hybrid retrieval score with either a
        structured incident score (when available) or TF-IDF cosine similarity.

        Args:
            query:             User query string.
            chunks:            Candidate chunks, each with a ``score`` field.
            top_k:             Number of chunks to return.
            structured_scores: Optional mapping of chunk_id → structured_score.
                               When present for a chunk, replaces TF-IDF cosine.

        Returns:
            Top-k chunks with scores normalised to [0, 1].
        """
        if not chunks:
            return []

        cosine_scores = self._tfidf_cosine(query, chunks)
        _struct = structured_scores or {}

        scored = []
        for i, chunk in enumerate(chunks):
            s_score = _struct.get(chunk["id"])
            if s_score is not None:
                # Structured score available: encode causal signal
                # 60% hybrid (FAISS+BM25) + 40% structured incident score
                final_score = 0.6 * chunk["score"] + 0.4 * s_score
                logger.debug(
                    "Chunk %s: hybrid=%.4f struct=%.4f → %.4f",
                    chunk["id"], chunk["score"], s_score, final_score,
                )
            else:
                # No structured match: fall back to lexical cosine
                # 60% hybrid + 40% TF-IDF cosine
                final_score = 0.6 * chunk["score"] + 0.4 * float(cosine_scores[i])
            scored.append((chunk, final_score))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        # Normalise so the top chunk is always 1.0
        top_score = ranked[0][1] if ranked else 1.0
        if top_score <= 0:
            top_score = 1.0

        return [
            {**chunk, "score": round(score / top_score, 4)}
            for chunk, score in ranked
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tfidf_cosine(self, query: str, chunks: list) -> np.ndarray:
        """Return per-chunk cosine similarity scores against the query."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            texts = [query] + [c["text"] for c in chunks]
            vec = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                sublinear_tf=True,   # log-normalise term frequencies
            )
            tfidf_matrix = vec.fit_transform(texts)
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            return scores.astype(float)
        except Exception as exc:
            logger.warning(
                "TF-IDF reranking unavailable, falling back to keyword overlap: %s", exc
            )
            return self._keyword_overlap(query, chunks)

    def _keyword_overlap(self, query: str, chunks: list) -> np.ndarray:
        """Normalised term-overlap fallback (no sklearn required)."""
        query_terms = set(query.lower().split())
        if not query_terms:
            return np.zeros(len(chunks))
        scores = []
        for chunk in chunks:
            chunk_terms = set(chunk["text"].lower().split())
            overlap = len(query_terms & chunk_terms)
            scores.append(overlap / len(query_terms))
        return np.array(scores)
