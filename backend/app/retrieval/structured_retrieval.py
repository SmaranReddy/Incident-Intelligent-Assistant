"""
Structured incident knowledge layer for the RAG system.

Loads JSON incident files from disk into memory, extracts structured attributes
from free-text queries (service name, error types, symptom keywords), and
computes a structured similarity score for each incident.

Designed to *augment* the existing FAISS + BM25 hybrid retrieval — not replace it.

When a hybrid result chunk can be linked to a structured incident (via source
filename), its structured score replaces TF-IDF cosine in the reranker formula:

    final_score = 0.6 * hybrid_score + 0.4 * structured_score

For top structural matches that were *not* retrieved by vector search, this
module can synthesise lightweight "structured-only" chunks so that causally
relevant incidents are never silently dropped.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service name aliases — maps query tokens to canonical service names
# ---------------------------------------------------------------------------

_SERVICE_ALIASES: Dict[str, str] = {
    "payment":        "payment-service",
    "payments":       "payment-service",
    "checkout":       "checkout-service",
    "order":          "order-service",
    "orders":         "order-service",
    "inventory":      "inventory-service",
    "auth":           "auth-service",
    "authentication": "auth-service",
    "user":           "user-service",
    "users":          "user-service",
    "api":            "api-gateway",
    "gateway":        "api-gateway",
    "notification":   "notification-service",
    "notifications":  "notification-service",
    "search":         "search-service",
    "cache":          "cache-service",
    "redis":          "cache-service",
}

# Error / symptom keywords grouped by canonical type
_ERROR_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "timeout":    ["timeout", "timed out", "deadline exceeded", "latency", "slow", "p99", "p95"],
    "rate_limit": ["429", "rate limit", "rate limiting", "throttl", "too many requests"],
    "connection": ["connection refused", "connection pool", "connection reset", "econnrefused"],
    "memory":     ["oom", "out of memory", "memory leak", "heap"],
    "cpu":        ["high cpu", "cpu spike"],
    "crash":      ["crash", "restart", "pod restart", "panic", "oom kill"],
    "deploy":     ["deploy", "deployment", "rollout", "release", "version", "upgrade"],
    "database":   ["deadlock", "db lock", "migration", "connection pool"],
    "queue":      ["kafka", "rabbitmq", "backlog", "consumer lag", "queue depth"],
    "network":    ["packet loss", "dns", "ssl", "tls"],
    "cache":      ["cache miss", "stale cache", "cache invalidat", "ttl"],
}

# Regex to match incident IDs in source paths (e.g. "INC-1000", "inc-1000")
_INCIDENT_ID_RE = re.compile(r"(INC-\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Query attribute extraction
# ---------------------------------------------------------------------------

@dataclass
class QueryAttributes:
    """Structured attributes extracted from a free-text query."""
    service: Optional[str] = None
    error_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class QueryAttributeExtractor:
    """
    Extracts service name, error types, and symptom keywords from a query
    using simple pattern matching — no LLM, no external calls.
    """

    _STOP = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "will", "would", "could",
        "should", "what", "how", "why", "when", "where", "which", "who",
        "to", "of", "in", "on", "for", "with", "and", "or", "but", "not",
        "after", "before", "during", "about", "tell", "me", "show", "find",
        "get", "all", "any", "incident", "issue", "problem", "error",
        "occurred", "happening", "related", "caused", "cause", "fix", "fixed",
    })

    def extract(self, query: str) -> QueryAttributes:
        lower = query.lower()
        tokens = re.findall(r"[a-z0-9_-]+", lower)

        # --- Service detection: check aliases then direct "-service" suffix ---
        service: Optional[str] = None
        for token in tokens:
            if token in _SERVICE_ALIASES:
                service = _SERVICE_ALIASES[token]
                break
            if token.endswith("-service"):
                service = token
                break

        # --- Error type detection: check synonyms against full lowercased query ---
        error_types: List[str] = []
        for etype, synonyms in _ERROR_TYPE_KEYWORDS.items():
            if any(syn in lower for syn in synonyms):
                error_types.append(etype)

        # --- General keywords: non-stopword alpha tokens longer than 2 chars ---
        keywords = [t for t in tokens if len(t) > 2 and t not in self._STOP]

        return QueryAttributes(service=service, error_types=error_types, keywords=keywords)


# ---------------------------------------------------------------------------
# In-memory structured incident store
# ---------------------------------------------------------------------------

class StructuredIncidentStore:
    """
    Loads and indexes structured incident JSON files from a directory.

    Thread-safe for read operations once `load()` has completed.
    """

    def __init__(self):
        self._incidents: List[dict] = []
        self._by_id: Dict[str, dict] = {}
        self._extractor = QueryAttributeExtractor()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, directory: str) -> int:
        """
        Load all *.json files from `directory`.  Safe to call multiple
        times — each call fully replaces the in-memory state.

        Returns:
            Number of incident files successfully loaded.
        """
        if not os.path.isdir(directory):
            logger.warning("Structured incidents directory not found: %s", directory)
            return 0

        incidents: List[dict] = []
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "incident_id" not in data:
                    data["incident_id"] = fname[:-5]  # strip .json
                incidents.append(data)
            except Exception as exc:
                logger.warning("Skipping incident file %s: %s", fname, exc)

        self._incidents = incidents
        self._by_id = {inc["incident_id"]: inc for inc in incidents}
        self._loaded = True
        logger.info(
            "StructuredIncidentStore: loaded %d incidents from %s",
            len(incidents), directory,
        )
        return len(incidents)

    def get(self, incident_id: str) -> Optional[dict]:
        """Return a single incident by ID, or None."""
        return self._by_id.get(incident_id)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_incident(self, incident: dict, attrs: QueryAttributes) -> float:
        """
        Compute a structured similarity score in [0, 1].

        Weight allocation:
            0.40  service match
            0.35  symptom keyword overlap
            0.25  root cause / resolution keyword match
        """
        score = 0.0

        # ── Service match (0.40) ─────────────────────────────────────────
        incident_service = incident.get("service", "").lower()
        if attrs.service:
            query_svc = attrs.service.lower()
            if query_svc == incident_service:
                score += 0.40
            elif query_svc.replace("-service", "") in incident_service:
                score += 0.20   # partial
        else:
            score += 0.10       # no service in query — small baseline

        # ── Symptom overlap (0.35) ────────────────────────────────────────
        if attrs.keywords:
            # Combine symptoms + root cause text for broad matching
            symptom_corpus = (
                " ".join(incident.get("symptoms", []))
                + " " + incident.get("confirmed_root_cause", "")
                + " " + incident.get("resolution_action", "")
            ).lower()
            corpus_tokens = set(re.findall(r"[a-z0-9]+", symptom_corpus))
            query_tokens = set(attrs.keywords)
            overlap = len(query_tokens & corpus_tokens)
            score += 0.35 * min(1.0, overlap / len(query_tokens))

        # ── Root cause / resolution keyword match (0.25) ─────────────────
        if attrs.keywords or attrs.error_types:
            cause_text = (
                incident.get("confirmed_root_cause", "")
                + " " + incident.get("resolution_action", "")
            ).lower()
            cause_tokens = set(re.findall(r"[a-z0-9]+", cause_text))

            # Keyword overlap with root cause text
            kw_overlap = (
                len(set(attrs.keywords) & cause_tokens) / max(len(attrs.keywords), 1)
                if attrs.keywords else 0.0
            )

            # Error-type synonym hit (binary: any match → 1.0)
            et_hit = 0.0
            for etype in attrs.error_types:
                synonyms = _ERROR_TYPE_KEYWORDS.get(etype, [])
                if any(syn in cause_text for syn in synonyms):
                    et_hit = 1.0
                    break

            score += 0.25 * max(kw_overlap, et_hit)

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.10,
    ) -> List[Tuple[dict, float]]:
        """
        Return (incident, structured_score) pairs ranked best-first.

        Args:
            query:     Raw user query.
            top_k:     Maximum results to return.
            min_score: Drop incidents below this threshold.
        """
        if not self._incidents:
            return []

        attrs = self._extractor.extract(query)
        logger.debug(
            "Structured query | service=%s | error_types=%s | keywords=%s",
            attrs.service, attrs.error_types, attrs.keywords[:6],
        )

        scored = [
            (inc, self.score_incident(inc, attrs))
            for inc in self._incidents
        ]
        scored = [(inc, s) for inc, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def build_structured_field(self, incident: dict, structured_score: float) -> dict:
        """
        Return the ``structured`` payload to embed in a result chunk.
        This is what surfaces in the API response under each source chunk.
        """
        return {
            "incident_id":       incident.get("incident_id", ""),
            "service":           incident.get("service", ""),
            "root_cause":        incident.get("confirmed_root_cause", ""),
            "resolution":        incident.get("resolution_action", ""),
            "services_affected": incident.get("services_affected", []),
            "structured_score":  round(structured_score, 4),
        }

    def make_synthetic_chunk(self, incident: dict, structured_score: float) -> dict:
        """
        Build a lightweight chunk from incident fields for injection into
        hybrid results when the incident was not retrieved by vector search.

        The synthetic chunk uses the structured_score as its retrieval score
        so it competes fairly with hybrid results during reranking.
        """
        symptoms_lines = "\n".join(f"- {s}" for s in incident.get("symptoms", []))
        text = (
            f"Incident {incident.get('incident_id', 'UNKNOWN')}"
            f" [{incident.get('service', 'unknown-service')}]\n"
            f"Symptoms:\n{symptoms_lines}\n"
            f"Root Cause: {incident.get('confirmed_root_cause', '')}\n"
            f"Resolution: {incident.get('resolution_action', '')}"
        )
        iid = incident.get("incident_id", "unknown")
        return {
            "id":           f"struct:{iid}",
            "text":         text,
            "source":       f"structured_incidents/{iid}.json",
            "score":        round(structured_score, 4),
            "metadata":     incident,
            "structured":   self.build_structured_field(incident, structured_score),
            "_is_synthetic": True,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def extract_incident_id(source: str) -> Optional[str]:
    """
    Extract an incident ID (e.g. "INC-1000") from a chunk's source path.
    Returns None if not found.
    """
    match = _INCIDENT_ID_RE.search(source or "")
    return match.group(1).upper() if match else None


def enrich_chunks_with_structured(
    chunks: List[dict],
    store: StructuredIncidentStore,
    structured_hits: List[Tuple[dict, float]],
    inject_limit: int = 2,
    inject_min_score: float = 0.50,
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Attach ``structured`` metadata to hybrid result chunks and optionally
    inject synthetic chunks for top structured matches that were not
    retrieved by vector search.

    Args:
        chunks:           Hybrid retrieval results (modified in place).
        store:            Loaded StructuredIncidentStore.
        structured_hits:  Output of store.search() — (incident, score) pairs.
        inject_limit:     Max number of synthetic chunks to inject.
        inject_min_score: Minimum structured score for injection.

    Returns:
        (enriched_chunks, chunk_structured_scores) where
        chunk_structured_scores maps chunk["id"] → structured_score for
        the reranker to consume.
    """
    # Build lookup: incident_id → structured_score
    incident_score_map: Dict[str, float] = {
        inc["incident_id"]: s for inc, s in structured_hits
    }

    # Track which incident IDs are already represented in hybrid results
    matched_incident_ids: set = set()

    chunk_structured_scores: Dict[str, float] = {}

    for chunk in chunks:
        source = chunk.get("source", "") or chunk.get("metadata", {}).get("source", "")
        iid = extract_incident_id(source)
        if iid and iid in incident_score_map:
            s_score = incident_score_map[iid]
            incident = store.get(iid)
            if incident:
                chunk["structured"] = store.build_structured_field(incident, s_score)
                chunk_structured_scores[chunk["id"]] = s_score
                matched_incident_ids.add(iid)
                logger.debug(
                    "Linked chunk %s → incident %s (structured_score=%.3f)",
                    chunk["id"], iid, s_score,
                )

    # Inject top unmatched structured hits as synthetic chunks
    injected = 0
    for incident, s_score in structured_hits:
        if injected >= inject_limit:
            break
        iid = incident["incident_id"]
        if iid in matched_incident_ids or s_score < inject_min_score:
            continue
        synthetic = store.make_synthetic_chunk(incident, s_score)
        chunks.append(synthetic)
        chunk_structured_scores[synthetic["id"]] = s_score
        matched_incident_ids.add(iid)
        injected += 1
        logger.debug(
            "Injected synthetic chunk for incident %s (structured_score=%.3f)",
            iid, s_score,
        )

    return chunks, chunk_structured_scores
