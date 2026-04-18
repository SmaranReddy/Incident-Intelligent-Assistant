"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    org_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    org_id: str
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Organization
# ---------------------------------------------------------------------------

class OrgOut(BaseModel):
    id: str
    name: str
    slug: str
    plan: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class KBUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None


class KBOut(BaseModel):
    id: str
    org_id: str
    name: str
    slug: str
    description: Optional[str]
    doc_count: int
    chunk_count: int
    index_status: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size_bytes: Optional[int]
    status: str
    error_message: Optional[str]
    chunk_count: int
    created_at: datetime
    indexed_at: Optional[datetime]

    class Config:
        from_attributes = True


class DocumentListOut(BaseModel):
    documents: List[DocumentOut]
    total: int


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_ids: Optional[List[str]] = None   # if set, restrict retrieval to these doc_ids


class StructuredInfo(BaseModel):
    """Structured incident metadata attached to a result chunk when the
    chunk can be linked to a known incident record."""
    incident_id: str
    service: str
    root_cause: str
    resolution: str
    services_affected: List[str]
    structured_score: float


class SourceChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float
    structured: Optional[StructuredInfo] = None


class SimilarIncident(BaseModel):
    incident_id: str
    service: str
    score: float


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    sources: List[SourceChunk]
    confidence: float
    latency_ms: int
    cache_hit: bool
    model_used: str
    likely_root_cause: Optional[str] = None
    suggested_fix: Optional[str] = None
    similar_incidents: List[SimilarIncident] = []
    affected_services: List[str] = []


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackCreate(BaseModel):
    rating: Literal[-1, 1]
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackOut(BaseModel):
    id: str
    query_log_id: str
    rating: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[Literal["query", "ingest", "admin"]] = ["query"]
    rate_limit_rpm: int = Field(default=60, ge=1, le=1000)


class APIKeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: List[str]
    rate_limit_rpm: int
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyCreated(APIKeyOut):
    """Returned only once at creation — includes the raw key."""
    raw_key: str


# ---------------------------------------------------------------------------
# Metrics / Analytics
# ---------------------------------------------------------------------------

class UsageSummary(BaseModel):
    total_queries: int
    total_documents: int
    total_chunks: int
    avg_latency_ms: Optional[float]
    cache_hit_rate: Optional[float]


class FeedbackSummary(BaseModel):
    total_feedback: int
    thumbs_up: int
    thumbs_down: int
    satisfaction_rate: float


class LatencyStats(BaseModel):
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str = "1.0.0"
