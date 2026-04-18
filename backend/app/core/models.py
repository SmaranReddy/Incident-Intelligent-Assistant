"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, SmallInteger, String, Text, JSON,
    func,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Organization (tenant isolation unit)
# ---------------------------------------------------------------------------

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    plan = Column(String, default="free")  # free | pro | enterprise
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="org", cascade="all, delete")
    knowledge_bases = relationship("KnowledgeBase", back_populates="org", cascade="all, delete")
    api_keys = relationship("APIKey", back_populates="org", cascade="all, delete")


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="member")  # owner | admin | member | viewer
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    org = relationship("Organization", back_populates="users")
    feedback = relationship("Feedback", back_populates="user")


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String, primary_key=True, default=_uuid)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    doc_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    index_status = Column(String, default="empty")  # empty | building | ready | error
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    org = relationship("Organization", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="kb", cascade="all, delete")
    query_logs = relationship("QueryLog", back_populates="kb")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    kb_id = Column(String, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)   # local path or S3 key
    file_type = Column(String, nullable=False)      # pdf | docx | txt | md
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(String, default="pending")      # pending | processing | indexed | failed
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    doc_metadata = Column(JSON, default=dict)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)

    kb = relationship("KnowledgeBase", back_populates="documents")


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=_uuid)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    name = Column(String, nullable=False)
    key_hash = Column(String, unique=True, nullable=False)  # SHA-256 of raw key
    key_prefix = Column(String, nullable=False)             # "ek_xxxx" shown in UI
    scopes = Column(JSON, default=list)                     # ["query", "ingest", "admin"]
    rate_limit_rpm = Column(Integer, default=60)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    org = relationship("Organization", back_populates="api_keys")


# ---------------------------------------------------------------------------
# Query Log
# ---------------------------------------------------------------------------

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=_uuid)
    org_id = Column(String, nullable=False)
    kb_id = Column(String, ForeignKey("knowledge_bases.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    api_key_id = Column(String, ForeignKey("api_keys.id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    chunk_ids = Column(JSON, default=list)   # list of chunk IDs used
    confidence = Column(Float, nullable=True)
    model_used = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    cache_hit = Column(Boolean, default=False)
    token_count_in = Column(Integer, nullable=True)
    token_count_out = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    kb = relationship("KnowledgeBase", back_populates="query_logs")
    feedback = relationship("Feedback", back_populates="query_log", uselist=False)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=_uuid)
    query_log_id = Column(String, ForeignKey("query_logs.id"), nullable=False)
    org_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    rating = Column(SmallInteger, nullable=False)   # 1 = thumbs up, -1 = thumbs down
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    query_log = relationship("QueryLog", back_populates="feedback")
    user = relationship("User", back_populates="feedback")
