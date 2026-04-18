import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        extra="ignore",
    )

    # App
    APP_NAME: str = "Enterprise Knowledge Platform"
    DEBUG: bool = False

    # Database — SQLite by default, override with postgresql+asyncpg://... for prod
    DATABASE_URL: str = "sqlite+aiosqlite:///./enterprise_kb.db"

    # Auth — SECRET_KEY has no default; must be set in .env
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_HOURS: int = 8
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # Groq
    GROQ_API_KEY: str = ""

    # Storage — local filesystem path for uploaded documents
    UPLOAD_DIR: str = "backend/data/uploads"

    # Index storage — per-KB FAISS + BM25 indexes
    INDEX_DIR: str = "backend/data/indexes"

    # Structured incident knowledge layer — absolute path resolved from project root
    STRUCTURED_INCIDENTS_DIR: str = str(
        Path(__file__).resolve().parents[3] / "data" / "structured_incidents"
    )
    # Max synthetic incident chunks to inject per query (0 = disabled)
    STRUCTURED_INJECT_LIMIT: int = 2
    # Minimum structured score to qualify for injection
    STRUCTURED_INJECT_MIN_SCORE: float = 0.50

    # Embedding model (must match the model used to build existing FAISS index)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Ingestion defaults
    CHUNK_SIZE_TOKENS: int = 400      # target 200-500 tokens per chunk
    CHUNK_OVERLAP_TOKENS: int = 50    # overlap 20-50 tokens
    MIN_CHUNK_TOKENS: int = 50        # merge chunks smaller than this
    MAX_UPLOAD_MB: int = 50

    # Query pipeline
    ENABLE_QUERY_REWRITING: bool = False   # opt-in: rewrites vague queries via LLM
    TOP_N_DOCS: int = 5                    # Stage 1: number of documents to pre-select

    # Rate limiting defaults (requests per minute)
    DEFAULT_RATE_LIMIT_RPM: int = 60

    # Query result cache
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_SIZE: int = 500

    # LLM
    LLM_MODEL: str = "llama-3.1-8b-instant"


settings = Settings()

# Ensure storage directories exist at import time
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
