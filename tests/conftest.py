"""
Shared pytest fixtures.

- Uses an in-memory SQLite database (isolated per test function)
- Mocks the LLM (AnswerGenerator) so tests never hit Groq
- Provides an async HTTP client wired to the FastAPI app
"""

import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("GROQ_API_KEY", "test-key")

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from api.main import app


# ---------------------------------------------------------------------------
# Per-test in-memory DB engine + session factory
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    """
    Async HTTP client with:
    - A fresh in-memory SQLite DB (tables created, dropped after each test)
    - get_db overridden to use the test DB
    - AnswerGenerator mocked so no real LLM calls are made
    """
    # StaticPool ensures all connections share the same in-memory SQLite
    # database — without it, each new connection sees an empty DB.
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def _override_get_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    # Expose the test session factory on the client for query tests that need
    # to manipulate DB state directly (e.g. marking a KB as ready)
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Mocked answer from the knowledge base."

    with patch("api.routes.query._get_generator", return_value=mock_generator), \
         patch("api.routes.documents.run_ingestion"):
        # run_ingestion is patched to a no-op: document upload tests verify the
        # API contract (202, correct response shape), not the ingestion pipeline.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            ac.test_session = TestSession   # attach for use in query tests
            yield ac

    app.dependency_overrides.clear()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

async def register_and_login(client: AsyncClient, email: str = "user@example.com") -> str:
    """Register a new org+user and return the JWT access token."""
    await client.post("/auth/register", json={
        "org_name": f"Org-{email}",
        "email": email,
        "password": "Password123!",
    })
    r = await client.post("/auth/login", json={
        "email": email,
        "password": "Password123!",
    })
    return r.json()["access_token"]
