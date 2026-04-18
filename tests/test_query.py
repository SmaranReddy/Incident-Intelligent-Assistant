"""Tests for /kb/{kb_id}/query endpoint."""

import pytest
import numpy as np
from httpx import AsyncClient
from unittest.mock import MagicMock, patch

from app.core.models import KnowledgeBase
from tests.conftest import register_and_login


async def _setup(client: AsyncClient, email: str):
    """Register user, create a KB, return (token, kb_id)."""
    token = await register_and_login(client, email)
    r = await client.post(
        "/kb", headers={"Authorization": f"Bearer {token}"},
        json={"name": "QueryTestKB"},
    )
    kb_id = r.json()["id"]
    return token, kb_id


async def _mark_kb_ready(client: AsyncClient, kb_id: str) -> None:
    """Directly update KB index_status to 'ready' in the test DB."""
    async with client.test_session() as db:
        kb = await db.get(KnowledgeBase, kb_id)
        kb.index_status = "ready"
        await db.commit()


def _mock_index(text: str = "Policy content."):
    mock = MagicMock()
    mock.search.return_value = [
        {"id": "chunk-1", "text": text, "source": "policy.txt", "score": 0.9}
    ]
    return mock


async def test_query_empty_kb_returns_422(client: AsyncClient):
    token, kb_id = await _setup(client, "query_empty@example.com")

    r = await client.post(
        f"/kb/{kb_id}/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What is the vacation policy?"},
    )
    assert r.status_code == 422
    assert "no documents" in r.json()["detail"].lower()


async def test_query_indexed_kb_returns_answer(client: AsyncClient):
    token, kb_id = await _setup(client, "query_indexed@example.com")
    await _mark_kb_ready(client, kb_id)

    with patch("api.routes.query.kb_manager.get", return_value=_mock_index()), \
         patch("api.routes.query.kb_manager.embed_query",
               return_value=np.zeros((1, 384), dtype="float32")):

        r = await client.post(
            f"/kb/{kb_id}/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "What is the vacation policy?"},
        )

    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["confidence"], float)
    assert isinstance(data["latency_ms"], int)
    assert "cache_hit" in data
    assert "request_id" in data
    assert data["model_used"] is not None


async def test_query_cache_hit_on_second_request(client: AsyncClient):
    token, kb_id = await _setup(client, "query_cache@example.com")
    await _mark_kb_ready(client, kb_id)

    # Use a fresh mock generator to verify call count
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Cached answer."

    with patch("api.routes.query.kb_manager.get", return_value=_mock_index()), \
         patch("api.routes.query.kb_manager.embed_query",
               return_value=np.zeros((1, 384), dtype="float32")), \
         patch("api.routes.query._get_generator", return_value=mock_generator), \
         patch("api.routes.query._cache.get", side_effect=[None, {"answer": "Cached answer.",
               "sources": [], "confidence": 0.9, "latency_ms": 100,
               "cache_hit": False, "model_used": "test"}]):

        headers = {"Authorization": f"Bearer {token}"}
        payload = {"query": "What is the cache test?"}

        r1 = await client.post(f"/kb/{kb_id}/query", headers=headers, json=payload)
        r2 = await client.post(f"/kb/{kb_id}/query", headers=headers, json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    # First call: cache miss → LLM called; second call: cache hit → LLM not called
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is True
    assert mock_generator.generate.call_count == 1


async def test_query_missing_token_returns_401(client: AsyncClient):
    r = await client.post(
        "/kb/some-kb-id/query",
        json={"query": "test"},
    )
    assert r.status_code == 401


async def test_query_other_orgs_kb_returns_404(client: AsyncClient):
    token_a = await register_and_login(client, "query_other_a@example.com")
    token_b = await register_and_login(client, "query_other_b@example.com")

    r = await client.post(
        "/kb", headers={"Authorization": f"Bearer {token_a}"},
        json={"name": "OrgAKB"},
    )
    kb_id = r.json()["id"]

    r2 = await client.post(
        f"/kb/{kb_id}/query",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"query": "test"},
    )
    assert r2.status_code == 404
