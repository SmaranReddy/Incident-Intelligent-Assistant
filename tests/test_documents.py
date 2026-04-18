"""Tests for /kb/{kb_id}/documents endpoints."""

import io
import pytest
from httpx import AsyncClient

from tests.conftest import register_and_login

pytestmark = pytest.mark.asyncio


async def _create_kb(client: AsyncClient, token: str, name: str = "DocTestKB") -> str:
    r = await client.post("/kb", headers={"Authorization": f"Bearer {token}"}, json={"name": name})
    return r.json()["id"]


async def test_upload_valid_text_file(client: AsyncClient):
    token = await register_and_login(client, "doc_upload@example.com")
    kb_id = await _create_kb(client, token)

    r = await client.post(
        f"/kb/{kb_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("sample.txt", io.BytesIO(b"Hello, this is test content."), "text/plain")},
    )
    assert r.status_code == 202
    data = r.json()
    assert data["filename"] == "sample.txt"
    assert data["status"] == "pending"
    assert data["file_type"] == "txt"


async def test_upload_empty_file_returns_400(client: AsyncClient):
    token = await register_and_login(client, "doc_empty@example.com")
    kb_id = await _create_kb(client, token)

    r = await client.post(
        f"/kb/{kb_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert r.status_code == 400


async def test_upload_oversized_file_returns_413(client: AsyncClient):
    token = await register_and_login(client, "doc_oversize@example.com")
    kb_id = await _create_kb(client, token)

    # Create a file slightly larger than MAX_UPLOAD_MB (default 50 MB)
    big_content = b"x" * (51 * 1024 * 1024)
    r = await client.post(
        f"/kb/{kb_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("big.txt", io.BytesIO(big_content), "text/plain")},
    )
    assert r.status_code == 413


async def test_get_document_status(client: AsyncClient):
    token = await register_and_login(client, "doc_status@example.com")
    kb_id = await _create_kb(client, token)

    upload_r = await client.post(
        f"/kb/{kb_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("status.txt", io.BytesIO(b"Some content here."), "text/plain")},
    )
    doc_id = upload_r.json()["id"]

    r = await client.get(
        f"/kb/{kb_id}/documents/{doc_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["id"] == doc_id


async def test_list_documents(client: AsyncClient):
    token = await register_and_login(client, "doc_list@example.com")
    kb_id = await _create_kb(client, token)

    for i in range(3):
        await client.post(
            f"/kb/{kb_id}/documents",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (f"file{i}.txt", io.BytesIO(b"content"), "text/plain")},
        )

    r = await client.get(f"/kb/{kb_id}/documents", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["total"] == 3
    assert len(r.json()["documents"]) == 3


async def test_upload_to_other_orgs_kb_returns_404(client: AsyncClient):
    token_a = await register_and_login(client, "doc_other_a@example.com")
    token_b = await register_and_login(client, "doc_other_b@example.com")
    kb_id = await _create_kb(client, token_a)

    r = await client.post(
        f"/kb/{kb_id}/documents",
        headers={"Authorization": f"Bearer {token_b}"},
        files={"file": ("evil.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert r.status_code == 404
