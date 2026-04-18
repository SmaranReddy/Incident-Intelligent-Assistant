"""Tests for /kb (knowledge base CRUD) endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import register_and_login

pytestmark = pytest.mark.asyncio


async def test_create_kb(client: AsyncClient):
    token = await register_and_login(client, "kb_create@example.com")
    r = await client.post("/kb", headers={"Authorization": f"Bearer {token}"}, json={
        "name": "My KB", "description": "Test knowledge base",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "My KB"
    assert data["index_status"] == "empty"
    assert data["doc_count"] == 0


async def test_list_kbs_returns_own_kbs_only(client: AsyncClient):
    token_a = await register_and_login(client, "kb_list_a@example.com")
    token_b = await register_and_login(client, "kb_list_b@example.com")

    await client.post("/kb", headers={"Authorization": f"Bearer {token_a}"}, json={"name": "A-KB"})
    await client.post("/kb", headers={"Authorization": f"Bearer {token_a}"}, json={"name": "A-KB-2"})
    await client.post("/kb", headers={"Authorization": f"Bearer {token_b}"}, json={"name": "B-KB"})

    r = await client.get("/kb", headers={"Authorization": f"Bearer {token_a}"})
    assert r.status_code == 200
    names = [kb["name"] for kb in r.json()]
    assert "A-KB" in names
    assert "A-KB-2" in names
    assert "B-KB" not in names


async def test_get_kb_by_id(client: AsyncClient):
    token = await register_and_login(client, "kb_get@example.com")
    create_r = await client.post("/kb", headers={"Authorization": f"Bearer {token}"}, json={"name": "GetKB"})
    kb_id = create_r.json()["id"]

    r = await client.get(f"/kb/{kb_id}", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["id"] == kb_id


async def test_get_kb_from_other_org_returns_404(client: AsyncClient):
    token_a = await register_and_login(client, "kb_403_a@example.com")
    token_b = await register_and_login(client, "kb_403_b@example.com")

    r = await client.post("/kb", headers={"Authorization": f"Bearer {token_a}"}, json={"name": "PrivateKB"})
    kb_id = r.json()["id"]

    r2 = await client.get(f"/kb/{kb_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert r2.status_code == 404


async def test_update_kb(client: AsyncClient):
    token = await register_and_login(client, "kb_update@example.com")
    r = await client.post("/kb", headers={"Authorization": f"Bearer {token}"}, json={"name": "OldName"})
    kb_id = r.json()["id"]

    r2 = await client.patch(f"/kb/{kb_id}", headers={"Authorization": f"Bearer {token}"}, json={"name": "NewName"})
    assert r2.status_code == 200
    assert r2.json()["name"] == "NewName"


async def test_delete_kb(client: AsyncClient):
    token = await register_and_login(client, "kb_delete@example.com")
    r = await client.post("/kb", headers={"Authorization": f"Bearer {token}"}, json={"name": "ToDelete"})
    kb_id = r.json()["id"]

    r2 = await client.delete(f"/kb/{kb_id}", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 204

    r3 = await client.get(f"/kb/{kb_id}", headers={"Authorization": f"Bearer {token}"})
    assert r3.status_code == 404
