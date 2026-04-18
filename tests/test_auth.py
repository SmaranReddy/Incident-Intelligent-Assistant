"""Tests for /auth endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import register_and_login

pytestmark = pytest.mark.asyncio


async def test_register_creates_user(client: AsyncClient):
    r = await client.post("/auth/register", json={
        "org_name": "Acme Corp",
        "email": "acme@example.com",
        "password": "Password123!",
    })
    assert r.status_code == 201
    assert "access_token" in r.json()
    assert r.json()["token_type"] == "bearer"


async def test_register_duplicate_email_returns_409(client: AsyncClient):
    payload = {"org_name": "DupOrg", "email": "dup@example.com", "password": "Password123!"}
    await client.post("/auth/register", json=payload)
    r = await client.post("/auth/register", json=payload)
    assert r.status_code == 409


async def test_login_valid_credentials(client: AsyncClient):
    await client.post("/auth/register", json={
        "org_name": "LoginOrg", "email": "login@example.com", "password": "Password123!",
    })
    r = await client.post("/auth/login", json={
        "email": "login@example.com", "password": "Password123!",
    })
    assert r.status_code == 200
    assert "access_token" in r.json()


async def test_login_wrong_password_returns_401(client: AsyncClient):
    await client.post("/auth/register", json={
        "org_name": "WrongPwOrg", "email": "wrongpw@example.com", "password": "Password123!",
    })
    r = await client.post("/auth/login", json={
        "email": "wrongpw@example.com", "password": "WrongPassword!",
    })
    assert r.status_code == 401


async def test_me_returns_user_profile(client: AsyncClient):
    token = await register_and_login(client, "me@example.com")
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert data["email"] == "me@example.com"
    assert data["role"] == "owner"


async def test_me_without_token_returns_401(client: AsyncClient):
    r = await client.get("/auth/me")
    assert r.status_code == 401


async def test_me_with_invalid_token_returns_401(client: AsyncClient):
    r = await client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert r.status_code == 401
