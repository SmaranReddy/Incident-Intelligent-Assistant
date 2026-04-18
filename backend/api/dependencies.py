"""
FastAPI dependency injection.

Every protected route uses get_current_principal() which resolves
either a JWT (web user) or an API key (programmatic access).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api_key import verify_api_key
from app.auth.jwt import decode_token
from app.core.database import get_db
from app.core.models import APIKey, User

# Per-API-key request counters — each entry holds a count, expires after 60s
# maxsize covers up to 10,000 active API keys simultaneously
_rate_counters: TTLCache = TTLCache(maxsize=10_000, ttl=60)

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Principal — unified auth context
# ---------------------------------------------------------------------------

@dataclass
class Principal:
    org_id: str
    role: str = "member"
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    scopes: list = field(default_factory=list)
    rate_limit_rpm: int = 60


# ---------------------------------------------------------------------------
# Resolve auth from header
# ---------------------------------------------------------------------------

async def get_current_principal(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # --- API Key path ---
    if token.startswith("ek_"):
        api_key: Optional[APIKey] = await verify_api_key(token, db)
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key",
            )
        return Principal(
            org_id=api_key.org_id,
            role="member",         # API keys have member-level role
            api_key_id=api_key.id,
            scopes=api_key.scopes or [],
            rate_limit_rpm=api_key.rate_limit_rpm,
        )

    # --- JWT path ---
    payload = decode_token(token)
    return Principal(
        org_id=payload["org_id"],
        role=payload.get("role", "member"),
        user_id=payload.get("sub"),
        scopes=["query", "ingest", "manage_kb", "manage_users", "view_metrics"],
        rate_limit_rpm=300,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def require_scope(scope: str):
    """Dependency: raises 403 if principal lacks the given scope."""
    async def _check(principal: Principal = Depends(get_current_principal)) -> Principal:
        # JWT users have all scopes; API key users are checked against their scope list
        if principal.api_key_id and scope not in principal.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required scope: '{scope}'",
            )
        return principal
    return _check


def require_role(permission: str):
    """Dependency: raises 403 if the user's role lacks the given permission."""
    from app.auth.permissions import check_permission
    async def _check(principal: Principal = Depends(get_current_principal)) -> Principal:
        if not principal.api_key_id:   # only role-check JWT users
            check_permission(principal.role, permission)
        return principal
    return _check


def check_rate_limit(principal: Principal) -> None:
    """
    Enforce per-API-key rate limits (requests per minute).
    Only applied to API key principals — JWT users are covered by the global slowapi limit.
    Raises HTTP 429 if the key has exceeded its rate_limit_rpm in the current minute.
    """
    if not principal.api_key_id:
        return

    key = principal.api_key_id
    current = _rate_counters.get(key, 0)
    if current >= principal.rate_limit_rpm:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {principal.rate_limit_rpm} requests/minute",
        )
    _rate_counters[key] = current + 1
