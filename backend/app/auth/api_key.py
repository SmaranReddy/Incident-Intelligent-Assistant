"""API key generation and verification."""

import hashlib
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import APIKey


def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (raw_key, key_hash, key_prefix).
    Only raw_key is shown to the user once — store key_hash in DB.
    """
    raw = f"ek_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    key_prefix = raw[:12] + "..."   # e.g. "ek_abc12345..."
    return raw, key_hash, key_prefix


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def verify_api_key(raw_key: str, db: AsyncSession) -> Optional[APIKey]:
    """Look up API key by hash. Returns None if not found, revoked, or expired."""
    key_hash = hash_key(raw_key)
    result = await db.execute(
        select(APIKey).where(
            APIKey.key_hash == key_hash,
            APIKey.revoked_at.is_(None),
        )
    )
    api_key = result.scalar_one_or_none()

    if api_key is None:
        return None

    # Check expiry
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        return None

    # Update last_used_at (best-effort, non-blocking)
    api_key.last_used_at = datetime.utcnow()

    return api_key
