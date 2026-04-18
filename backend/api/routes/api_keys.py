"""API key management routes."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, require_role
from app.auth.api_key import generate_api_key
from app.core.database import get_db
from app.core.models import APIKey
from app.core.schemas import APIKeyCreate, APIKeyCreated, APIKeyOut

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


@router.get("", response_model=List[APIKeyOut])
async def list_api_keys(
    principal: Principal = Depends(require_role("manage_users")),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(APIKey).where(
            APIKey.org_id == principal.org_id,
            APIKey.revoked_at.is_(None),
        )
    )
    return result.scalars().all()


@router.post("", response_model=APIKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: APIKeyCreate,
    principal: Principal = Depends(require_role("manage_users")),
    db: AsyncSession = Depends(get_db),
):
    raw_key, key_hash, key_prefix = generate_api_key()

    api_key = APIKey(
        org_id=principal.org_id,
        created_by=principal.user_id,
        name=body.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        scopes=body.scopes,
        rate_limit_rpm=body.rate_limit_rpm,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return APIKeyCreated(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        scopes=api_key.scopes,
        rate_limit_rpm=api_key.rate_limit_rpm,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        raw_key=raw_key,   # shown ONCE only
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    principal: Principal = Depends(require_role("manage_users")),
    db: AsyncSession = Depends(get_db),
):
    from datetime import datetime

    api_key = await db.get(APIKey, key_id)
    if not api_key or api_key.org_id != principal.org_id:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.revoked_at = datetime.utcnow()
    await db.commit()
