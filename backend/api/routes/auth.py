"""Auth routes: register, login, me."""

import re
from datetime import datetime

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import Principal, get_current_principal
from app.auth.jwt import create_access_token
from app.core.database import get_db
from app.core.models import Organization, User
from app.core.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------------------
# Register — creates org + owner user in one shot
# ---------------------------------------------------------------------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    slug = _slug(body.org_name)
    slug_check = await db.execute(select(Organization).where(Organization.slug == slug))
    if slug_check.scalar_one_or_none():
        slug = f"{slug}-{body.email.split('@')[0]}"

    org = Organization(name=body.org_name, slug=slug)
    db.add(org)
    await db.flush()

    user = User(
        org_id=org.id,
        email=body.email,
        password_hash=_hash_password(body.password),
        role="owner",
    )
    db.add(user)
    await db.commit()

    return TokenResponse(access_token=create_access_token(user.id, org.id, user.role))


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user.last_login_at = datetime.utcnow()
    await db.commit()

    return TokenResponse(access_token=create_access_token(user.id, user.org_id, user.role))


# ---------------------------------------------------------------------------
# Current user profile
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserOut)
async def me(
    principal: Principal = Depends(get_current_principal),
    db: AsyncSession = Depends(get_db),
):
    if not principal.user_id:
        raise HTTPException(
            status_code=403,
            detail="API keys cannot access /auth/me — use a user JWT token",
        )
    user = await db.get(User, principal.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
