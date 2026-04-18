"""Role-based access control."""

from fastapi import HTTPException, status


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": {"query"},
    "member": {"query", "ingest"},
    "admin":  {"query", "ingest", "manage_kb", "manage_users", "view_metrics"},
    "owner":  {"query", "ingest", "manage_kb", "manage_users", "view_metrics", "billing"},
}


def check_permission(role: str, permission: str) -> None:
    """Raise 403 if the role does not have the given permission."""
    allowed = ROLE_PERMISSIONS.get(role, set())
    if permission not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your role '{role}' does not have '{permission}' permission.",
        )


def check_scope(scopes: list[str], required: str) -> None:
    """Raise 403 if required scope is not in the API key's scopes."""
    if required not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key missing required scope: '{required}'",
        )
