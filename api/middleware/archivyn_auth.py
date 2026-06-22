from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

import jwt
from fastapi import HTTPException, Request

COOKIE_NAME = os.getenv("ARCHIVYN_SESSION_COOKIE", "archivyn_session")
JWT_ALG = "HS256"


def _auth_required() -> bool:
    raw = os.getenv("AUTH_REQUIRED", "false")
    return raw.lower() in {"1", "true", "yes"}


def _allowed_roles() -> Set[str]:
    raw = os.getenv("ARCHIVYN_AUTH_ALLOWED_ROLES", "company_admin")
    return {role.strip() for role in raw.split(",") if role.strip()}


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="JWT_SECRET is not configured on the automation backend.")
    return secret


def _extract_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return request.cookies.get(COOKIE_NAME)


def _validate_session_jti(jti: str) -> bool:
    if os.getenv("ARCHIVYN_AUTH_VALIDATE_SESSION", "false").lower() not in {"1", "true", "yes"}:
        return True

    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    if not mongodb_uri:
        return True

    try:
        from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

        client = create_mongo_client(mongodb_uri)
        db_name = resolve_mongo_db_name(mongodb_uri)
        collection = os.getenv("ARCHIVYN_ADMIN_SESSIONS_COLLECTION", "admin_sessions")
        record = client[db_name][collection].find_one({"jti": jti}, {"_id": 1})
        client.close()
        return record is not None
    except Exception:
        return False


def _role_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("role", "roles"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first
    return None


def _roles_from_payload(payload: Dict[str, Any]) -> Set[str]:
    roles: Set[str] = set()
    role = payload.get("role")
    if isinstance(role, str) and role:
        roles.add(role)
    raw_roles = payload.get("roles")
    if isinstance(raw_roles, list):
        roles.update(str(entry) for entry in raw_roles if entry)
    return roles


async def require_archivyn_auth(request: Request) -> None:
    if not _auth_required():
        return

    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="not_authenticated")

    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALG])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid_session") from exc

    jti = payload.get("jti")
    if isinstance(jti, str) and jti and not _validate_session_jti(jti):
        raise HTTPException(status_code=401, detail="session_revoked")

    allowed = _allowed_roles()
    roles = _roles_from_payload(payload)
    if allowed and not roles.intersection(allowed):
        primary_role = _role_from_payload(payload)
        if primary_role not in allowed:
            raise HTTPException(status_code=403, detail="forbidden")
