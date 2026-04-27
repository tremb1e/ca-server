from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..config import settings


management_api_key_header = APIKeyHeader(name="X-Management-API-Key", auto_error=False)


async def require_management_api_key(api_key: str | None = Security(management_api_key_header)) -> None:
    if not settings.management_api_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="management_api_disabled")
    expected = settings.management_api_key
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="management_api_key_not_configured")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_management_api_key")
    if not secrets.compare_digest(str(api_key), str(expected)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid_management_api_key")
