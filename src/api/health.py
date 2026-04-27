from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from ..config import settings
from ..management.runtime import get_runtime_context
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    try:
        storage = get_runtime_context().storage
        storage_stats = storage.get_storage_stats()

        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "app_name": settings.app_name,
                "version": settings.version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "storage_stats": storage_stats
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/")
async def root():
    return JSONResponse(
        content={
            "app_name": settings.app_name,
            "version": settings.version,
            "status": "running"
        }
    )
