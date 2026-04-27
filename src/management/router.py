from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from . import snapshot
from .runtime import get_runtime_context
from .schemas import (
    AuthResultsResponse,
    AuthSessionsResponse,
    ClientMetricsResponse,
    DeviceDetailResponse,
    DeviceListResponse,
    ModelInfoResponse,
    RawSessionsResponse,
    RuntimeResponse,
    SummaryResponse,
    TrainingStatusResponse,
)
from .security import require_management_api_key
from ..utils.path_safety import UnsafePathSegmentError, validate_storage_id

router = APIRouter(
    prefix="/api/v1/management",
    tags=["management"],
    dependencies=[Depends(require_management_api_key)],
)


def _device_id(value: str) -> str:
    try:
        return validate_storage_id(value, field_name="device_id")
    except UnsafePathSegmentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _session_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return validate_storage_id(value, field_name="session_id")
    except UnsafePathSegmentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/summary", response_model=SummaryResponse)
async def get_summary():
    return snapshot.summary_snapshot(get_runtime_context())


@router.get("/runtime", response_model=RuntimeResponse)
async def get_runtime():
    return snapshot.runtime_snapshot(get_runtime_context())


@router.get("/devices", response_model=DeviceListResponse)
async def list_devices():
    ctx = get_runtime_context()
    devices = [snapshot.device_list_item(device_id, ctx) for device_id in snapshot.get_device_ids()]
    return {"devices": devices, "total": len(devices)}


@router.get("/devices/{device_id}", response_model=DeviceDetailResponse)
async def get_device(device_id: str):
    return snapshot.device_detail(_device_id(device_id), get_runtime_context())


@router.get("/devices/{device_id}/raw-sessions", response_model=RawSessionsResponse)
async def list_raw_sessions(device_id: str, limit: int = Query(default=100, ge=1, le=1000)):
    return snapshot.raw_sessions(_device_id(device_id), limit=limit)


@router.get("/devices/{device_id}/training", response_model=TrainingStatusResponse)
async def get_training(device_id: str):
    return snapshot.training_status(_device_id(device_id), get_runtime_context())


@router.get("/devices/{device_id}/models", response_model=ModelInfoResponse)
async def get_models(device_id: str):
    return snapshot.model_info(_device_id(device_id))


@router.get("/devices/{device_id}/auth/sessions", response_model=AuthSessionsResponse)
async def get_auth_sessions(device_id: str):
    device_id = _device_id(device_id)
    sessions = [s for s in get_runtime_context().auth_manager.snapshot_sessions() if s.get("user_id") == device_id]
    return {"device_id_hash": device_id, "sessions": sessions, "total": len(sessions)}


@router.get("/devices/{device_id}/auth/results", response_model=AuthResultsResponse)
async def get_auth_results(
    device_id: str,
    session_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    return snapshot.auth_results(_device_id(device_id), session_id=_session_id(session_id), limit=limit)


@router.get("/auth/results/latest", response_model=AuthResultsResponse)
async def get_latest_auth_results(limit: int = Query(default=100, ge=1, le=1000)):
    return snapshot.latest_auth_results(limit=limit)


@router.get("/client-metrics", response_model=ClientMetricsResponse)
async def get_client_metrics():
    return snapshot.client_metrics_snapshot(get_runtime_context())
