from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class FileInfo(FlexibleModel):
    path: str
    exists: bool
    size_bytes: int = 0
    size_mb: float = 0.0
    modified_at: Optional[str] = None


class StorageStats(FlexibleModel):
    base_path: str
    total_devices: int = 0
    total_sessions: int = 0
    total_size_bytes: int = 0
    total_size_mb: float = 0.0


class RuntimeResponse(FlexibleModel):
    app_name: str
    version: str
    started_at: str
    uptime_seconds: float
    http: Dict[str, Any]
    grpc: Dict[str, Any]
    tls: Dict[str, Any]
    backend: Dict[str, Any]
    metrics: Dict[str, Any]
    training_tasks: Dict[str, Any]
    model_cache: Dict[str, Any]


class SummaryResponse(FlexibleModel):
    status: str
    runtime: Dict[str, Any]
    storage: Dict[str, Any]
    counts: Dict[str, int]
    recent_errors: List[Dict[str, Any]]


class DeviceListItem(FlexibleModel):
    device_id_hash: str
    raw: Dict[str, Any]
    processed: Dict[str, Any]
    model: Dict[str, Any]
    inference: Dict[str, Any]
    training: Dict[str, Any]


class DeviceListResponse(FlexibleModel):
    devices: List[DeviceListItem]
    total: int


class DeviceDetailResponse(FlexibleModel):
    device_id_hash: str
    raw: Dict[str, Any]
    processed: Dict[str, Any]
    inference: Dict[str, Any]
    training: Dict[str, Any]
    model: Dict[str, Any]
    active_auth_sessions: List[Dict[str, Any]]


class TrainingStatusResponse(FlexibleModel):
    device_id_hash: str
    status: str
    total_bytes: int
    total_mb: float
    min_bytes: int
    min_mb: float
    has_enough_data: bool
    is_ready: bool
    last_trained_bytes: int
    last_error: str
    updated_at: str
    task: Optional[Dict[str, Any]] = None


class RawSessionsResponse(FlexibleModel):
    device_id_hash: str
    sessions: List[Dict[str, Any]]
    total: int


class ModelInfoResponse(FlexibleModel):
    device_id_hash: str
    ready: bool
    policy: Optional[Dict[str, Any]] = None
    files: Dict[str, FileInfo]
    training_summary: List[Dict[str, Any]]
    policy_search: Dict[str, Any]
    error: Optional[str] = None


class AuthSessionsResponse(FlexibleModel):
    device_id_hash: str
    sessions: List[Dict[str, Any]]
    total: int


class AuthResultsResponse(FlexibleModel):
    results: List[Dict[str, Any]]
    total: int


class ClientMetricsResponse(FlexibleModel):
    latest_by_device: Dict[str, Dict[str, Any]]
    recent: List[Dict[str, Any]]
