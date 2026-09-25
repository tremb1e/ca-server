import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path

import lz4.frame
import pytest
from ca_train.reconstruction import SENSOR_INPUT_MASKING_VERSION
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import health, sensor_data
from src.config import settings
from src.grpc_server import SensorDataService
from src.management.router import router as management_router
from src.management.runtime import get_runtime_context, reset_runtime_context
from src.protos import sensor_data_pb2


API_KEY = "contract-management-key"


def _configure_runtime(tmp_path: Path, monkeypatch, *, management_enabled: bool, api_key: str | None) -> None:
    raw_root = tmp_path / "raw_data"
    processed_root = tmp_path / "processed_data"
    inference_root = tmp_path / "inference"
    log_root = tmp_path / "logs"
    for path in (raw_root, processed_root, inference_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "data_storage_path", raw_root)
    monkeypatch.setattr(settings, "processed_data_path", processed_root)
    monkeypatch.setattr(settings, "inference_storage_path", inference_root)
    monkeypatch.setattr(settings, "log_path", log_root)
    monkeypatch.setattr(settings, "host", "127.0.0.1")
    monkeypatch.setattr(settings, "port", 18000)
    monkeypatch.setattr(settings, "http_enabled", True)
    monkeypatch.setattr(settings, "grpc_host", "127.0.0.1")
    monkeypatch.setattr(settings, "grpc_port", 10500)
    monkeypatch.setattr(settings, "grpc_max_message_size", 4 * 1024 * 1024)
    monkeypatch.setattr(settings, "management_api_enabled", management_enabled)
    monkeypatch.setattr(settings, "management_api_key", api_key)
    reset_runtime_context()


def _contract_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(health.router)
    app.include_router(sensor_data.router)
    if settings.management_api_enabled:
        app.include_router(management_router)
    return app


@pytest.fixture()
def contract_client(tmp_path, monkeypatch):
    _configure_runtime(tmp_path, monkeypatch, management_enabled=True, api_key=API_KEY)
    client = TestClient(_contract_app())
    yield client
    reset_runtime_context()


def _seed_contract_device(device_id: str = "contract-device") -> str:
    raw_root = Path(settings.data_storage_path)
    processed_root = Path(settings.processed_data_path)
    inference_root = Path(settings.inference_storage_path)
    models_root = raw_root.parent / "models"

    raw_device = raw_root / device_id
    raw_device.mkdir(parents=True, exist_ok=True)
    (raw_device / "session_raw-session.jsonl").write_text(
        json.dumps({"packet_seq_no": 1, "decryption_status": "parsed_sensor_batch"}) + "\n",
        encoding="utf-8",
    )

    z_dir = processed_root / "z-score" / device_id
    z_dir.mkdir(parents=True, exist_ok=True)
    (z_dir / "scaler.json").write_text("{}", encoding="utf-8")
    for split in ("train", "val", "test"):
        (z_dir / f"{split}.csv").write_text("subject,session,timestamp\n", encoding="utf-8")

    window_dir = processed_root / "window" / "0.2" / device_id
    window_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (window_dir / f"{split}.csv").write_text("subject,session,timestamp\n", encoding="utf-8")

    model_dir = models_root / device_id
    checkpoint = model_dir / "checkpoints" / "vqgan.pt"
    config = model_dir / "checkpoints" / "vqgan.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"contract-model")
    config.write_text(
        json.dumps({
            "base_channels": 16,
            "latent_dim": 32,
            "input_height": 9,
            "input_width": 20,
            "sensor_weights": [0.5, 0.5, 0.0],
            "sensor_input_masking_version": SENSOR_INPUT_MASKING_VERSION,
        }),
        encoding="utf-8",
    )
    (model_dir / "best_lock_policy.json").write_text(
        json.dumps(
            {
                device_id: {
                    "user": device_id,
                    "window": 0.2,
                    "overlap": 0.5,
                    "target_width": 20,
                    "threshold": -0.12,
                    "interrupt_rule": "k",
                    "decision_strategy": "k",
                    "k_rejects": 20,
                    "vote_window_size": 0,
                    "vote_min_rejects": 0,
                    "ema_alpha": 0.25,
                    "vqgan_checkpoint": "checkpoints/vqgan.pt",
                    "vqgan_config": "checkpoints/vqgan.json",
                    "model_version": "vqgan-contract.pt",
                    # Readiness fields so this seeded policy is accepted as a valid
                    # READY policy by check_trained_model (StartAuthentication gate).
                    "policy_status": "ready",
                    "policy_search_completed": True,
                    "threshold_strategy": "interrupt_window_frr",
                    "score_metric": "mse",
                    "score_scale": "negative_reconstruction_error",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "training_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "last_trained_bytes": 1024,
                "last_error": "",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "training_summary.json").write_text(
        json.dumps([{"window": 0.2, "val": {"auc": 0.9}, "test": {"auc": 0.8}}]),
        encoding="utf-8",
    )
    search_dir = model_dir / "policy_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "grid_results_vqgan_only.csv").write_text("threshold,auc\n-0.12,0.9\n", encoding="utf-8")

    result_dir = inference_root / device_id / "auth-session-a"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "window_id": 1,
                        "decision_score": 0.1,
                        "decision_threshold": -0.12,
                        "decision_accept": True,
                        "server_written_timestamp": "2026-01-01T00:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "window_id": 2,
                        "score": -0.5,
                        "threshold": -0.12,
                        "accept": False,
                        "interrupt": True,
                        "server_written_timestamp": "2026-01-01T00:00:01+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return device_id


def _encrypt(plaintext: bytes, key: str = "Continuous_Authentication") -> bytes:
    key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    iv = os.urandom(12)
    cipher = Cipher(algorithms.AES(key_bytes), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return iv + encryptor.tag + ciphertext


def _http_packet(device_id: str, session_id: str, packet_seq_no: int) -> bytes:
    now_ms = 1_780_000_000_000 + int(packet_seq_no)
    payload = {
        "device_id_hash": device_id,
        "session_id": session_id,
        "packet_seq_no": packet_seq_no,
        "timestamp_ms": now_ms,
        "window_start_ms": now_ms - 1000,
        "window_end_ms": now_ms,
        "type": "sensor",
        "foreground_app_name": "com.example.contract",
        "sensor_data": [
            {
                "sensor_name": "accelerometer",
                "sensor_type": 1,
                "timestamp_ns": 1234567890,
                "values": {"x": 0.1, "y": 0.2, "z": 9.8},
                "accuracy": 3,
            }
        ],
    }
    return _encrypt(gzip.compress(json.dumps(payload).encode("utf-8")))


def test_management_routes_are_absent_when_disabled(tmp_path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, management_enabled=False, api_key=None)
    client = TestClient(_contract_app())

    assert client.get("/api/v1/management/summary").status_code == 404
    assert "/api/v1/management/summary" not in client.get("/openapi.json").json()["paths"]


def test_management_api_key_errors_match_external_contract(tmp_path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, management_enabled=True, api_key=None)
    client = TestClient(_contract_app())
    assert client.get("/api/v1/management/summary").status_code == 503

    monkeypatch.setattr(settings, "management_api_key", API_KEY)
    assert client.get("/api/v1/management/summary").status_code == 401
    assert client.get("/api/v1/management/summary", headers={"X-Management-API-Key": "wrong"}).status_code == 403
    assert client.get("/api/v1/management/summary", headers={"X-Management-API-Key": API_KEY}).status_code == 200


def test_management_documented_endpoints_return_contract(contract_client) -> None:
    device_id = _seed_contract_device()
    ctx = get_runtime_context()
    accepted, message, _ = ctx.auth_manager.start_session(device_id, "active-auth-session")
    assert accepted, message
    ctx.metrics.record_client_metrics(
        {
            "device_id_hash": device_id,
            "timestamp_ms": 123,
            "reporting_period_ms": 1000,
            "batches_processed": 2,
            "uploads_success": 3,
            "uploads_failed": 1,
            "sensor_samples_collected": 400,
            "anomalies_detected": 0,
            "avg_upload_latency_ms": 12.5,
            "avg_cpu_usage_percent": 22.5,
            "peak_memory_usage_mb": 128,
            "upload_success_rate": 0.75,
        }
    )
    headers = {"X-Management-API-Key": API_KEY}

    summary = contract_client.get("/api/v1/management/summary", headers=headers).json()
    assert summary["status"] == "ok"
    assert summary["counts"]["devices"] == 1
    assert summary["counts"]["models"] == 1
    assert summary["counts"]["active_auth_sessions"] == 1

    runtime = contract_client.get("/api/v1/management/runtime", headers=headers).json()
    assert runtime["app_name"] == settings.app_name
    assert runtime["http"]["enabled"] is True
    assert runtime["grpc"]["port"] == 10500
    assert "counters" in runtime["metrics"]

    devices = contract_client.get("/api/v1/management/devices", headers=headers).json()
    assert devices["total"] == 1
    assert devices["devices"][0]["device_id_hash"] == device_id
    assert devices["devices"][0]["model"]["ready"] is True

    detail = contract_client.get(f"/api/v1/management/devices/{device_id}", headers=headers).json()
    assert detail["raw"]["sessions"] == 1
    assert detail["processed"]["windows"]["0.2"]["train"]["exists"] is True
    assert detail["model"]["ready"] is True
    assert detail["active_auth_sessions"][0]["session_id"] == "active-auth-session"

    raw_sessions = contract_client.get(
        f"/api/v1/management/devices/{device_id}/raw-sessions?limit=50",
        headers=headers,
    ).json()
    assert raw_sessions["total"] == 1
    assert raw_sessions["sessions"][0]["session_id"] == "raw-session"
    assert raw_sessions["sessions"][0]["packet_count"] == 1

    training = contract_client.get(f"/api/v1/management/devices/{device_id}/training", headers=headers).json()
    assert training["status"] == "completed"
    assert training["last_trained_bytes"] == 1024
    assert training["is_ready"] == (training["status"] == "completed" and training["has_enough_data"])

    model = contract_client.get(f"/api/v1/management/devices/{device_id}/models", headers=headers).json()
    assert model["ready"] is True
    assert model["policy"]["window_size"] == 0.2
    assert model["files"]["vqgan_checkpoint"]["exists"] is True
    assert model["files"]["vqgan_config"]["exists"] is True
    assert model["files"]["scaler"]["exists"] is True
    assert model["training_summary"][0]["val"]["auc"] == 0.9
    assert model["policy_search"]["exists"] is True

    sessions = contract_client.get(f"/api/v1/management/devices/{device_id}/auth/sessions", headers=headers).json()
    assert sessions["total"] == 1
    assert sessions["sessions"][0]["policy"]["model_version"] == "vqgan-contract.pt"

    results = contract_client.get(
        f"/api/v1/management/devices/{device_id}/auth/results?session_id=auth-session-a&limit=1",
        headers=headers,
    ).json()
    assert results["total"] == 1
    assert results["results"][0]["window_id"] == 2
    assert results["results"][0]["device_id_hash"] == device_id
    assert results["results"][0]["session_id"] == "auth-session-a"

    latest = contract_client.get("/api/v1/management/auth/results/latest?limit=10", headers=headers).json()
    assert latest["total"] == 2
    assert latest["results"][0]["window_id"] == 2

    metrics = contract_client.get("/api/v1/management/client-metrics", headers=headers).json()
    assert metrics["latest_by_device"][device_id]["upload_success_rate"] == 0.75
    assert metrics["recent"][0]["received_at"]


def test_basic_http_endpoints_and_sensor_upload_match_external_contract(contract_client) -> None:
    assert contract_client.get("/").json()["status"] == "running"
    assert contract_client.get("/health").json()["status"] == "healthy"

    schema = contract_client.get("/openapi.json").json()
    assert "/api/v1/sensor-data" in schema["paths"]
    assert "/api/v1/management/summary" in schema["paths"]
    assert "APIKeyHeader" in schema["components"]["securitySchemes"]

    device_id = "http-contract-device"
    session_id = "http-contract-session"
    response = contract_client.post(
        "/api/v1/sensor-data",
        content=_http_packet(device_id, session_id, 1),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Device-ID-Hash": device_id,
            "X-Session-ID": session_id,
            "X-Packet-Sequence": "1",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    bad_response = contract_client.post(
        "/api/v1/sensor-data",
        content=b"invalid encrypted data",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Device-ID-Hash": device_id,
            "X-Session-ID": session_id,
            "X-Packet-Sequence": "2",
        },
    )
    assert bad_response.status_code == 400
    assert bad_response.json()["reason"] == "decryption_failed"


@pytest.mark.asyncio
async def test_grpc_documented_methods_match_external_contract(tmp_path, monkeypatch) -> None:
    _configure_runtime(tmp_path, monkeypatch, management_enabled=True, api_key=API_KEY)
    device_id = _seed_contract_device("grpc-contract-device")
    service = SensorDataService(get_runtime_context())

    policy = await service.GetInitialPolicy(
        sensor_data_pb2.PolicyRequest(device_id_hash=device_id, app_version="1.0.0", android_api_level=35),
        None,
    )
    assert policy.policy_id == "default"
    assert policy.policy_version == settings.version
    assert policy.batch_interval_ms == 1000
    assert policy.max_payload_size_bytes == min(settings.max_request_size, settings.grpc_max_message_size)
    assert policy.compression_algorithm == "LZ4"
    assert policy.sensor_sampling_rates["ACCELEROMETER"] == 100

    rejected = await service.StartAuthentication(
        sensor_data_pb2.AuthSessionRequest(device_id_hash="untrained-device"),
        None,
    )
    assert rejected.accepted is False
    assert rejected.message.startswith("data_insufficient:")

    accepted = await service.StartAuthentication(
        sensor_data_pb2.AuthSessionRequest(device_id_hash=device_id, session_id="grpc-auth-session"),
        None,
    )
    assert accepted.accepted is True
    assert accepted.session_id == "grpc-auth-session"
    assert accepted.message == "ok"
    assert accepted.model_version == "vqgan-contract.pt"
    assert accepted.window_size_sec == pytest.approx(0.2)
    assert accepted.decision_time_sec > 0.0

    heartbeat = await service.SendHeartbeat(
        sensor_data_pb2.Heartbeat(client_timestamp=123, pending_packets=2, last_packet_seq_no=99),
        None,
    )
    assert heartbeat.client_timestamp_echo == 123
    assert heartbeat.server_timestamp > 0

    metric_response = await service.ReportMetrics(
        sensor_data_pb2.MetricsReport(
            device_id_hash=device_id,
            timestamp_ms=456,
            reporting_period_ms=1000,
            batches_processed=2,
            uploads_success=2,
            uploads_failed=0,
            sensor_samples_collected=300,
            anomalies_detected=1,
            avg_upload_latency_ms=8.5,
            avg_cpu_usage_percent=20.0,
            peak_memory_usage_mb=256,
            upload_success_rate=1.0,
        ),
        None,
    )
    assert metric_response.accepted is True
    assert metric_response.message == "ok"
    assert service.metrics.latest_client_metrics_by_device[device_id]["timestamp_ms"] == 456

    batch = sensor_data_pb2.SerializedSensorBatch(
        session_id="grpc-stream-session",
        samples=[
            sensor_data_pb2.SensorSample(
                type=sensor_data_pb2.ACCELEROMETER,
                event_timestamp_ns=1,
                x=1.0,
                y=2.0,
                z=3.0,
                accuracy=3,
                seq_no=1,
                foreground_app_name="com.example.contract",
            )
        ],
    )
    compressed = lz4.frame.compress(batch.SerializeToString())
    packet = sensor_data_pb2.DataPacket(
        packet_id="packet-1",
        device_id_hash=device_id,
        base_wall_ms=123,
        device_uptime_ns=456,
        packet_seq_no=1,
        encrypted_sensor_payload=_encrypt(compressed),
        metadata=sensor_data_pb2.Metadata(compression="lz4", uncompressed_size_bytes=len(batch.SerializeToString())),
        sha256=hashlib.sha256(compressed).digest(),
    )
    pending_inference: set = set()
    directive = await service._handle_packet(packet, response_queue=None, pending_inference=pending_inference)
    if pending_inference:
        await asyncio.gather(*pending_inference)

    assert directive.ack.success is True
    assert directive.ack.packet_id == "packet-1"
    assert directive.ack.error_code == ""

    missing_payload = sensor_data_pb2.DataPacket(
        packet_id="packet-2",
        device_id_hash=device_id,
        packet_seq_no=2,
    )
    failed = await service._handle_packet(missing_payload, response_queue=None, pending_inference=set())
    assert failed.ack.success is False
    assert failed.ack.error_code == "INVALID_FORMAT"
