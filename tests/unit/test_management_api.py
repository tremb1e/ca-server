import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import settings
from src.management.router import router
from src.management.runtime import get_runtime_context, reset_runtime_context


def _client(tmp_path, monkeypatch) -> TestClient:
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
    monkeypatch.setattr(settings, "management_api_enabled", True)
    monkeypatch.setattr(settings, "management_api_key", "secret-token")
    reset_runtime_context()

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _seed_device(tmp_path, monkeypatch, device_id: str = "device123") -> str:
    raw_root = settings.data_storage_path
    models_root = raw_root.parent / "models"
    processed_root = settings.processed_data_path
    inference_root = settings.inference_storage_path

    raw_device = raw_root / device_id
    raw_device.mkdir(parents=True, exist_ok=True)
    (raw_device / "session_session-a.jsonl").write_text(
        json.dumps({"packet_seq_no": 1, "decryption_status": "parsed_sensor_batch"}) + "\n",
        encoding="utf-8",
    )

    user_dir = models_root / device_id
    ckpt = user_dir / "checkpoints" / "vqgan.pt"
    cfg = user_dir / "checkpoints" / "vqgan.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"model-bytes")
    cfg.write_text(
        json.dumps({"base_channels": 96, "latent_dim": 256, "input_height": 12, "input_width": 20}),
        encoding="utf-8",
    )
    (user_dir / "best_lock_policy.json").write_text(
        json.dumps(
            {
                device_id: {
                    "user": device_id,
                    "window": 0.2,
                    "overlap": 0.5,
                    "target_width": 20,
                    "threshold": -0.12,
                    "interrupt_rule": "k",
                    "k_rejects": 20,
                    "vqgan_checkpoint": "checkpoints/vqgan.pt",
                    "vqgan_config": "checkpoints/vqgan.json",
                    "model_version": "vqgan.pt",
                }
            }
        ),
        encoding="utf-8",
    )
    (user_dir / "training_state.json").write_text(
        json.dumps({"status": "completed", "last_trained_bytes": 1234, "last_error": "", "updated_at": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    (user_dir / "training_summary.json").write_text(
        json.dumps([{"window": 0.2, "val": {"auc": 0.9}, "test": {"auc": 0.8}}]),
        encoding="utf-8",
    )

    z_dir = processed_root / "z-score" / device_id
    z_dir.mkdir(parents=True, exist_ok=True)
    (z_dir / "scaler.json").write_text("{}", encoding="utf-8")
    for split in ("train", "val", "test"):
        (z_dir / f"{split}.csv").write_text("subject,session,timestamp\n", encoding="utf-8")

    result_dir = inference_root / device_id / "auth-session-a"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"window_id": 1, "accept": True, "server_written_timestamp": "2026-01-01T00:00:00+00:00"}),
                json.dumps({"window_id": 2, "accept": False, "interrupt": True, "server_written_timestamp": "2026-01-01T00:00:01+00:00"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return device_id


def test_management_security(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    assert client.get("/api/v1/management/summary").status_code == 401
    assert client.get("/api/v1/management/summary", headers={"X-Management-API-Key": "wrong"}).status_code == 403
    assert client.get("/api/v1/management/summary", headers={"X-Management-API-Key": "secret-token"}).status_code == 200


def test_management_summary_and_device_details(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    device_id = _seed_device(tmp_path, monkeypatch)

    headers = {"X-Management-API-Key": "secret-token"}
    summary = client.get("/api/v1/management/summary", headers=headers).json()
    assert summary["counts"]["devices"] == 1
    assert summary["counts"]["models"] == 1

    device = client.get(f"/api/v1/management/devices/{device_id}", headers=headers).json()
    assert device["raw"]["sessions"] == 1
    assert device["training"]["status"] == "completed"
    assert device["model"]["ready"] is True


def test_management_models_and_auth_results(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    device_id = _seed_device(tmp_path, monkeypatch)
    headers = {"X-Management-API-Key": "secret-token"}

    model = client.get(f"/api/v1/management/devices/{device_id}/models", headers=headers).json()
    assert model["ready"] is True
    assert model["files"]["vqgan_checkpoint"]["size_bytes"] == len(b"model-bytes")
    assert model["training_summary"][0]["val"]["auc"] == 0.9

    results = client.get(f"/api/v1/management/devices/{device_id}/auth/results?limit=1", headers=headers).json()
    assert results["total"] == 1
    assert results["results"][0]["window_id"] == 2


def test_management_openapi_has_api_key_scheme(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    schema = client.get("/openapi.json").json()

    assert "APIKeyHeader" in schema["components"]["securitySchemes"]
    assert "/api/v1/management/summary" in schema["paths"]


def test_management_rejects_unsafe_device_id(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get(
        "/api/v1/management/devices/bad%5Cdevice",
        headers={"X-Management-API-Key": "secret-token"},
    )

    assert response.status_code == 400

    response = client.get(
        "/api/v1/management/devices/device123/auth/results?session_id=..%2Foutside",
        headers={"X-Management-API-Key": "secret-token"},
    )
    assert response.status_code == 400


def test_runtime_context_uses_configured_models_root(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    ctx = get_runtime_context()

    assert ctx.models_root == settings.data_storage_path.parent / "models"
    assert ctx.auth_manager._models_root == ctx.models_root
