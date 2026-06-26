import os
import hashlib
from types import SimpleNamespace

import lz4.frame
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.config import settings
from src.grpc_server import SensorDataService
from src.management.runtime import RuntimeMetrics
from src.protos import sensor_data_pb2


class _Storage:
    def __init__(self):
        self.records = []

    async def append_packet(self, device_id_hash, session_id, packet_data):
        self.records.append(
            {
                "device_id_hash": device_id_hash,
                "session_id": session_id,
                "packet_data": dict(packet_data),
            }
        )
        return True, None


class _TrainingManager:
    def __init__(self):
        self.submitted = []

    def get_readiness(self, device_id_hash):
        return SimpleNamespace(
            status="pending",
            total_bytes=0,
            min_bytes=100,
            has_enough_data=False,
        )

    async def submit_if_ready(self, device_id_hash, *, force=False):
        self.submitted.append((device_id_hash, force))
        return None


class _AuthManager:
    def check_trained_model(self, user_id):
        return False, "missing policy"

    async def handle_packet(self, *, user_id, session_id, parsed_batch):
        return None


def _service(tmp_path) -> SensorDataService:
    ctx = SimpleNamespace(
        storage=_Storage(),
        training_manager=_TrainingManager(),
        auth_manager=_AuthManager(),
        metrics=RuntimeMetrics(),
        models_root=tmp_path / "models",
    )
    return SensorDataService(ctx)


def _encrypt(plaintext: bytes, key: str = "Continuous_Authentication") -> bytes:
    key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    iv = os.urandom(12)
    cipher = Cipher(algorithms.AES(key_bytes), modes.GCM(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return iv + encryptor.tag + ciphertext


@pytest.mark.asyncio
async def test_initial_policy_respects_grpc_message_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_request_size", 10 * 1024 * 1024)
    monkeypatch.setattr(settings, "grpc_max_message_size", 4 * 1024 * 1024)
    service = _service(tmp_path)

    policy = await service.GetInitialPolicy(sensor_data_pb2.PolicyRequest(device_id_hash="device-a"), None)

    assert policy.max_payload_size_bytes == 4 * 1024 * 1024


@pytest.mark.asyncio
async def test_stream_packet_rejects_declared_decompressed_size_above_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_decompressed_size", 100)
    service = _service(tmp_path)
    packet = sensor_data_pb2.DataPacket(
        packet_id="packet-1",
        device_id_hash="device-a",
        packet_seq_no=1,
        encrypted_sensor_payload=b"not-used",
        metadata=sensor_data_pb2.Metadata(compression="lz4", uncompressed_size_bytes=101),
    )

    directive = await service._handle_packet(packet, response_queue=None, pending_inference=set())

    assert directive.ack.success is False
    assert directive.ack.error_code == "INVALID_FORMAT"
    assert service.storage.records[0]["packet_data"]["decryption_status"] == "decompress_failed"
    assert "encrypted_sensor_payload_b64" in service.storage.records[0]["packet_data"]


@pytest.mark.asyncio
async def test_stream_packet_uses_device_hash_as_batch_id(tmp_path):
    service = _service(tmp_path)
    batch = sensor_data_pb2.SerializedSensorBatch(
        session_id="auth-session",
        samples=[
            sensor_data_pb2.SensorSample(
                type=sensor_data_pb2.ACCELEROMETER,
                event_timestamp_ns=1,
                x=1.0,
                y=2.0,
                z=3.0,
                accuracy=3,
                seq_no=1,
                foreground_app_name="com.example.current",
            )
        ],
    )
    compressed = lz4.frame.compress(batch.SerializeToString())
    packet = sensor_data_pb2.DataPacket(
        packet_id="packet-2",
        device_id_hash="device-a",
        packet_seq_no=2,
        encrypted_sensor_payload=_encrypt(compressed),
        metadata=sensor_data_pb2.Metadata(
            compression="lz4",
            uncompressed_size_bytes=len(batch.SerializeToString()),
        ),
    )

    directive = await service._handle_packet(packet, response_queue=None, pending_inference=set())

    assert directive.ack.success is True
    stored = service.storage.records[0]["packet_data"]
    assert stored["decryption_status"] == "parsed_sensor_batch"
    assert set(stored["sensor_batch"]) == {"samples", "session_id"}
    sample = stored["sensor_batch"]["samples"][0]
    assert sample["foreground_app_name"] == "com.example.current"
    assert "foreground_app_hash" not in sample
    assert service.storage.records[0]["device_id_hash"] == "device-a"
    assert service.storage.records[0]["session_id"] == "auth-session"


@pytest.mark.asyncio
async def test_stream_packet_rejects_unsafe_batch_session_id_without_server_error(tmp_path):
    service = _service(tmp_path)
    batch = sensor_data_pb2.SerializedSensorBatch(
        session_id="../outside",
        samples=[
            sensor_data_pb2.SensorSample(
                type=sensor_data_pb2.ACCELEROMETER,
                event_timestamp_ns=1,
                x=1.0,
                y=2.0,
                z=3.0,
                accuracy=3,
                seq_no=1,
            )
        ],
    )
    compressed = lz4.frame.compress(batch.SerializeToString())
    packet = sensor_data_pb2.DataPacket(
        packet_id="packet-bad-session",
        device_id_hash="device-a",
        packet_seq_no=3,
        encrypted_sensor_payload=_encrypt(compressed),
        metadata=sensor_data_pb2.Metadata(
            compression="lz4",
            uncompressed_size_bytes=len(batch.SerializeToString()),
        ),
    )

    directive = await service._handle_packet(packet, response_queue=None, pending_inference=set())

    assert directive.ack.success is False
    assert directive.ack.error_code == "INVALID_IDENTIFIER"
    assert service.storage.records[0]["session_id"] == "default"
    assert service.storage.records[0]["packet_data"]["decryption_status"] == "invalid_identifier"
    assert service.metrics.counters["packet_status_invalid_identifier"] == 1


@pytest.mark.asyncio
async def test_report_metrics_keeps_numeric_management_fields(tmp_path):
    service = _service(tmp_path)

    response = await service.ReportMetrics(
        sensor_data_pb2.MetricsReport(
            device_id_hash="device-a",
            timestamp_ms=123,
            reporting_period_ms=1000,
            batches_processed=2,
            uploads_success=3,
            uploads_failed=1,
            sensor_samples_collected=400,
            anomalies_detected=5,
            avg_upload_latency_ms=12.5,
            avg_cpu_usage_percent=23.5,
            peak_memory_usage_mb=128,
            upload_success_rate=0.75,
        ),
        None,
    )

    assert response.accepted is True
    record = service.metrics.latest_client_metrics_by_device["device-a"]
    assert record["timestamp_ms"] == 123
    assert isinstance(record["timestamp_ms"], int)
    assert record["avg_upload_latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_start_authentication_reports_model_not_ready_after_completed_training(tmp_path):
    class _CompletedTrainingManager(_TrainingManager):
        def get_readiness(self, device_id_hash):
            return SimpleNamespace(
                status="completed",
                total_bytes=200,
                min_bytes=100,
                has_enough_data=True,
            )

    class _NotReadyAuthManager(_AuthManager):
        def check_trained_model(self, user_id):
            return False, "missing checkpoint: /models/device-a/checkpoint.pt"

    ctx = SimpleNamespace(
        storage=_Storage(),
        training_manager=_CompletedTrainingManager(),
        auth_manager=_NotReadyAuthManager(),
        metrics=RuntimeMetrics(),
        models_root=tmp_path / "models",
    )
    service = SensorDataService(ctx)

    response = await service.StartAuthentication(
        sensor_data_pb2.AuthSessionRequest(device_id_hash="device-a"),
        None,
    )

    assert response.accepted is False
    assert response.message.startswith("model_not_ready: missing checkpoint")
    assert service.training_manager.submitted == []
