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
    async def submit_if_ready(self, device_id_hash, *, force=False):
        return None


class _AuthManager:
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

    assert directive.ack.success is True
    assert service.storage.records[0]["packet_data"]["decryption_status"] == "decompress_failed"
    assert "encrypted_sensor_payload_b64" in service.storage.records[0]["packet_data"]


@pytest.mark.asyncio
async def test_stream_packet_rejects_batch_user_mismatch(tmp_path):
    service = _service(tmp_path)
    batch = sensor_data_pb2.SerializedSensorBatch(
        user_id_hash="other-device",
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
    assert stored["decryption_status"] == "validation_failed"
    assert "sensor_batch" not in stored
