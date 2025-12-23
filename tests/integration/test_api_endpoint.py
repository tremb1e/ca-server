import pytest
import json
import gzip
import hashlib
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import lz4.frame
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def client(temp_data_dir, monkeypatch):
    monkeypatch.setenv("DATA_STORAGE_PATH", str(temp_data_dir))
    from src.main import app
    return TestClient(app)


class TestSensorDataEndpoint:
    def encrypt_data(self, plaintext: bytes, key: str) -> bytes:
        key_bytes = hashlib.sha256(key.encode('utf-8')).digest()
        iv = os.urandom(12)

        cipher = Cipher(
            algorithms.AES(key_bytes),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return iv + encryptor.tag + ciphertext

    def prepare_test_packet(
        self,
        device_id_hash: int,
        session_id: int,
        packet_seq_no: int,
        compression: str = "gzip",
    ):
        packet_data = {
            "device_id_hash": device_id_hash,
            "session_id": session_id,
            "packet_seq_no": packet_seq_no,
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "window_start_ms": int(datetime.now().timestamp() * 1000) - 5000,
            "window_end_ms": int(datetime.now().timestamp() * 1000),
            "type": "sensor",
            "foreground_package_name": "com.test.app",
            "sensor_data": [
                {
                    "sensor_name": "accelerometer",
                    "sensor_type": 1,
                    "timestamp_ns": 1234567890123456789,
                    "values": {"x": 0.1, "y": 0.2, "z": 9.8},
                    "accuracy": 3
                }
            ]
        }

        json_str = json.dumps(packet_data)
        if compression == "gzip":
            compressed = gzip.compress(json_str.encode('utf-8'))
        elif compression == "lz4":
            compressed = lz4.frame.compress(json_str.encode('utf-8'))
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
        encrypted = self.encrypt_data(compressed, "Continuous_Authentication")

        return encrypted, packet_data

    def test_successful_packet_submission_gzip(self, client):
        device_id_hash = 123456789
        session_id = 987654321
        packet_seq_no = 1

        encrypted_data, original_packet = self.prepare_test_packet(
            device_id_hash, session_id, packet_seq_no, compression="gzip"
        )

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted_data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Device-ID-Hash": str(device_id_hash),
                "X-Session-ID": str(session_id),
                "X-Packet-Sequence": str(packet_seq_no)
            }
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "ok"

    def test_multiple_packets_submission(self, client):
        device_id_hash = 123456789
        session_id = 987654321

        for packet_seq_no in range(1, 4):
            encrypted_data, _ = self.prepare_test_packet(
                device_id_hash, session_id, packet_seq_no, compression="gzip"
            )

            response = client.post(
                "/api/v1/sensor-data",
                content=encrypted_data,
                headers={
                    "X-Device-ID-Hash": str(device_id_hash),
                    "X-Session-ID": str(session_id),
                    "X-Packet-Sequence": str(packet_seq_no)
                }
            )

            assert response.status_code == 200

    def test_device_id_mismatch(self, client):
        device_id_hash = 123456789
        session_id = 987654321
        packet_seq_no = 1

        encrypted_data, _ = self.prepare_test_packet(
            device_id_hash, session_id, packet_seq_no, compression="gzip"
        )

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted_data,
            headers={
                "X-Device-ID-Hash": "999999999",
                "X-Session-ID": str(session_id),
                "X-Packet-Sequence": str(packet_seq_no)
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "validation_failed"

    def test_session_id_mismatch(self, client):
        device_id_hash = 123456789
        session_id = 987654321
        packet_seq_no = 1

        encrypted_data, _ = self.prepare_test_packet(
            device_id_hash, session_id, packet_seq_no, compression="gzip"
        )

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted_data,
            headers={
                "X-Device-ID-Hash": str(device_id_hash),
                "X-Session-ID": "999999999",
                "X-Packet-Sequence": str(packet_seq_no)
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "validation_failed"

    def test_sequence_number_mismatch(self, client):
        device_id_hash = 123456789
        session_id = 987654321
        packet_seq_no = 1

        encrypted_data, _ = self.prepare_test_packet(
            device_id_hash, session_id, packet_seq_no, compression="gzip"
        )

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted_data,
            headers={
                "X-Device-ID-Hash": str(device_id_hash),
                "X-Session-ID": str(session_id),
                "X-Packet-Sequence": "999"
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "validation_failed"

    def test_empty_request_body(self, client):
        response = client.post(
            "/api/v1/sensor-data",
            content=b"",
            headers={
                "X-Device-ID-Hash": "123",
                "X-Session-ID": "456",
                "X-Packet-Sequence": "1"
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "no_data"

    def test_invalid_encrypted_data(self, client):
        response = client.post(
            "/api/v1/sensor-data",
            content=b"invalid encrypted data",
            headers={
                "X-Device-ID-Hash": "123",
                "X-Session-ID": "456",
                "X-Packet-Sequence": "1"
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "decryption_failed"

    def test_invalid_compressed_data(self, client):
        invalid_compressed = b"not compressed"
        encrypted = self.encrypt_data(invalid_compressed, "Continuous_Authentication")

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted,
            headers={
                "X-Device-ID-Hash": "123",
                "X-Session-ID": "456",
                "X-Packet-Sequence": "1"
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "decompression_failed"

    def test_invalid_json_data(self, client):
        invalid_json = b"not json"
        compressed = lz4.frame.compress(invalid_json)
        encrypted = self.encrypt_data(compressed, "Continuous_Authentication")

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted,
            headers={
                "X-Device-ID-Hash": "123",
                "X-Session-ID": "456",
                "X-Packet-Sequence": "1"
            }
        )

        assert response.status_code == 400
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["reason"] == "invalid_json"

    def test_health_endpoint(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert "storage_stats" in response_data
        assert response_data["storage_stats"]["total_devices"] >= 0

    def test_root_endpoint(self, client):
        response = client.get("/")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["app_name"] == "Continuous Authentication Server"
        assert response_data["status"] == "running"
    def test_successful_packet_submission_lz4(self, client):
        device_id_hash = 222333444
        session_id = 111222333
        packet_seq_no = 5

        encrypted_data, _ = self.prepare_test_packet(
            device_id_hash, session_id, packet_seq_no, compression="lz4"
        )

        response = client.post(
            "/api/v1/sensor-data",
            content=encrypted_data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Device-ID-Hash": str(device_id_hash),
                "X-Session-ID": str(session_id),
                "X-Packet-Sequence": str(packet_seq_no),
            },
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "ok"
