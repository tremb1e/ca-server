import pytest
import asyncio
import json
from pathlib import Path
import tempfile
import shutil
from src.storage.file_storage import FileStorage


class TestFileStorage:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir):
        return FileStorage(temp_dir)

    @pytest.fixture
    def sample_packet(self):
        return {
            "device_id_hash": 123456789,
            "session_id": 987654321,
            "packet_seq_no": 1,
            "timestamp_ms": 1234567890000,
            "window_start_ms": 1234567885000,
            "window_end_ms": 1234567890000,
            "type": "sensor",
            "sensor_data": []
        }

    @pytest.mark.asyncio
    async def test_append_packet_success(self, storage, sample_packet):
        device_id_hash = "device123"
        session_id = "session456"

        success, error = await storage.append_packet(
            device_id_hash, session_id, sample_packet
        )

        assert success is True
        assert error is None

        file_path = storage._get_session_file_path(device_id_hash, session_id)
        assert file_path.exists()

        with open(file_path, 'r') as f:
            line = f.readline()
            stored_data = json.loads(line)
            assert stored_data["device_id_hash"] == sample_packet["device_id_hash"]
            assert "server_received_timestamp" in stored_data

    @pytest.mark.asyncio
    async def test_append_multiple_packets(self, storage, sample_packet):
        device_id_hash = "device123"
        session_id = "session456"

        for i in range(3):
            packet = sample_packet.copy()
            packet["packet_seq_no"] = i
            success, error = await storage.append_packet(
                device_id_hash, session_id, packet
            )
            assert success is True

        file_path = storage._get_session_file_path(device_id_hash, session_id)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_read_session_success(self, storage, sample_packet):
        device_id_hash = "device123"
        session_id = "session456"

        for i in range(3):
            packet = sample_packet.copy()
            packet["packet_seq_no"] = i
            await storage.append_packet(device_id_hash, session_id, packet)

        success, packets, error = await storage.read_session(device_id_hash, session_id)

        assert success is True
        assert error is None
        assert len(packets) == 3
        assert all(p["packet_seq_no"] == i for i, p in enumerate(packets))

    @pytest.mark.asyncio
    async def test_read_nonexistent_session(self, storage):
        device_id_hash = "nonexistent"
        session_id = "nonexistent"

        success, packets, error = await storage.read_session(device_id_hash, session_id)

        assert success is True
        assert error is None
        assert packets == []

    @pytest.mark.asyncio
    async def test_read_session_sorted(self, storage, sample_packet):
        device_id_hash = "device123"
        session_id = "session456"

        for i in [2, 0, 1]:
            packet = sample_packet.copy()
            packet["packet_seq_no"] = i
            await storage.append_packet(device_id_hash, session_id, packet)

        success, packets, error = await storage.read_session(device_id_hash, session_id)

        assert success is True
        assert packets[0]["packet_seq_no"] == 0
        assert packets[1]["packet_seq_no"] == 1
        assert packets[2]["packet_seq_no"] == 2

    def test_get_storage_stats_empty(self, storage):
        stats = storage.get_storage_stats()

        assert stats["total_devices"] == 0
        assert stats["total_sessions"] == 0
        assert stats["total_size_mb"] == 0

    @pytest.mark.asyncio
    async def test_get_storage_stats_with_data(self, storage, sample_packet):
        for device in ["device1", "device2"]:
            for session in ["session1", "session2"]:
                await storage.append_packet(device, session, sample_packet)

        stats = storage.get_storage_stats()

        assert stats["total_devices"] == 2
        assert stats["total_sessions"] == 4
        assert stats["total_size_mb"] >= 0

    def test_get_session_file_path(self, storage):
        device_id_hash = "device123"
        session_id = "session456"

        path = storage._get_session_file_path(device_id_hash, session_id)

        assert "device123" in str(path)
        assert "session_session456.jsonl" in str(path)