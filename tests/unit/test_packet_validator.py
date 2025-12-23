import pytest
from datetime import datetime
from src.validators.packet_validator import PacketValidator, DataPacket
from pydantic import ValidationError


class TestPacketValidator:
    @pytest.fixture
    def validator(self):
        return PacketValidator()

    @pytest.fixture
    def valid_packet_data(self):
        return {
            "device_id_hash": 123456789,
            "session_id": 987654321,
            "packet_seq_no": 1,
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "window_start_ms": int(datetime.now().timestamp() * 1000) - 5000,
            "window_end_ms": int(datetime.now().timestamp() * 1000),
            "type": "sensor",
            "foreground_package_name": "com.example.app",
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

    def test_validate_success(self, validator, valid_packet_data):
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is True
        assert error is None
        assert isinstance(packet, DataPacket)
        assert packet.device_id_hash == 123456789
        assert packet.session_id == 987654321
        assert packet.packet_seq_no == 1

    def test_validate_device_id_mismatch(self, validator, valid_packet_data):
        device_id_hash = "999999999"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "device id hash mismatch" in error.lower()

    def test_validate_session_id_mismatch(self, validator, valid_packet_data):
        device_id_hash = "123456789"
        session_id = "999999999"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "session id mismatch" in error.lower()

    def test_validate_sequence_mismatch(self, validator, valid_packet_data):
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 999

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "packet sequence mismatch" in error.lower()

    def test_validate_invalid_window(self, validator, valid_packet_data):
        valid_packet_data["window_start_ms"] = valid_packet_data["window_end_ms"] + 1000
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "invalid window" in error.lower()

    def test_validate_invalid_type(self, validator, valid_packet_data):
        valid_packet_data["type"] = "invalid_type"
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "validation failed" in error.lower()

    def test_validate_missing_field(self, validator):
        incomplete_data = {
            "device_id_hash": 123456789,
            "session_id": 987654321,
        }
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            incomplete_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "validation failed" in error.lower()

    def test_validate_negative_timestamp(self, validator, valid_packet_data):
        valid_packet_data["timestamp_ms"] = -1000
        device_id_hash = "123456789"
        session_id = "987654321"
        packet_sequence = 1

        success, packet, error = validator.validate(
            valid_packet_data, device_id_hash, session_id, packet_sequence
        )

        assert success is False
        assert packet is None
        assert "validation failed" in error.lower()