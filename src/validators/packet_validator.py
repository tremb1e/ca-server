from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, field_validator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SensorValue(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class SensorData(BaseModel):
    sensor_name: str
    sensor_type: int
    timestamp_ns: int
    values: SensorValue
    accuracy: int


class DataPacket(BaseModel):
    device_id_hash: int
    session_id: int
    packet_seq_no: int
    timestamp_ms: int
    window_start_ms: int
    window_end_ms: int
    type: str
    foreground_package_name: Optional[str] = None
    sensor_data: list[Dict[str, Any]]

    @field_validator("timestamp_ms", "window_start_ms", "window_end_ms")
    @classmethod
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError("Timestamp must be positive")

        min_ts = datetime(2020, 1, 1).timestamp() * 1000
        max_ts = datetime(2030, 1, 1).timestamp() * 1000

        if v < min_ts or v > max_ts:
            raise ValueError(f"Timestamp {v} outside valid range (2020-2030)")
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in ['sensor', 'event', 'mixed']:
            raise ValueError(f"Invalid packet type: {v}")
        return v


class PacketValidator:
    def __init__(self):
        logger.info("PacketValidator initialized")

    def validate(
        self,
        json_data: Dict[str, Any],
        device_id_hash: str,
        session_id: str,
        packet_sequence: int
    ) -> Tuple[bool, Optional[DataPacket], Optional[str]]:
        try:
            packet = DataPacket(**json_data)

            if str(packet.device_id_hash) != device_id_hash:
                error_msg = f"Device ID hash mismatch: packet={packet.device_id_hash}, header={device_id_hash}"
                logger.error(error_msg)
                return False, None, error_msg

            if str(packet.session_id) != session_id:
                error_msg = f"Session ID mismatch: packet={packet.session_id}, header={session_id}"
                logger.error(error_msg)
                return False, None, error_msg

            if packet.packet_seq_no != packet_sequence:
                error_msg = f"Packet sequence mismatch: packet={packet.packet_seq_no}, header={packet_sequence}"
                logger.error(error_msg)
                return False, None, error_msg

            if packet.window_end_ms < packet.window_start_ms:
                error_msg = f"Invalid window: start={packet.window_start_ms}, end={packet.window_end_ms}"
                logger.error(error_msg)
                return False, None, error_msg

            logger.info(f"Successfully validated packet: device={device_id_hash}, session={session_id}, seq={packet_sequence}")
            return True, packet, None

        except Exception as e:
            error_msg = f"Validation failed: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
