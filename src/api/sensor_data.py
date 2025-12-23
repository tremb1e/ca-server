from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import json
import logging
from typing import Optional
from ..crypto.decryption import AESDecryptor
from ..crypto.decompression import DataDecompressor
from ..validators.packet_validator import PacketValidator
from ..storage.file_storage import FileStorage
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

decryptor = AESDecryptor(settings.encryption_key)
decompressor = DataDecompressor()
validator = PacketValidator()
storage = FileStorage(settings.data_storage_path)


@router.post("/api/v1/sensor-data")
async def receive_sensor_data(
    request: Request,
    x_device_id_hash: str = Header(..., alias="X-Device-ID-Hash"),
    x_session_id: str = Header(..., alias="X-Session-ID"),
    x_packet_sequence: int = Header(..., alias="X-Packet-Sequence"),
    content_length: Optional[int] = Header(None)
):
    device_id_hash = x_device_id_hash
    session_id = x_session_id
    packet_sequence = x_packet_sequence
    request_id = f"{device_id_hash}_{session_id}_{packet_sequence}"

    try:
        logger.info(f"Received request: {request_id}, Content-Length: {content_length}")

        if content_length and content_length > settings.max_request_size:
            error_msg = f"Request too large: {content_length} bytes"
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=413,
                content={"status": "error", "reason": "request_too_large"}
            )

        encrypted_data = await request.body()
        if not encrypted_data:
            error_msg = "No data received"
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "no_data"}
            )

        logger.info(f"{request_id}: Received {len(encrypted_data)} bytes of encrypted data")

        success, decrypted_data, error = decryptor.decrypt(encrypted_data)
        if not success:
            logger.error(f"{request_id}: Decryption failed - {error}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "decryption_failed"}
            )

        logger.info(f"{request_id}: Successfully decrypted to {len(decrypted_data)} bytes")

        success, decompressed_data, error = decompressor.decompress(decrypted_data)
        if not success:
            logger.error(f"{request_id}: Decompression failed - {error}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "decompression_failed"}
            )

        logger.info(f"{request_id}: Successfully decompressed to {len(decompressed_data)} bytes")

        try:
            json_data = json.loads(decompressed_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_msg = f"Invalid JSON data: {str(e)}"
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "invalid_json"}
            )

        success, validated_packet, error = validator.validate(
            json_data, device_id_hash, session_id, packet_sequence
        )
        if not success:
            logger.error(f"{request_id}: Validation failed - {error}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "validation_failed"}
            )

        logger.info(f"{request_id}: Data validation successful")

        success, error = await storage.append_packet(
            device_id_hash=device_id_hash,
            session_id=session_id,
            packet_data=json_data
        )
        if not success:
            logger.error(f"{request_id}: Storage failed - {error}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "reason": "storage_failed"}
            )

        logger.info(f"{request_id}: Successfully processed and stored packet")

        return JSONResponse(
            status_code=200,
            content={"status": "ok"}
        )

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{request_id}: {error_msg}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "reason": "internal_error"}
        )
