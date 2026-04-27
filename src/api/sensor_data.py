from fastapi import APIRouter, Request, Header
from fastapi.responses import JSONResponse
import json
import logging
from typing import Optional
from ..crypto.decryption import AESDecryptor
from ..crypto.decompression import DataDecompressor
from ..crypto.payload_codec import decrypt_then_decompress
from ..management.runtime import get_runtime_context
from ..validators.packet_validator import PacketValidator
from ..config import settings
from ..utils.path_safety import UnsafePathSegmentError, validate_storage_id

logger = logging.getLogger(__name__)

router = APIRouter()

decryptor = AESDecryptor(settings.encryption_key)
decompressor = DataDecompressor(max_output_size=settings.max_decompressed_size)
validator = PacketValidator()


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
    runtime_context = get_runtime_context()
    storage = runtime_context.storage
    decryption_status = "skipped"
    storage_ok = False
    error_detail: Optional[str] = None

    try:
        try:
            device_id_hash = validate_storage_id(device_id_hash, field_name="device_id_hash")
            session_id = validate_storage_id(session_id, field_name="session_id")
        except UnsafePathSegmentError as exc:
            decryption_status = "invalid_identifier"
            error_detail = str(exc)
            logger.error("%s: Unsafe request identifier - %s", request_id, exc)
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "invalid_identifier"}
            )

        logger.info(f"Received request: {request_id}, Content-Length: {content_length}")

        if content_length and content_length > settings.max_request_size:
            error_msg = f"Request too large: {content_length} bytes"
            decryption_status = "request_too_large"
            error_detail = error_msg
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=413,
                content={"status": "error", "reason": "request_too_large"}
            )

        encrypted_data = await request.body()
        if len(encrypted_data) > settings.max_request_size:
            error_msg = f"Request too large: {len(encrypted_data)} bytes"
            decryption_status = "request_too_large"
            error_detail = error_msg
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=413,
                content={"status": "error", "reason": "request_too_large"}
            )
        if not encrypted_data:
            error_msg = "No data received"
            decryption_status = "no_data"
            error_detail = error_msg
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "no_data"}
            )

        logger.info(f"{request_id}: Received {len(encrypted_data)} bytes of encrypted data")

        success, decompressed_data, reason, error = decrypt_then_decompress(
            encrypted_payload=encrypted_data,
            decryptor=decryptor,
            decompressor=decompressor,
        )
        if not success:
            decryption_status = "decompress_failed" if reason == "decompression_failed" else "decrypt_failed"
            error_detail = error
            logger.error(f"{request_id}: Payload processing failed ({reason}) - {error}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": reason}
            )

        logger.info(f"{request_id}: Successfully decompressed to {len(decompressed_data)} bytes")
        decryption_status = "decompressed"

        try:
            json_data = json.loads(decompressed_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_msg = f"Invalid JSON data: {str(e)}"
            decryption_status = "parse_failed"
            error_detail = error_msg
            logger.error(f"{request_id}: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "invalid_json"}
            )

        success, validated_packet, error = validator.validate(
            json_data, device_id_hash, session_id, packet_sequence
        )
        if not success:
            decryption_status = "validation_failed"
            error_detail = error
            logger.error(f"{request_id}: Validation failed - {error}")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "reason": "validation_failed"}
            )

        logger.info(f"{request_id}: Data validation successful")
        decryption_status = "parsed_sensor_batch"

        success, error = await storage.append_packet(
            device_id_hash=device_id_hash,
            session_id=session_id,
            packet_data=json_data
        )
        storage_ok = bool(success)
        if not success:
            decryption_status = "storage_failed"
            error_detail = error
            logger.error(f"{request_id}: Storage failed - {error}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "reason": "storage_failed"}
            )

        logger.info(f"{request_id}: Successfully processed and stored packet")
        await runtime_context.training_manager.submit_if_ready(device_id_hash)

        return JSONResponse(
            status_code=200,
            content={"status": "ok"}
        )

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        error_detail = error_msg
        logger.error(f"{request_id}: {error_msg}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "reason": "internal_error"}
        )
    finally:
        runtime_context.metrics.record_packet(
            device_id=device_id_hash,
            session_id=str(session_id),
            packet_id=request_id,
            packet_seq_no=int(packet_sequence),
            status=str(decryption_status),
            storage_ok=bool(storage_ok),
            error_detail=error_detail,
        )
