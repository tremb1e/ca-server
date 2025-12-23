import asyncio
import base64
import inspect
import logging
import os
import time
import uuid
from typing import Optional, Tuple

import grpc
from google.protobuf import json_format
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .ca_config import get_ca_config
from .config import settings
from .crypto.decryption import AESDecryptor
from .crypto.decompression import DataDecompressor
from .authentication.manager import AuthSessionManager
from .storage.file_storage import FileStorage
from .training.manager import TrainingManager
from .protos import sensor_data_pb2, sensor_data_pb2_grpc
from .utils.tls import TLS12_13_CIPHERS, TLSProbeResult, probe_tls_configuration

logger = logging.getLogger(__name__)

_DEFAULT_FIELDS_KWARG: str | None = None
try:
    _sig = inspect.signature(json_format.MessageToDict)
    if "including_default_value_fields" in _sig.parameters:
        _DEFAULT_FIELDS_KWARG = "including_default_value_fields"
    elif "always_print_fields_with_no_presence" in _sig.parameters:
        # protobuf>=5 renamed this flag (protobuf 6 drops the old name)
        _DEFAULT_FIELDS_KWARG = "always_print_fields_with_no_presence"
except Exception:  # noqa: BLE001
    _DEFAULT_FIELDS_KWARG = None


def _b64(data: bytes | None) -> Optional[str]:
    if not data:
        return None
    return base64.b64encode(data).decode("ascii")


def _message_to_dict(message, include_defaults: bool = False) -> dict:
    """Safe protobuf -> dict conversion across protobuf versions."""
    kwargs = {"preserving_proto_field_name": True}
    if include_defaults and _DEFAULT_FIELDS_KWARG:
        kwargs[_DEFAULT_FIELDS_KWARG] = True
    try:
        return json_format.MessageToDict(message, **kwargs)
    except TypeError:
        # Extremely defensive: if protobuf changes keyword names again, fall back
        # to the minimal call instead of crashing the gRPC handler.
        kwargs.pop("including_default_value_fields", None)
        kwargs.pop("always_print_fields_with_no_presence", None)
        return json_format.MessageToDict(message, **kwargs)


class SensorDataService(sensor_data_pb2_grpc.SensorDataServiceServicer):
    def __init__(self) -> None:
        self.decryptor = AESDecryptor(settings.encryption_key)
        self.decompressor = DataDecompressor()
        self.storage = FileStorage(settings.data_storage_path)
        self.training_manager = TrainingManager(
            max_concurrent=settings.training_max_concurrent,
            check_interval_sec=settings.training_check_interval_sec,
        )
        self.auth_manager = AuthSessionManager(
            max_cached_models=settings.auth_max_cached_models,
            session_ttl_sec=settings.auth_session_ttl_sec,
            max_concurrent_inference=settings.auth_max_concurrent,
        )

    async def StreamSensorData(self, request_iterator, context):
        response_queue: asyncio.Queue = asyncio.Queue()
        pending_inference: set[asyncio.Task] = set()

        async def _consume_packets() -> None:
            async for packet in request_iterator:
                directive = await self._handle_packet(packet, response_queue, pending_inference)
                if directive is not None:
                    await response_queue.put(directive)
            if pending_inference:
                await asyncio.gather(*pending_inference, return_exceptions=True)
            await response_queue.put(None)

        consumer_task = asyncio.create_task(_consume_packets())
        try:
            while True:
                directive = await response_queue.get()
                if directive is None:
                    break
                yield directive
        finally:
            if not consumer_task.done():
                consumer_task.cancel()
                await asyncio.gather(consumer_task, return_exceptions=True)

    async def GetInitialPolicy(self, request, context):
        logger.info(
            "Initial policy requested: device_id_hash=%s, app_version=%s, current_policy=%s",
            getattr(request, "device_id_hash", ""),
            getattr(request, "app_version", ""),
            getattr(request, "current_policy_id", ""),
        )
        return sensor_data_pb2.PolicyUpdate(
            policy_id="default",
            policy_version=settings.version,
            batch_interval_ms=1000,
            max_payload_size_bytes=settings.max_request_size,
            upload_rate_limit=50.0,
            compression_algorithm="LZ4",
            batch_size_threshold=50,
            sensor_sampling_rates={"ACCELEROMETER": 100, "GYROSCOPE": 100, "MAGNETOMETER": 100},
        )

    async def StartAuthentication(self, request, context):
        device_id_hash = getattr(request, "device_id_hash", "") or "unknown_device"
        session_id = getattr(request, "session_id", "") or f"auth_{uuid.uuid4().hex}"
        ca_cfg = get_ca_config()
        if self.auth_manager.has_trained_model(device_id_hash):
            accepted, message, policy = self.auth_manager.start_session(device_id_hash, session_id)
            logger.info(
                "Auth session start: device=%s session=%s accepted=%s message=%s model=%s",
                device_id_hash,
                session_id,
                accepted,
                message,
                policy.model_version if policy else "",
            )
            return sensor_data_pb2.AuthSessionResponse(
                accepted=bool(accepted),
                session_id=str(session_id),
                message=str(message),
                model_version=str(policy.model_version if policy else ""),
                window_size_sec=float(policy.window_size if policy else 0.0),
                decision_time_sec=float(ca_cfg.auth.max_decision_time_sec),
            )

        readiness = self.training_manager.get_readiness(device_id_hash)
        min_bytes = max(readiness.min_bytes, 100 * 1024 * 1024)
        total_mb = readiness.total_bytes / (1024 * 1024)
        min_mb = min_bytes / (1024 * 1024)
        if readiness.total_bytes >= min_bytes:
            await self.training_manager.submit_if_ready(device_id_hash, force=True)
            reason = "training_in_progress"
        else:
            reason = f"data_insufficient: {total_mb:.1f}MB/{min_mb:.0f}MB"

        logger.info(
            "Auth session rejected: device=%s session=%s reason=%s total_mb=%.1f required_mb=%.0f",
            device_id_hash,
            session_id,
            reason,
            total_mb,
            min_mb,
        )
        return sensor_data_pb2.AuthSessionResponse(
            accepted=False,
            session_id=str(session_id),
            message=reason,
            model_version="",
            window_size_sec=0.0,
            decision_time_sec=float(ca_cfg.auth.max_decision_time_sec),
        )

    async def SendHeartbeat(self, request, context):
        server_ts = int(time.time() * 1000)
        logger.debug("Heartbeat received: pending=%s, last_seq=%s", request.pending_packets, request.last_packet_seq_no)
        return sensor_data_pb2.HeartbeatAck(
            server_timestamp=server_ts,
            client_timestamp_echo=request.client_timestamp,
        )

    async def ReportMetrics(self, request, context):
        logger.info(
            "Metrics received: device=%s, period_ms=%s, uploads_success=%s, uploads_failed=%s",
            getattr(request, "device_id_hash", ""),
            getattr(request, "reporting_period_ms", 0),
            getattr(request, "uploads_success", 0),
            getattr(request, "uploads_failed", 0),
        )
        return sensor_data_pb2.MetricsResponse(accepted=True, message="ok")

    async def _handle_packet(
        self,
        packet: sensor_data_pb2.DataPacket,
        response_queue: asyncio.Queue,
        pending_inference: set[asyncio.Task],
    ) -> sensor_data_pb2.ServerDirective:
        received_ms = int(time.time() * 1000)
        device_id_hash = packet.device_id_hash or "unknown_device"
        session_id = "default"
        metadata_dict = None
        decryption_status = "skipped"
        error_detail: Optional[str] = None
        parsed_batch: Optional[dict] = None

        if packet.HasField("metadata"):
            metadata_dict = _message_to_dict(packet.metadata, include_defaults=True)

        # Try decrypt -> decompress -> parse SerializedSensorBatch
        if packet.encrypted_sensor_payload:
            decryption_status = "decrypt_failed"
            decrypt_ok, decrypted_payload, decrypt_err = self.decryptor.decrypt(packet.encrypted_sensor_payload)
            if decrypt_ok and decrypted_payload:
                decryption_status = "decrypt_ok"
                decompress_ok, decompressed_payload, decompress_err = self.decompressor.decompress(
                    decrypted_payload,
                    compression_hint=(packet.metadata.compression if packet.HasField("metadata") else None),
                )
                if decompress_ok and decompressed_payload:
                    decryption_status = "decompressed"
                    try:
                        batch = sensor_data_pb2.SerializedSensorBatch()
                        batch.ParseFromString(decompressed_payload)
                        parsed_batch = _message_to_dict(batch, include_defaults=True)
                        # Ensure axis fields exist even if zero to avoid “missing” impressions
                        for sample in parsed_batch.get("samples", []):
                            for axis in ("x", "y", "z"):
                                sample.setdefault(axis, 0.0)
                        session_id = batch.session_id or session_id
                        decryption_status = "parsed_sensor_batch"
                    except Exception as exc:  # noqa: BLE001
                        decryption_status = "parse_failed"
                        error_detail = f"parse_batch_failed: {exc}"
                        logger.warning("Failed to parse SerializedSensorBatch: %s", exc)
                else:
                    error_detail = decompress_err
                    logger.warning("Decompression failed for packet %s: %s", packet.packet_id, decompress_err)
            else:
                error_detail = decrypt_err
                logger.warning("Decryption failed for packet %s: %s", packet.packet_id, decrypt_err)
        else:
            error_detail = "no_encrypted_payload"
            logger.warning("Received packet without encrypted payload: %s", packet.packet_id)

        packet_record = {
            "packet_id": packet.packet_id,
            "device_id_hash": device_id_hash,
            "packet_seq_no": packet.packet_seq_no,
            "base_wall_ms": packet.base_wall_ms,
            "device_uptime_ns": packet.device_uptime_ns,
            "ntp_offset_ms": packet.ntp_offset_ms if packet.HasField("ntp_offset_ms") else None,
            "metadata": metadata_dict,
            "decryption_status": decryption_status,
            "decryption_error": error_detail,
            "encrypted_dek_b64": _b64(packet.encrypted_dek),
            "sha256_b64": _b64(packet.sha256),
        }

        if parsed_batch:
            packet_record["sensor_batch"] = parsed_batch
            session_id = parsed_batch.get("session_id", session_id)
        else:
            packet_record["encrypted_sensor_payload_b64"] = _b64(packet.encrypted_sensor_payload)

        storage_ok, storage_err = await self.storage.append_packet(
            device_id_hash=device_id_hash,
            session_id=str(session_id),
            packet_data=packet_record,
        )

        if not storage_ok:
            logger.error("Failed to store packet %s: %s", packet.packet_id, storage_err)
        else:
            asyncio.create_task(self.training_manager.submit_if_ready(device_id_hash))

        if parsed_batch:
            task = asyncio.create_task(
                self._run_auth_inference(
                    device_id_hash=device_id_hash,
                    session_id=str(session_id),
                    parsed_batch=parsed_batch,
                    response_queue=response_queue,
                )
            )
            pending_inference.add(task)
            task.add_done_callback(lambda t: pending_inference.discard(t))

        ack = sensor_data_pb2.Ack(
            packet_id=packet.packet_id,
            creation_server_ts=received_ms,
            success=storage_ok,
            error_code="SERVER_ERROR" if not storage_ok else "",
        )

        directive = sensor_data_pb2.ServerDirective()
        directive.ack.CopyFrom(ack)
        return directive

    async def _run_auth_inference(
        self,
        *,
        device_id_hash: str,
        session_id: str,
        parsed_batch: dict,
        response_queue: asyncio.Queue,
    ) -> None:
        start_ms = int(time.time() * 1000)
        payload = await self.auth_manager.handle_packet(
            user_id=device_id_hash,
            session_id=session_id,
            parsed_batch=parsed_batch,
        )
        if payload is None:
            return
        end_ms = int(time.time() * 1000)
        inference_latency_ms = max(0, end_ms - start_ms)
        result = sensor_data_pb2.AuthResult(
            device_id_hash=str(device_id_hash),
            session_id=str(session_id),
            server_timestamp_ms=int(time.time() * 1000),
            score=float(payload.score),
            threshold=float(payload.threshold),
            accept=bool(payload.accept),
            interrupt=bool(payload.interrupt),
            window_size_sec=float(payload.window_size),
            window_id=int(payload.window_id),
            normalized_score=float(payload.normalized_score),
            k_rejects=int(payload.k_rejects),
            model_version=str(payload.model_version),
            message=str(payload.message),
        )
        logger.info(
            "Auth result: device=%s session=%s window_id=%s score=%.6f normalized=%.6f threshold=%.6f "
            "accept=%s interrupt=%s model=%s inference_latency_ms=%s",
            device_id_hash,
            session_id,
            payload.window_id,
            payload.score,
            payload.normalized_score,
            payload.threshold,
            payload.accept,
            payload.interrupt,
            payload.model_version,
            inference_latency_ms,
        )
        if payload.interrupt or not payload.accept:
            logger.warning(
                "Auth failure: device=%s session=%s window_id=%s accept=%s interrupt=%s score=%.6f",
                device_id_hash,
                session_id,
                payload.window_id,
                payload.accept,
                payload.interrupt,
                payload.score,
            )
        directive = sensor_data_pb2.ServerDirective()
        directive.auth_result.CopyFrom(result)
        await response_queue.put(directive)


def _build_server_credentials() -> Tuple[Optional[grpc.ServerCredentials], TLSProbeResult]:
    probe = probe_tls_configuration(
        settings.tls_certfile,
        settings.tls_keyfile,
        settings.tls_ca_certs,
        key_password=settings.tls_keyfile_password,
        allow_encrypted_key=False,  # gRPC python cannot decrypt password-protected private keys
    )

    if not probe.is_tls:
        if probe.reason == "cert_or_key_not_configured":
            logger.info("No TLS certificate configured; gRPC will serve via h2c (cleartext HTTP/2).")
        elif probe.reason == "key_password_not_supported_for_grpc":
            logger.warning(
                "TLS keyfile password detected but gRPC cannot use encrypted private keys; using h2c instead "
                "(cert=%s, key=%s).",
                probe.certfile,
                probe.keyfile,
            )
        else:
            logger.warning(
                "TLS disabled for gRPC due to %s; using h2c (cert=%s, key=%s).",
                probe.reason,
                probe.certfile,
                probe.keyfile,
            )
        return None, probe

    os.environ.setdefault("GRPC_SSL_CIPHER_SUITES", TLS12_13_CIPHERS)

    credentials = grpc.ssl_server_credentials(
        [(probe.private_key, probe.cert_chain)],
        root_certificates=probe.root_certs,
    )
    return credentials, probe


async def create_grpc_server() -> grpc.aio.Server:
    server = grpc.aio.server(
        maximum_concurrent_rpcs=int(settings.grpc_max_concurrent_rpcs),
        options=[
            ("grpc.max_send_message_length", settings.grpc_max_message_size),
            ("grpc.max_receive_message_length", settings.grpc_max_message_size),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 5_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ]
    )
    sensor_data_pb2_grpc.add_SensorDataServiceServicer_to_server(SensorDataService(), server)
    # gRPC health service so we can probe the single exposed port.
    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("com.continuousauth.proto.SensorDataService", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    creds, probe = _build_server_credentials()
    bind_addr = f"{settings.grpc_host}:{settings.grpc_port}"

    if creds:
        bound_port = server.add_secure_port(bind_addr, creds)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind secure gRPC port on {bind_addr}")
        logger.info(
            "gRPC server listening on %s (TLS 1.2/1.3, ALPN h2; cert=%s, key=%s)",
            bind_addr,
            probe.certfile,
            probe.keyfile,
        )
    else:
        bound_port = server.add_insecure_port(bind_addr)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC port on {bind_addr}")
        logger.info(
            "gRPC server listening on %s (h2c cleartext%s)",
            bind_addr,
            f", reason={probe.reason}" if probe.reason else "",
        )

    return server
