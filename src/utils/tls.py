import logging
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# TLS 1.2/1.3 cipher suites only; prevents downgrades to legacy TLS versions.
TLS12_13_CIPHERS = (
    "TLS_AES_256_GCM_SHA384:"
    "TLS_AES_128_GCM_SHA256:"
    "TLS_CHACHA20_POLY1305_SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:"
    "ECDHE-RSA-AES256-GCM-SHA384:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES128-GCM-SHA256"
)

logger = logging.getLogger(__name__)


@dataclass
class TLSProbeResult:
    mode: str
    reason: str
    certfile: Optional[Path]
    keyfile: Optional[Path]
    ca_certs: Optional[Path]
    cert_chain: Optional[bytes] = None
    private_key: Optional[bytes] = None
    root_certs: Optional[bytes] = None
    ssl_context: Optional[ssl.SSLContext] = None

    @property
    def is_tls(self) -> bool:
        return self.mode == "tls"


def probe_tls_configuration(
    cert_path: Optional[Path],
    key_path: Optional[Path],
    ca_path: Optional[Path] = None,
    *,
    key_password: Optional[str] = None,
    allow_encrypted_key: bool = True,
) -> TLSProbeResult:
    """Determine whether TLS can be enabled and load materials when available.

    Returns TLSProbeResult with mode "tls" when cert/key are present and valid,
    otherwise "h2c" with a reason to fall back to cleartext HTTP/2.
    """
    if not (cert_path and key_path):
        return TLSProbeResult(
            mode="h2c",
            reason="cert_or_key_not_configured",
            certfile=cert_path,
            keyfile=key_path,
            ca_certs=ca_path,
        )

    if key_password and not allow_encrypted_key:
        return TLSProbeResult(
            mode="h2c",
            reason="key_password_not_supported_for_grpc",
            certfile=cert_path,
            keyfile=key_path,
            ca_certs=ca_path,
        )

    missing = [name for path, name in ((cert_path, "cert"), (key_path, "key")) if not path.is_file()]
    if missing:
        return TLSProbeResult(
            mode="h2c",
            reason=f"missing_{'_'.join(missing)}",
            certfile=cert_path,
            keyfile=key_path,
            ca_certs=ca_path,
        )

    try:
        cert_chain = cert_path.read_bytes()
        private_key = key_path.read_bytes()
    except OSError as exc:
        return TLSProbeResult(
            mode="h2c",
            reason=f"read_error:{exc}",
            certfile=cert_path,
            keyfile=key_path,
            ca_certs=ca_path,
        )

    root_certs: Optional[bytes] = None
    if ca_path:
        if ca_path.is_file():
            try:
                root_certs = ca_path.read_bytes()
            except OSError as exc:
                logger.warning("Failed to read TLS CA bundle from %s: %s", ca_path, exc)
        else:
            logger.warning("TLS CA bundle configured but not found at %s; skipping CA load", ca_path)
            ca_path = None

    try:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        ssl_context.set_ciphers(TLS12_13_CIPHERS)
        ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path, password=key_password)
        if ca_path:
            ssl_context.load_verify_locations(cafile=ca_path)
    except (ssl.SSLError, ValueError) as exc:
        logger.warning("TLS configuration invalid, falling back to h2c: %s", exc)
        return TLSProbeResult(
            mode="h2c",
            reason=f"ssl_error:{exc}",
            certfile=cert_path,
            keyfile=key_path,
            ca_certs=ca_path,
            cert_chain=cert_chain,
            private_key=private_key,
            root_certs=root_certs,
        )

    return TLSProbeResult(
        mode="tls",
        reason="",
        certfile=cert_path,
        keyfile=key_path,
        ca_certs=ca_path,
        cert_chain=cert_chain,
        private_key=private_key,
        root_certs=root_certs,
        ssl_context=ssl_context,
    )
