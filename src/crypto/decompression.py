import gzip
import logging
from typing import Optional, Tuple

import lz4.block
import lz4.frame

logger = logging.getLogger(__name__)


class DataDecompressor:
    """Adaptive decompressor supporting both LZ4 and GZIP payloads."""

    LZ4_MAGIC = b"\x04\x22\x4d\x18"
    GZIP_MAGIC = b"\x1f\x8b"

    def __init__(self, max_output_size: Optional[int] = None):
        self.max_output_size = int(max_output_size) if max_output_size else None
        logger.info("DataDecompressor initialized with LZ4 & GZIP support")

    def decompress(
        self,
        compressed_data: bytes,
        compression_hint: Optional[str] = None,
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        try:
            if not compressed_data:
                error_msg = "Empty compressed data"
                logger.error(error_msg)
                return False, None, error_msg

            logger.debug(
                "Attempting decompression: bytes=%s, hint=%s",
                len(compressed_data),
                compression_hint,
            )

            strategies = self._build_strategies(compression_hint, compressed_data)
            last_error: Optional[str] = None

            for strategy in strategies:
                try:
                    decompressed = strategy(compressed_data)
                    size_error = self._validate_output_size(decompressed)
                    if size_error:
                        logger.error(size_error)
                        return False, None, size_error
                    logger.info(
                        "Successfully decompressed %s bytes to %s bytes using %s",
                        len(compressed_data),
                        len(decompressed),
                        strategy.__name__,
                    )
                    return True, decompressed, None
                except Exception as inner_exc:  # noqa: BLE001 - contextual logging below
                    last_error = f"{strategy.__name__} failed: {inner_exc}"
                    if "exceeds limit" in str(inner_exc) or "too large" in str(inner_exc):
                        logger.error(last_error)
                        return False, None, last_error
                    logger.debug(last_error)

            error_msg = last_error or "Unsupported compression format"
            logger.error(error_msg)
            return False, None, error_msg

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Decompression failed: {exc}"
            logger.error(error_msg)
            return False, None, error_msg

    def _build_strategies(self, hint: Optional[str], data: bytes):
        hint_lower = (hint or "").lower()
        strategies = []

        if hint_lower == "lz4":
            strategies.append(self._decompress_lz4)
        elif hint_lower == "gzip":
            strategies.append(self._decompress_gzip)
        elif hint_lower:
            logger.warning("Unsupported compression hint '%s', falling back to auto-detect", hint)

        if len(data) >= 4:
            magic_prefix = data[:4]
            if magic_prefix.startswith(self.GZIP_MAGIC):
                strategies.append(self._decompress_gzip)
            elif magic_prefix == self.LZ4_MAGIC:
                strategies.append(self._decompress_lz4)

        if self._decompress_lz4 not in strategies:
            strategies.append(self._decompress_lz4)
        if self._decompress_gzip not in strategies:
            strategies.append(self._decompress_gzip)

        return strategies

    def _validate_output_size(self, payload: bytes) -> Optional[str]:
        if self.max_output_size is None:
            return None
        if len(payload) > self.max_output_size:
            return (
                f"Decompressed payload too large: {len(payload)} bytes "
                f"(limit {self.max_output_size} bytes)"
            )
        return None

    def _validate_expected_size(self, expected_size: int) -> None:
        if self.max_output_size is not None and expected_size > self.max_output_size:
            raise ValueError(
                f"declared decompressed size {expected_size} exceeds limit {self.max_output_size}"
            )

    def _decompress_lz4(self, data: bytes) -> bytes:
        try:
            frame_info = lz4.frame.get_frame_info(data)
            content_size = int(frame_info.get("content_size") or 0)
            if content_size > 0:
                self._validate_expected_size(content_size)
        except RuntimeError:
            pass
        # 先尝试标准 LZ4 frame
        try:
            return lz4.frame.decompress(data)
        except Exception:
            # 兼容客户端使用的“长度前缀 + 原始块”格式
            if len(data) > 4:
                expected_size = int.from_bytes(data[:4], byteorder="big", signed=False)
                self._validate_expected_size(expected_size)
                raw = data[4:]
                return lz4.block.decompress(raw, uncompressed_size=expected_size)
            # 如果没有长度前缀，继续抛出异常以便外层尝试其他策略
            raise

    def _decompress_gzip(self, data: bytes) -> bytes:
        if len(data) >= 4:
            expected_size = int.from_bytes(data[-4:], byteorder="little", signed=False)
            if expected_size:
                self._validate_expected_size(expected_size)
        return gzip.decompress(data)
