import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import aiofiles
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class FileStorage:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStorage initialized with base path: {self.base_path}")

    def _get_session_file_path(self, device_id_hash: str, session_id: str) -> Path:
        device_dir = self.base_path / device_id_hash
        device_dir.mkdir(parents=True, exist_ok=True)
        return device_dir / f"session_{session_id}.jsonl"

    async def append_packet(
        self,
        device_id_hash: str,
        session_id: str,
        packet_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        try:
            file_path = self._get_session_file_path(device_id_hash, session_id)

            packet_data["server_received_timestamp"] = datetime.now(timezone.utc).isoformat()

            json_line = json.dumps(packet_data, ensure_ascii=False) + '\n'

            async with aiofiles.open(file_path, mode='a', encoding='utf-8') as f:
                await f.write(json_line)

            logger.info(
                f"Successfully stored packet: device={device_id_hash}, "
                f"session={session_id}, seq={packet_data.get('packet_seq_no', 'unknown')}, "
                f"file={file_path}"
            )
            return True, None

        except Exception as e:
            error_msg = f"Failed to store packet: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def read_session(
        self,
        device_id_hash: str,
        session_id: str
    ) -> Tuple[bool, Optional[list], Optional[str]]:
        try:
            file_path = self._get_session_file_path(device_id_hash, session_id)

            if not file_path.exists():
                return True, [], None

            packets = []
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                async for line in f:
                    if line.strip():
                        packets.append(json.loads(line))

            packets.sort(key=lambda x: x.get('packet_seq_no', 0))

            logger.info(f"Read {len(packets)} packets from session: {file_path}")
            return True, packets, None

        except Exception as e:
            error_msg = f"Failed to read session: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def get_storage_stats(self) -> Dict[str, Any]:
        try:
            total_devices = 0
            total_sessions = 0
            total_size_bytes = 0

            if self.base_path.exists():
                for device_dir in self.base_path.iterdir():
                    if device_dir.is_dir():
                        total_devices += 1
                        for session_file in device_dir.glob("session_*.jsonl"):
                            total_sessions += 1
                            total_size_bytes += session_file.stat().st_size

            return {
                "base_path": str(self.base_path),
                "total_devices": total_devices,
                "total_sessions": total_sessions,
                "total_size_mb": round(total_size_bytes / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {
                "base_path": str(self.base_path),
                "error": str(e)
            }
