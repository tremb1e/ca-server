from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiofiles
import logging

logger = logging.getLogger(__name__)


class InferenceStorage:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("InferenceStorage initialized at %s", self.base_path)

    def _get_session_dir(self, device_id_hash: str, session_id: str) -> Path:
        session_dir = self.base_path / device_id_hash / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    async def append_raw_packet(
        self,
        device_id_hash: str,
        session_id: str,
        packet_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            session_dir = self._get_session_dir(device_id_hash, session_id)
            file_path = session_dir / "raw.jsonl"
            payload = dict(packet_data)
            payload["server_received_timestamp"] = datetime.now(timezone.utc).isoformat()
            async with aiofiles.open(file_path, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return True, None
        except Exception as exc:
            error_msg = f"Failed to store inference raw packet: {exc}"
            logger.error(error_msg)
            return False, error_msg

    async def append_result(
        self,
        device_id_hash: str,
        session_id: str,
        result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        try:
            session_dir = self._get_session_dir(device_id_hash, session_id)
            file_path = session_dir / "results.jsonl"
            payload = dict(result)
            payload["server_written_timestamp"] = datetime.now(timezone.utc).isoformat()
            async with aiofiles.open(file_path, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return True, None
        except Exception as exc:
            error_msg = f"Failed to store inference result: {exc}"
            logger.error(error_msg)
            return False, error_msg
