import json

import pytest

from src.storage.inference_storage import InferenceStorage


@pytest.mark.asyncio
async def test_inference_storage_writes(tmp_path) -> None:
    storage = InferenceStorage(tmp_path)

    ok, err = await storage.append_raw_packet("user1", "session1", {"packet": 1})
    assert ok, err
    ok, err = await storage.append_result("user1", "session1", {"accept": True})
    assert ok, err

    raw_path = tmp_path / "user1" / "session1" / "raw.jsonl"
    result_path = tmp_path / "user1" / "session1" / "results.jsonl"
    assert raw_path.exists()
    assert result_path.exists()

    raw_payload = json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])
    result_payload = json.loads(result_path.read_text(encoding="utf-8").splitlines()[0])
    assert raw_payload["packet"] == 1
    assert result_payload["accept"] is True
