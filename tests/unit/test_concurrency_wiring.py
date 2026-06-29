"""验证并发参数从 settings 正确传导到训练/推理管理器（问题2修复）。

- 训练：TrainingManager 的 asyncio.Semaphore 初值 = settings.training_max_concurrent（真正并行执行上限）。
- 推理：AuthSessionManager 的推理信号量 = settings.auth_max_concurrent；模型 LRU 缓存 = settings.auth_max_cached_models。
"""
from __future__ import annotations

import pytest


def test_runtime_context_wires_concurrency(monkeypatch, tmp_path) -> None:
    pytest.importorskip("torch")  # AuthSessionManager 间接依赖 torch
    import src.management.runtime as rt
    from src.config import settings

    monkeypatch.setattr(settings, "training_max_concurrent", 8)
    monkeypatch.setattr(settings, "auth_max_concurrent", 100)
    monkeypatch.setattr(settings, "auth_max_cached_models", 100)
    monkeypatch.setattr(settings, "data_storage_path", tmp_path / "raw_data")
    monkeypatch.setattr(settings, "inference_storage_path", tmp_path / "inference")

    ctx = rt.create_runtime_context()

    # 训练：真正并行执行上限 = 8（每卡 1 个）。
    assert ctx.training_manager._semaphore._value == 8
    # 推理：100 并发前向 + 100 个 per-user 模型缓存。
    assert ctx.auth_manager._inference_semaphore._value == 100
    assert ctx.auth_manager._model_cache._max_models == 100


def test_training_manager_semaphore_supports_100(monkeypatch, tmp_path) -> None:
    """若把执行上限调大到 100，TrainingManager 也能正确接受（队列接纳由 asyncio 任务保证）。"""
    from src.training.manager import TrainingManager

    mgr = TrainingManager(max_concurrent=100, check_interval_sec=30)
    assert mgr._semaphore._value == 100
