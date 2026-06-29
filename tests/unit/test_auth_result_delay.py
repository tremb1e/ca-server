"""认证结果发布延迟（auth.result_delay_sec）测试。

覆盖：
1. load_ca_config 解析 `[auth] result_delay_sec`（含默认值与负值钳制）。
2. windows_for_delay：把延迟秒数折算成需累积的窗口数。
3. ResultEmitGate：按折算窗口数做发布节流（含余数进位）。
4. AuthSessionManager.handle_packet：真实链路按延迟节流向 App 发布 AuthResult，
   全量窗口仍照常累计（重依赖被 monkeypatch，只验证发布节奏）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils.reject_trackers import ResultEmitGate, windows_for_delay


# --------------------------------------------------------------------------- #
# 1. 配置解析
# --------------------------------------------------------------------------- #
def test_load_ca_config_parses_result_delay_sec(tmp_path) -> None:
    from src.ca_config import load_ca_config

    p = tmp_path / "ca_config.toml"
    p.write_text("[auth]\nresult_delay_sec = 5.0\nmax_decision_time_sec = 20.0\n", encoding="utf-8")
    cfg = load_ca_config(p)
    assert cfg.auth.result_delay_sec == 5.0
    assert cfg.auth.max_decision_time_sec == 20.0


def test_load_ca_config_defaults_result_delay_sec_zero(tmp_path) -> None:
    from src.ca_config import load_ca_config

    p = tmp_path / "ca_config.toml"
    p.write_text("[auth]\nmax_decision_time_sec = 2.0\n", encoding="utf-8")
    cfg = load_ca_config(p)
    assert cfg.auth.result_delay_sec == 0.0


def test_load_ca_config_clamps_negative_result_delay(tmp_path) -> None:
    from src.ca_config import load_ca_config

    p = tmp_path / "ca_config.toml"
    p.write_text("[auth]\nresult_delay_sec = -3.0\n", encoding="utf-8")
    cfg = load_ca_config(p)
    assert cfg.auth.result_delay_sec == 0.0


# --------------------------------------------------------------------------- #
# 2. 秒 -> 窗口数 折算
# --------------------------------------------------------------------------- #
def test_windows_for_delay_converts_seconds_to_window_count() -> None:
    # 0.2s 窗 + 50% overlap => stride 0.1s => 每秒约 10 窗
    assert windows_for_delay(5.0, window_size_sec=0.2, overlap=0.5) == 50
    assert windows_for_delay(1.0, window_size_sec=0.2, overlap=0.5) == 10
    assert windows_for_delay(2.0, window_size_sec=0.2, overlap=0.5) == 20
    # overlap=0 => stride = window_size
    assert windows_for_delay(2.0, window_size_sec=0.5, overlap=0.0) == 4


def test_windows_for_delay_zero_or_negative_disables_throttle() -> None:
    assert windows_for_delay(0.0, window_size_sec=0.2, overlap=0.5) == 0
    assert windows_for_delay(-1.0, window_size_sec=0.2, overlap=0.5) == 0


def test_windows_for_delay_clamps_and_guards() -> None:
    # 正延迟但短于一个 stride，至少 1 窗
    assert windows_for_delay(0.01, window_size_sec=0.2, overlap=0.5) == 1
    # overlap>=1 => stride<=0，视为不可用返回 0
    assert windows_for_delay(5.0, window_size_sec=0.2, overlap=1.0) == 0
    assert windows_for_delay(5.0, window_size_sec=0.0, overlap=0.5) == 0


# --------------------------------------------------------------------------- #
# 3. 发布节流闸门
# --------------------------------------------------------------------------- #
def test_result_emit_gate_fires_every_period_with_remainder_carry() -> None:
    gate = ResultEmitGate(target_windows=50)
    fired = [i for i in range(1, 101) if gate.feed()]  # 连续喂 100 个窗口
    assert fired == [50, 100]
    assert gate.pending == 0


def test_result_emit_gate_disabled_never_fires() -> None:
    gate = ResultEmitGate(target_windows=0)
    assert not any(gate.feed() for _ in range(100))


def test_result_emit_gate_carries_uneven_batches() -> None:
    # 每包 9 窗、target=50：累计跨包进位，108 窗 => 发布 2 次
    gate = ResultEmitGate(target_windows=50)
    fires = sum(1 for _ in range(12) for _w in range(9) if gate.feed())
    assert fires == 2
    assert gate.pending == 8  # 108 - 2*50


# --------------------------------------------------------------------------- #
# 4. handle_packet 真实链路（重依赖 monkeypatch）
# --------------------------------------------------------------------------- #
def _build_throttle_manager(monkeypatch, tmp_path, *, result_delay_sec, user, session, windows_per_packet=10):
    """构造一个 AuthSessionManager，stub 掉重依赖，注入一个已就绪的认证会话。"""
    pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")  # src.authentication.vqgan_inference 模块导入即依赖 torch
    import pandas as pd

    import src.authentication.manager as mgr_mod
    import src.authentication.vqgan_inference as vq_mod
    import src.processing.pipeline as pipeline_mod
    import src.processing.scaler as scaler_mod
    from src.authentication.manager import AuthSessionManager, AuthSessionState
    from src.authentication.vqgan_inference import VQGANPolicy
    from src.ca_config import AuthConfig, CAConfig, WindowConfig
    from src.config import settings
    from src.utils.reject_trackers import (
        ConsecutiveRejectTracker,
        EMAScoreTracker,
        ResultEmitGate,
        VoteRejectTracker,
    )

    cfg = CAConfig(
        auth=AuthConfig(decision_strategy="ema", ema_alpha=0.25, result_delay_sec=result_delay_sec),
        windows=WindowConfig(sizes=[0.2], overlap=0.5, sampling_rate_hz=100),
    )
    monkeypatch.setattr(mgr_mod, "get_ca_config", lambda *a, **k: cfg)

    # handle_packet 内部按需 import 的重依赖，全部替换为轻量桩。
    monkeypatch.setattr(pipeline_mod, "build_config", lambda: SimpleNamespace(sampling_rate_hz=100))
    monkeypatch.setattr(pipeline_mod, "_extract_sensor_records", lambda packets: {"acc": [], "gyr": [], "mag": []})
    monkeypatch.setattr(pipeline_mod, "_resample_records", lambda *a, **k: pd.DataFrame({"acc_x": [0.0] * 5}))
    monkeypatch.setattr(scaler_mod, "load_scaler", lambda p: {"dummy": True})
    monkeypatch.setattr(scaler_mod, "apply_scaler", lambda df, scaler: df)
    monkeypatch.setattr(
        vq_mod,
        "windowize_dataframe",
        lambda *a, **k: (list(range(windows_per_packet)), np.zeros((windows_per_packet, 1, 6, 20), dtype="float32")),
    )
    monkeypatch.setattr(
        vq_mod,
        "score_windows",
        lambda *a, **k: np.zeros((windows_per_packet,), dtype="float32"),
    )

    # scaler.json 必须存在（内容无关，load_scaler 已被 stub）。
    monkeypatch.setattr(settings, "processed_data_path", str(tmp_path))
    scaler_path = tmp_path / "z-score" / user / "scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text("{}", encoding="utf-8")

    mgr = AuthSessionManager(models_root=tmp_path / "models")

    class _FakeInferenceStorage:
        async def append_raw_packet(self, *a, **k):
            return None

        async def append_result(self, *a, **k):
            return None

    mgr._inference_storage = _FakeInferenceStorage()
    fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    monkeypatch.setattr(mgr._model_cache, "get", lambda policy: fake_model)

    policy = VQGANPolicy(
        user=user,
        window_size=0.2,
        overlap=0.5,
        target_width=20,
        threshold=0.0,
        interrupt_rule="ema",
        decision_strategy="ema",
        k_rejects=0,
        vqgan_checkpoint=tmp_path / "m.pt",
        vqgan_config=tmp_path / "m.json",
        vote_window_size=0,
        vote_min_rejects=0,
        ema_alpha=0.25,
        model_version="v1",
    )
    state = AuthSessionState(user_id=user, session_id=session, policy=policy)
    state.consecutive_rejects = ConsecutiveRejectTracker()
    state.vote_rejects = VoteRejectTracker()
    state.ema_rejects = EMAScoreTracker()
    state.emit_gate = ResultEmitGate()
    mgr._sessions[mgr._session_key(user, session)] = state
    return mgr


@pytest.mark.asyncio
async def test_handle_packet_publishes_once_per_delay_period(monkeypatch, tmp_path) -> None:
    user, session = "user-delay", "sess-1"
    # delay=5s、每包 10 窗 => target 50 窗 => 每 5 个包发布一次。
    mgr = _build_throttle_manager(monkeypatch, tmp_path, result_delay_sec=5.0, user=user, session=session)
    parsed_batch = {"session_id": session, "samples": []}

    published = []
    for _ in range(10):  # 10 包 * 10 窗 = 100 窗
        payload = await mgr.handle_packet(user_id=user, session_id=session, parsed_batch=parsed_batch)
        published.append(payload is not None)

    assert published == [False, False, False, False, True, False, False, False, False, True]
    # 节流只影响向 App 的回包；全量窗口仍逐窗处理，window_index 累计到 100。
    state = mgr._sessions[mgr._session_key(user, session)]
    assert state.window_index == 100
    assert state.emit_gate.target_windows == 50


@pytest.mark.asyncio
async def test_handle_packet_publishes_every_packet_when_delay_zero(monkeypatch, tmp_path) -> None:
    user, session = "user-nodelay", "sess-2"
    mgr = _build_throttle_manager(monkeypatch, tmp_path, result_delay_sec=0.0, user=user, session=session)
    parsed_batch = {"session_id": session, "samples": []}

    published = []
    for _ in range(3):
        payload = await mgr.handle_packet(user_id=user, session_id=session, parsed_batch=parsed_batch)
        published.append(payload is not None)

    # 不延迟时保留旧行为：每个数据包都回一次结果。
    assert published == [True, True, True]
