"""验证 ca_config.toml [auth] 对 per-user best_lock_policy.json 的权威覆盖（问题1修复）。

修复点：认证运行时的“聚合策略 / 投票窗口 / 阈值计数”一律以 ca_config.toml [auth] 为准，
覆盖 policy_search 为该用户产出的投票窗口（如 5-of-9），从而保证 App 端展示的
“M of N”（如 30-of-50）始终来自配置。per-user 策略仅继续提供单窗口原始分数阈值 threshold
与模型工件路径。仍为单一 primary 阶段，不引入任何二次迟滞/二次投票。
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


def _make_manager(monkeypatch, tmp_path, *, ca_cfg):
    pytest.importorskip("torch")  # vqgan_inference 导入即依赖 torch
    import src.authentication.manager as mgr_mod
    from src.authentication.manager import AuthSessionManager

    monkeypatch.setattr(mgr_mod, "get_ca_config", lambda *a, **k: ca_cfg)
    return AuthSessionManager(models_root=tmp_path / "models")


def _peruser_cfg(tmp_path, **overrides):
    """构造一个模拟 per-user best_lock_policy.json 的 AuthRunConfig（部署现状全为 ema）。"""
    from src.authentication.runner import AuthRunConfig

    base = dict(
        user="u1",
        window_size=0.2,
        target_width=20,
        overlap=0.5,
        threshold=-0.81,            # per-user 原始分数阈值，应被保留
        interrupt_rule="ema",
        decision_strategy="ema",    # per-user 策略
        k_rejects=0,
        vote_window_size=0,         # per-user 无投票窗口
        vote_min_rejects=0,
        ema_alpha=0.9,
        vqgan_checkpoint=tmp_path / "m.pt",
        vqgan_config=tmp_path / "m.json",
        model_version="v1",
    )
    base.update(overrides)
    return AuthRunConfig(**base)


def _ca_cfg(**auth_overrides):
    from src.ca_config import AuthConfig, CAConfig, WindowConfig

    auth = dict(
        decision_strategy="vote",
        vote_window_size=50,
        vote_min_rejects=30,
        ema_alpha=0.25,
        result_delay_sec=5.0,
        max_decision_time_sec=20.0,
    )
    auth.update(auth_overrides)
    return CAConfig(
        auth=AuthConfig(**auth),
        windows=WindowConfig(sizes=[0.2], overlap=0.5, sampling_rate_hz=100),
    )


def test_config_overrides_peruser_vote_window(monkeypatch, tmp_path) -> None:
    """config 的 vote 30/50 覆盖 per-user 的 ema 策略；threshold 仍来自 per-user。"""
    mgr = _make_manager(monkeypatch, tmp_path, ca_cfg=_ca_cfg())
    policy = mgr._policy_from_config(_peruser_cfg(tmp_path))
    assert policy.decision_strategy == "vote"
    assert policy.vote_window_size == 50      # 来自 ca_config.toml，而非 per-user(0)
    assert policy.vote_min_rejects == 30      # 来自 ca_config.toml
    assert policy.threshold == pytest.approx(-0.81)  # per-user 原始分数阈值被保留
    assert policy.ema_alpha == pytest.approx(0.25)   # 聚合参数来自 config


def test_config_overrides_even_when_peruser_is_vote(monkeypatch, tmp_path) -> None:
    """即使 per-user 已是 vote 9/5（即旧的 “5 of 9” 来源），也被 config 30/50 覆盖。"""
    mgr = _make_manager(monkeypatch, tmp_path, ca_cfg=_ca_cfg())
    peruser = _peruser_cfg(
        tmp_path,
        decision_strategy="vote",
        interrupt_rule="vote",
        vote_window_size=9,
        vote_min_rejects=5,
    )
    policy = mgr._policy_from_config(peruser)
    assert policy.decision_strategy == "vote"
    assert policy.vote_window_size == 50
    assert policy.vote_min_rejects == 30


def test_config_ema_strategy_overrides_peruser_vote(monkeypatch, tmp_path) -> None:
    """config=ema 覆盖 per-user=vote（策略选择以 config 为准）。"""
    mgr = _make_manager(monkeypatch, tmp_path, ca_cfg=_ca_cfg(decision_strategy="ema"))
    peruser = _peruser_cfg(
        tmp_path,
        decision_strategy="vote",
        interrupt_rule="vote",
        vote_window_size=9,
        vote_min_rejects=5,
    )
    policy = mgr._policy_from_config(peruser)
    assert policy.decision_strategy == "ema"


def test_vote_message_format_uses_config_window(monkeypatch, tmp_path) -> None:
    """投票满窗后的展示文案使用 config 的 30/50（即 App 端 “30 of 50” 来源）。"""
    mgr = _make_manager(monkeypatch, tmp_path, ca_cfg=_ca_cfg())
    msg = mgr._format_vote_message(window_size=50, min_rejects=30, recent_windows=50, recent_rejects=31)
    assert "50" in msg and "30" in msg  # 近50窗恶意 31/50，阈值 30


def test_vote_window_mismatch_warns(monkeypatch, tmp_path, caplog) -> None:
    """vote_window_size 与 result_delay_sec 折算窗口数(50)不一致时打告警，便于排查配置。"""
    mgr = _make_manager(monkeypatch, tmp_path, ca_cfg=_ca_cfg(vote_window_size=9, vote_min_rejects=5))
    with caplog.at_level(logging.WARNING, logger="src.authentication.manager"):
        policy = mgr._policy_from_config(_peruser_cfg(tmp_path))
    assert policy.vote_window_size == 9
    assert any("result_delay_sec" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_handle_packet_emits_config_vote_30_of_50(monkeypatch, tmp_path) -> None:
    """端到端：config=vote 30/50 + per-user=ema，满 50 窗后回包 message 为“近50窗恶意 …/50，阈值 30”。

    这证明 App 端最终收到的窗口阈值确实来自 ca_config.toml，而非 per-user 策略。
    """
    pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    import pandas as pd

    import src.authentication.manager as mgr_mod
    import src.authentication.vqgan_inference as vq_mod
    import src.processing.pipeline as pipeline_mod
    import src.processing.scaler as scaler_mod
    from src.authentication.manager import AuthSessionManager, AuthSessionState
    from src.config import settings
    from src.utils.reject_trackers import (
        ConsecutiveRejectTracker,
        EMAScoreTracker,
        ResultEmitGate,
        VoteRejectTracker,
    )

    user, session = "u-e2e", "s1"
    ca_cfg = _ca_cfg()  # vote 50/30, result_delay_sec=5.0 => target 50 窗
    monkeypatch.setattr(mgr_mod, "get_ca_config", lambda *a, **k: ca_cfg)

    wpp = 10  # windows per packet
    monkeypatch.setattr(pipeline_mod, "build_config", lambda: SimpleNamespace(sampling_rate_hz=100))
    monkeypatch.setattr(pipeline_mod, "_extract_sensor_records", lambda packets: {"acc": [], "gyr": [], "mag": []})
    monkeypatch.setattr(pipeline_mod, "_resample_records", lambda *a, **k: pd.DataFrame({"acc_x": [0.0] * 5}))
    monkeypatch.setattr(scaler_mod, "load_scaler", lambda p: {"d": True})
    monkeypatch.setattr(scaler_mod, "apply_scaler", lambda df, scaler: df)
    monkeypatch.setattr(
        vq_mod,
        "windowize_dataframe",
        lambda *a, **k: (list(range(wpp)), np.zeros((wpp, 1, 6, 20), dtype="float32")),
    )
    # 分数全部远低于阈值(-0.81) => 每窗 reject，便于累计投票计数到 50/50。
    monkeypatch.setattr(vq_mod, "score_windows", lambda *a, **k: np.full((wpp,), -5.0, dtype="float32"))
    monkeypatch.setattr(settings, "processed_data_path", str(tmp_path))
    sp = tmp_path / "z-score" / user / "scaler.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text("{}", encoding="utf-8")

    mgr = AuthSessionManager(models_root=tmp_path / "models")

    class _FS:
        async def append_raw_packet(self, *a, **k):
            return None

        async def append_result(self, *a, **k):
            return None

    mgr._inference_storage = _FS()
    fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    monkeypatch.setattr(mgr._model_cache, "get", lambda policy: fake_model)

    # 关键：policy 由 _policy_from_config 用 per-user(ema) + config(vote 30/50) 构建。
    policy = mgr._policy_from_config(_peruser_cfg(tmp_path, user=user))
    assert policy.vote_window_size == 50 and policy.vote_min_rejects == 30
    state = AuthSessionState(user_id=user, session_id=session, policy=policy)
    state.consecutive_rejects = ConsecutiveRejectTracker()
    state.vote_rejects = VoteRejectTracker()
    state.ema_rejects = EMAScoreTracker()
    state.emit_gate = ResultEmitGate()
    mgr._sessions[mgr._session_key(user, session)] = state

    parsed = {"session_id": session, "samples": []}
    payloads = []
    for _ in range(5):  # 5 包 * 10 窗 = 50 窗 => 第 5 包触发一次回包
        payloads.append(await mgr.handle_packet(user_id=user, session_id=session, parsed_batch=parsed))

    emitted = [p for p in payloads if p is not None]
    assert len(emitted) == 1                         # 50 窗 = 1 个发布周期
    msg = emitted[0].message
    assert "50" in msg and "30" in msg               # “近50窗恶意 50/50，阈值 30”
    assert emitted[0].interrupt is True              # 50/50 reject >= 30 => 打断
    assert emitted[0].result_stage == "primary"      # 单一 primary 阶段，无二次迟滞
    assert state.window_index == 50
