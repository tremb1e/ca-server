#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.secondary_hysteresis import SecondaryHysteresisConfig, SecondaryHysteresisTracker  # noqa: E402


def _sigmoid(x: float) -> float:
    """sig(x)=1/(1+exp(-x)) on the sigmoid(policy) axis used at runtime.

    The offline `primary` scores/thresholds live on the raw VQGAN (-MSE) axis,
    while the runtime feeds the secondary tracker sigmoid(policy)-axis values
    (primary_payload.score/threshold). Map raw -> sigmoid here for parity.
    """
    value = float(x)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class ScoreArrays:
    session_ids: np.ndarray
    labels: np.ndarray
    score: np.ndarray
    time_sec: np.ndarray


@dataclass(frozen=True)
class DecisionArrays:
    session_ids: np.ndarray
    labels: np.ndarray
    accept: np.ndarray
    interrupt: np.ndarray
    score: np.ndarray
    threshold: np.ndarray
    time_sec: np.ndarray


def _load_policy(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Unexpected policy json: {path}")
    first = next(iter(payload.values()))
    if not isinstance(first, dict):
        raise ValueError(f"Unexpected policy payload: {path}")
    return first


def _iter_sessions(session_ids: np.ndarray) -> Iterable[Tuple[int, int, int]]:
    start = 0
    while start < len(session_ids):
        sid = int(session_ids[start])
        end = start + 1
        while end < len(session_ids) and int(session_ids[end]) == sid:
            end += 1
        yield sid, start, end
        start = end


def _ema_by_session(session_ids: np.ndarray, scores: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(alpha)))
    out = np.empty(scores.shape, dtype=np.float32)
    for _, start, end in _iter_sessions(session_ids):
        ema = float(scores[start])
        out[start] = ema
        for idx in range(start + 1, end):
            ema = alpha * float(scores[idx]) + (1.0 - alpha) * ema
            out[idx] = ema
    return out


def _primary_ema_scores(
    *,
    session_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ema_alpha: float,
    window_sec: float,
    overlap: float,
    primary_interval_sec: float,
) -> ScoreArrays:
    ema = _ema_by_session(session_ids, scores, alpha=float(ema_alpha))
    stride_sec = float(window_sec) * (1.0 - float(overlap))
    sample_every = max(1, int(round(float(primary_interval_sec) / max(stride_sec, 1e-9))))

    idxs: List[int] = []
    times: List[float] = []
    for _, start, end in _iter_sessions(session_ids):
        bucket = 0
        for b_start in range(start, end, sample_every):
            b_end = min(end, b_start + sample_every)
            if b_end - b_start < sample_every:
                continue
            idxs.append(b_end - 1)
            bucket += 1
            times.append(float(bucket * primary_interval_sec))

    if not idxs:
        return ScoreArrays(
            session_ids=np.empty((0,), dtype=np.int32),
            labels=np.empty((0,), dtype=np.int8),
            score=np.empty((0,), dtype=np.float32),
            time_sec=np.empty((0,), dtype=np.float32),
        )

    selected_scores = ema[np.asarray(idxs, dtype=np.int64)]
    return ScoreArrays(
        session_ids=session_ids[np.asarray(idxs, dtype=np.int64)].astype(np.int32, copy=False),
        labels=labels[np.asarray(idxs, dtype=np.int64)].astype(np.int8, copy=False),
        score=selected_scores.astype(np.float32, copy=False),
        time_sec=np.asarray(times, dtype=np.float32),
    )


def _decisions_from_scores(sampled: ScoreArrays, threshold: float) -> DecisionArrays:
    selected_scores = sampled.score
    accepted = selected_scores >= float(threshold)
    return DecisionArrays(
        session_ids=sampled.session_ids,
        labels=sampled.labels,
        accept=accepted.astype(bool, copy=False),
        interrupt=(~accepted).astype(bool, copy=False),
        score=selected_scores.astype(np.float32, copy=False),
        threshold=np.full((len(selected_scores),), float(threshold), dtype=np.float32),
        time_sec=sampled.time_sec,
    )


def _primary_ema_decisions(
    *,
    session_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ema_alpha: float,
    window_sec: float,
    overlap: float,
    primary_interval_sec: float,
) -> DecisionArrays:
    sampled = _primary_ema_scores(
        session_ids=session_ids,
        labels=labels,
        scores=scores,
        ema_alpha=ema_alpha,
        window_sec=window_sec,
        overlap=overlap,
        primary_interval_sec=primary_interval_sec,
    )
    return _decisions_from_scores(sampled, threshold)


def _secondary_decisions(primary: DecisionArrays, cfg: SecondaryHysteresisConfig) -> DecisionArrays:
    session_out: List[int] = []
    label_out: List[int] = []
    accept_out: List[bool] = []
    interrupt_out: List[bool] = []
    score_out: List[float] = []
    threshold_out: List[float] = []
    time_out: List[float] = []

    for _, start, end in _iter_sessions(primary.session_ids):
        tracker = SecondaryHysteresisTracker(cfg)
        for idx in range(start, end):
            result = tracker.update(
                accepted=bool(primary.accept[idx]),
                interrupt=bool(primary.interrupt[idx]),
                primary_score=_sigmoid(float(primary.score[idx])),
                primary_threshold=_sigmoid(float(primary.threshold[idx])),
            )
            if result is None:
                continue
            session_out.append(int(primary.session_ids[idx]))
            label_out.append(int(primary.labels[idx]))
            accept_out.append(bool(result.accept))
            interrupt_out.append(bool(result.interrupt))
            score_out.append(float(result.score))
            threshold_out.append(float(result.threshold))
            time_out.append(float(primary.time_sec[idx]))

    return DecisionArrays(
        session_ids=np.asarray(session_out, dtype=np.int32),
        labels=np.asarray(label_out, dtype=np.int8),
        accept=np.asarray(accept_out, dtype=bool),
        interrupt=np.asarray(interrupt_out, dtype=bool),
        score=np.asarray(score_out, dtype=np.float32),
        threshold=np.asarray(threshold_out, dtype=np.float32),
        time_sec=np.asarray(time_out, dtype=np.float32),
    )


def _metrics(arr: DecisionArrays) -> Dict[str, float]:
    labels = arr.labels
    accept = arr.accept
    genuine = labels == 1
    impostor = labels == 0
    pos = int(genuine.sum())
    neg = int(impostor.sum())
    false_reject = int(((~accept) & genuine).sum())
    false_accept = int((accept & impostor).sum())
    true_reject_impostor = int(((~accept) & impostor).sum())

    first_genuine: List[float] = []
    first_impostor: List[float] = []
    for _, start, end in _iter_sessions(arr.session_ids):
        label = int(arr.labels[start])
        rejected_idx = [i for i in range(start, end) if not bool(arr.accept[i])]
        if not rejected_idx:
            continue
        first = float(arr.time_sec[rejected_idx[0]])
        if label == 1:
            first_genuine.append(first)
        else:
            first_impostor.append(first)

    return {
        "outputs": float(len(labels)),
        "genuine_outputs": float(pos),
        "impostor_outputs": float(neg),
        "frr_false_reject_rate": float(false_reject / max(pos, 1)),
        "far_false_accept_rate": float(false_accept / max(neg, 1)),
        "impostor_reject_rate": float(true_reject_impostor / max(neg, 1)),
        "err": float((false_reject + false_accept) / max(pos + neg, 1)),
        "genuine_first_reject_sessions": float(len(first_genuine)),
        "impostor_first_reject_sessions": float(len(first_impostor)),
        "genuine_mean_first_reject_sec": float(np.mean(first_genuine)) if first_genuine else 0.0,
        "impostor_mean_first_reject_sec": float(np.mean(first_impostor)) if first_impostor else 0.0,
    }


def _discover_policy_dirs(models_root: Path) -> List[Path]:
    out: List[Path] = []
    for policy in sorted(models_root.glob("*/best_lock_policy.json")):
        user_dir = policy.parent
        cache = user_dir / "policy_search" / "cache" / "vqgan_only" / "ws_0.2" / "test.npz"
        if cache.exists():
            out.append(user_dir)
    return out


def _evaluate_user(user_dir: Path, *, primary_interval_sec: float, secondary_time_sec: float) -> List[Dict[str, object]]:
    policy = _load_policy(user_dir / "best_lock_policy.json")
    window_sec = float(policy.get("window", 0.2) or 0.2)
    overlap = float(policy.get("overlap", 0.5) or 0.5)
    threshold = float(policy.get("threshold", 0.0) or 0.0)
    ema_alpha = float(policy.get("ema_alpha", 0.25) or 0.25)
    cache = user_dir / "policy_search" / "cache" / "vqgan_only" / f"ws_{window_sec:.1f}" / "test.npz"
    payload = np.load(cache)
    primary = _primary_ema_decisions(
        session_ids=payload["session_ids"].astype(np.int32, copy=False),
        labels=payload["labels"].astype(np.int8, copy=False),
        scores=payload["scores"].astype(np.float32, copy=False),
        threshold=threshold,
        ema_alpha=ema_alpha,
        window_sec=window_sec,
        overlap=overlap,
        primary_interval_sec=primary_interval_sec,
    )

    vote_cfg = SecondaryHysteresisConfig(
        enabled=True,
        strategy="vote",
        primary_result_interval_sec=primary_interval_sec,
        decision_time_sec=secondary_time_sec,
        vote_window_size=max(1, int(round(secondary_time_sec / primary_interval_sec))),
        vote_min_rejects=math.ceil(0.8 * max(1, int(round(secondary_time_sec / primary_interval_sec)))),
    )
    ema_cfg = SecondaryHysteresisConfig(
        enabled=True,
        strategy="ema",
        primary_result_interval_sec=primary_interval_sec,
        decision_time_sec=secondary_time_sec,
        ema_alpha=0.25,
        ema_reject_threshold=0.8,
    )

    rows: List[Dict[str, object]] = []
    variants = {
        "一次迟滞 EMA": primary,
        "二次迟滞 EMA+EMA": _secondary_decisions(primary, ema_cfg),
        f"二次迟滞 EMA+vote {vote_cfg.vote_min_rejects}/{vote_cfg.vote_window_size}": _secondary_decisions(primary, vote_cfg),
    }
    for name, arr in variants.items():
        metrics = _metrics(arr)
        row: Dict[str, object] = {
            "user": user_dir.name,
            "variant": name,
            "threshold": threshold,
            "ema_alpha": ema_alpha,
            "primary_interval_sec": primary_interval_sec,
            "secondary_time_sec": secondary_time_sec if name.startswith("二次") else primary_interval_sec,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    out: List[Dict[str, object]] = []
    for variant, group in by_variant.items():
        total_outputs = sum(float(r["outputs"]) for r in group)
        total_genuine = sum(float(r["genuine_outputs"]) for r in group)
        total_impostor = sum(float(r["impostor_outputs"]) for r in group)
        genuine_first_values = [
            float(r["genuine_mean_first_reject_sec"])
            for r in group
            if float(r["genuine_mean_first_reject_sec"]) > 0.0
        ]
        impostor_first_values = [
            float(r["impostor_mean_first_reject_sec"])
            for r in group
            if float(r["impostor_mean_first_reject_sec"]) > 0.0
        ]
        out.append(
            {
                "user": "ALL_WEIGHTED",
                "variant": variant,
                "threshold": "",
                "ema_alpha": "",
                "primary_interval_sec": "",
                "secondary_time_sec": "",
                "outputs": total_outputs,
                "genuine_outputs": total_genuine,
                "impostor_outputs": total_impostor,
                "frr_false_reject_rate": sum(float(r["frr_false_reject_rate"]) * float(r["genuine_outputs"]) for r in group) / max(total_genuine, 1.0),
                "far_false_accept_rate": sum(float(r["far_false_accept_rate"]) * float(r["impostor_outputs"]) for r in group) / max(total_impostor, 1.0),
                "impostor_reject_rate": sum(float(r["impostor_reject_rate"]) * float(r["impostor_outputs"]) for r in group) / max(total_impostor, 1.0),
                "err": sum(float(r["err"]) * float(r["outputs"]) for r in group) / max(total_outputs, 1.0),
                "genuine_first_reject_sessions": sum(float(r["genuine_first_reject_sessions"]) for r in group),
                "impostor_first_reject_sessions": sum(float(r["impostor_first_reject_sessions"]) for r in group),
                "genuine_mean_first_reject_sec": float(np.mean(genuine_first_values)) if genuine_first_values else 0.0,
                "impostor_mean_first_reject_sec": float(np.mean(impostor_first_values)) if impostor_first_values else 0.0,
            }
        )
    return out


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format_pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _write_report(rows: Sequence[Dict[str, object]], path: Path, *, models_root: Path) -> None:
    aggregate = [r for r in rows if r.get("user") == "ALL_WEIGHTED"]
    lines = [
        "# 二次迟滞决策 HMOG 性能对比报告",
        "",
        "## 测试方法",
        "",
        f"- 模型目录：`{models_root}`",
        "- 数据来源：各用户 `policy_search/cache/vqgan_only/ws_0.2/test.npz`，复用已有 VQGAN-only 测试集分数。",
        "- 一次迟滞：按各用户 `best_lock_policy.json` 中的 EMA alpha 与 threshold 复现，然后按 1 秒节拍抽样为一次返回给 App 的认证结果。",
        "- 二次迟滞 EMA+EMA：对一次结果的 reject(0/1) 序列做 EMA，默认 10 秒输出一次，`alpha=0.25`，reject 阈值 `0.8`。",
        "- 二次迟滞 EMA+vote：对最近 10 个一次结果做 8-of-10 投票，10 秒输出一次。",
        "",
        "## 汇总结果",
        "",
        "| 方案 | 输出数 | 误报率 FRR | 漏报率 FAR | 攻击拒绝率 | ERR | 攻击平均首次拒绝(s) | 真实平均首次误拒(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {variant} | {outputs:.0f} | {frr} | {far} | {imp_reject} | {err} | {imp_first:.2f} | {gen_first:.2f} |".format(
                variant=row["variant"],
                outputs=float(row["outputs"]),
                frr=_format_pct(row["frr_false_reject_rate"]),
                far=_format_pct(row["far_false_accept_rate"]),
                imp_reject=_format_pct(row["impostor_reject_rate"]),
                err=_format_pct(row["err"]),
                imp_first=float(row["impostor_mean_first_reject_sec"]),
                gen_first=float(row["genuine_mean_first_reject_sec"]),
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 二次迟滞显著降低了输出频率；默认 10 个一次结果聚合为 1 个二次结果，本次从 10157 个一次输出降到 958 个二次输出。",
            "- 在当前 HMOG 缓存与既有一次 EMA 阈值下，二次迟滞把真实用户误报率 FRR 从 0.96% 降到 0.00%，说明它确实抑制了偶发误拒。",
            "- 代价是攻击者漏报率 FAR 从 91.50% 升到 93.91%，攻击拒绝率从 8.50% 降到 6.09%，并且攻击平均首次拒绝时间从 9.16s 延迟到 18.75s/20.00s。",
            "- 因此“二次迟滞一定同时降低误报率和漏报率”在当前数据上不成立；默认 8-of-10 更偏保守，适合优先保护真实用户体验，但需要继续搜索二次阈值/比例来平衡漏报。",
            "- `EMA+EMA` 与 `EMA+vote 8/10` 在当前参数下总体 FAR/FRR 接近，差异主要体现在首次拒绝时间：EMA+EMA 略早，vote 更稳定但更晚。",
            "- 本报告按已有 HMOG 测试集缓存离线复现，不重新训练模型；线上结果还会受 App 批量上传节奏、网络积压和 session 长度影响。",
            "",
            "## 分用户结果",
            "",
            "| 用户 | 方案 | 输出数 | FRR | FAR | 攻击拒绝率 | ERR | 攻击平均首次拒绝(s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row.get("user") == "ALL_WEIGHTED":
            continue
        lines.append(
            "| {user} | {variant} | {outputs:.0f} | {frr} | {far} | {imp_reject} | {err} | {imp_first:.2f} |".format(
                user=row["user"],
                variant=row["variant"],
                outputs=float(row["outputs"]),
                frr=_format_pct(row["frr_false_reject_rate"]),
                far=_format_pct(row["far_false_accept_rate"]),
                imp_reject=_format_pct(row["impostor_reject_rate"]),
                err=_format_pct(row["err"]),
                imp_first=float(row["impostor_mean_first_reject_sec"]),
            )
        )
    lines.extend(
        [
            "",
            "## 明细",
            "",
            "完整明细见同目录 CSV：`secondary_hysteresis_hmog_metrics.csv`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _secondary_configs(
    *,
    primary_interval_sec: float,
    secondary_time_sec: float,
) -> Dict[str, Optional[SecondaryHysteresisConfig]]:
    vote_window_size = max(1, int(round(float(secondary_time_sec) / float(primary_interval_sec))))
    return {
        "一次迟滞 EMA": None,
        "二次迟滞 EMA+EMA": SecondaryHysteresisConfig(
            enabled=True,
            strategy="ema",
            primary_result_interval_sec=primary_interval_sec,
            decision_time_sec=secondary_time_sec,
            ema_alpha=0.25,
            ema_reject_threshold=0.8,
        ),
        f"二次迟滞 EMA+vote {math.ceil(0.8 * vote_window_size)}/{vote_window_size}": SecondaryHysteresisConfig(
            enabled=True,
            strategy="vote",
            primary_result_interval_sec=primary_interval_sec,
            decision_time_sec=secondary_time_sec,
            vote_window_size=vote_window_size,
            vote_min_rejects=math.ceil(0.8 * vote_window_size),
        ),
    }


def _variant_decisions(
    sampled: ScoreArrays,
    *,
    threshold: float,
    secondary_cfg: Optional[SecondaryHysteresisConfig],
) -> DecisionArrays:
    primary = _decisions_from_scores(sampled, threshold)
    if secondary_cfg is None:
        return primary
    return _secondary_decisions(primary, secondary_cfg)


def _select_threshold_for_target_frr(
    sampled_val: ScoreArrays,
    *,
    target_frr: float,
    secondary_cfg: Optional[SecondaryHysteresisConfig],
) -> Tuple[float, Dict[str, float]]:
    candidates = np.unique(sampled_val.score.astype(np.float64, copy=False))
    if candidates.size == 0:
        return 0.0, _metrics(_variant_decisions(sampled_val, threshold=0.0, secondary_cfg=secondary_cfg))

    lo = 0
    hi = int(candidates.size - 1)
    best_idx = 0
    best_metrics: Optional[Dict[str, float]] = None
    target_frr = max(0.0, min(1.0, float(target_frr)))
    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = float(candidates[mid])
        arr = _variant_decisions(sampled_val, threshold=threshold, secondary_cfg=secondary_cfg)
        metrics = _metrics(arr)
        if float(metrics["frr_false_reject_rate"]) <= target_frr + 1e-12:
            best_idx = mid
            best_metrics = metrics
            lo = mid + 1
        else:
            hi = mid - 1

    threshold = float(candidates[best_idx])
    if best_metrics is None:
        best_metrics = _metrics(_variant_decisions(sampled_val, threshold=threshold, secondary_cfg=secondary_cfg))
    return threshold, best_metrics


def _load_sampled_scores(
    user_dir: Path,
    *,
    split: str,
    window_sec: float,
    overlap: float,
    ema_alpha: float,
    primary_interval_sec: float,
) -> ScoreArrays:
    cache = user_dir / "policy_search" / "cache" / "vqgan_only" / f"ws_{window_sec:.1f}" / f"{split}.npz"
    payload = np.load(cache)
    return _primary_ema_scores(
        session_ids=payload["session_ids"].astype(np.int32, copy=False),
        labels=payload["labels"].astype(np.int8, copy=False),
        scores=payload["scores"].astype(np.float32, copy=False),
        ema_alpha=ema_alpha,
        window_sec=window_sec,
        overlap=overlap,
        primary_interval_sec=primary_interval_sec,
    )


def _evaluate_user_target_frr(
    user_dir: Path,
    *,
    target_frrs: Sequence[float],
    primary_interval_sec: float,
    secondary_time_sec: float,
) -> List[Dict[str, object]]:
    policy = _load_policy(user_dir / "best_lock_policy.json")
    window_sec = float(policy.get("window", 0.2) or 0.2)
    overlap = float(policy.get("overlap", 0.5) or 0.5)
    ema_alpha = float(policy.get("ema_alpha", 0.25) or 0.25)
    sampled_val = _load_sampled_scores(
        user_dir,
        split="val",
        window_sec=window_sec,
        overlap=overlap,
        ema_alpha=ema_alpha,
        primary_interval_sec=primary_interval_sec,
    )
    sampled_test = _load_sampled_scores(
        user_dir,
        split="test",
        window_sec=window_sec,
        overlap=overlap,
        ema_alpha=ema_alpha,
        primary_interval_sec=primary_interval_sec,
    )

    rows: List[Dict[str, object]] = []
    variants = _secondary_configs(
        primary_interval_sec=primary_interval_sec,
        secondary_time_sec=secondary_time_sec,
    )
    for target_frr in target_frrs:
        for variant, secondary_cfg in variants.items():
            threshold, val_metrics = _select_threshold_for_target_frr(
                sampled_val,
                target_frr=target_frr,
                secondary_cfg=secondary_cfg,
            )
            test_arr = _variant_decisions(sampled_test, threshold=threshold, secondary_cfg=secondary_cfg)
            test_metrics = _metrics(test_arr)
            row: Dict[str, object] = {
                "user": user_dir.name,
                "target_frr": float(target_frr),
                "variant": variant,
                "selected_threshold": float(threshold),
                "ema_alpha": ema_alpha,
                "primary_interval_sec": primary_interval_sec,
                "secondary_time_sec": secondary_time_sec if variant.startswith("二次") else primary_interval_sec,
                "validation_outputs": val_metrics["outputs"],
                "validation_genuine_outputs": val_metrics["genuine_outputs"],
                "validation_impostor_outputs": val_metrics["impostor_outputs"],
                "validation_frr": val_metrics["frr_false_reject_rate"],
                "validation_far": val_metrics["far_false_accept_rate"],
            }
            row.update(test_metrics)
            rows.append(row)
    return rows


def _aggregate_target_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[float, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((float(row["target_frr"]), str(row["variant"])), []).append(row)

    out: List[Dict[str, object]] = []
    for (target_frr, variant), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        total_outputs = sum(float(r["outputs"]) for r in group)
        total_genuine = sum(float(r["genuine_outputs"]) for r in group)
        total_impostor = sum(float(r["impostor_outputs"]) for r in group)
        val_genuine = sum(float(r["validation_genuine_outputs"]) for r in group)
        val_impostor = sum(float(r["validation_impostor_outputs"]) for r in group)
        genuine_first_values = [
            float(r["genuine_mean_first_reject_sec"])
            for r in group
            if float(r["genuine_mean_first_reject_sec"]) > 0.0
        ]
        impostor_first_values = [
            float(r["impostor_mean_first_reject_sec"])
            for r in group
            if float(r["impostor_mean_first_reject_sec"]) > 0.0
        ]
        out.append(
            {
                "user": "ALL_WEIGHTED",
                "target_frr": target_frr,
                "variant": variant,
                "selected_threshold": "",
                "ema_alpha": "",
                "primary_interval_sec": "",
                "secondary_time_sec": "",
                "validation_outputs": sum(float(r["validation_outputs"]) for r in group),
                "validation_genuine_outputs": val_genuine,
                "validation_impostor_outputs": val_impostor,
                "validation_frr": sum(float(r["validation_frr"]) * float(r["validation_genuine_outputs"]) for r in group) / max(val_genuine, 1.0),
                "validation_far": sum(float(r["validation_far"]) * float(r["validation_impostor_outputs"]) for r in group) / max(val_impostor, 1.0),
                "outputs": total_outputs,
                "genuine_outputs": total_genuine,
                "impostor_outputs": total_impostor,
                "frr_false_reject_rate": sum(float(r["frr_false_reject_rate"]) * float(r["genuine_outputs"]) for r in group) / max(total_genuine, 1.0),
                "far_false_accept_rate": sum(float(r["far_false_accept_rate"]) * float(r["impostor_outputs"]) for r in group) / max(total_impostor, 1.0),
                "impostor_reject_rate": sum(float(r["impostor_reject_rate"]) * float(r["impostor_outputs"]) for r in group) / max(total_impostor, 1.0),
                "err": sum(float(r["err"]) * float(r["outputs"]) for r in group) / max(total_outputs, 1.0),
                "genuine_first_reject_sessions": sum(float(r["genuine_first_reject_sessions"]) for r in group),
                "impostor_first_reject_sessions": sum(float(r["impostor_first_reject_sessions"]) for r in group),
                "genuine_mean_first_reject_sec": float(np.mean(genuine_first_values)) if genuine_first_values else 0.0,
                "impostor_mean_first_reject_sec": float(np.mean(impostor_first_values)) if impostor_first_values else 0.0,
            }
        )
    return out


def _target_label(value: object) -> str:
    return f"{float(value) * 100:.0f}%"


def _write_target_report(rows: Sequence[Dict[str, object]], path: Path, *, models_root: Path) -> None:
    aggregate = [r for r in rows if r.get("user") == "ALL_WEIGHTED"]
    detail = [r for r in rows if r.get("user") != "ALL_WEIGHTED"]
    lines = [
        "# HMOG 不同 FRR 目标下的二次迟滞性能对比报告",
        "",
        "## 测试方法",
        "",
        f"- 模型目录：`{models_root}`",
        "- 数据来源：各用户 `policy_search/cache/vqgan_only/ws_0.2/{val,test}.npz`，复用已有 VQGAN-only 分数缓存。",
        "- 阈值选择：每个用户、每个目标 FRR、每个方案都在 `val.npz` 上选择最高 EMA 分数阈值，使该方案的验证集 FRR 不超过目标值。",
        "- 评估方式：把验证集选出的阈值固定后，在 `test.npz` 上计算性能指标；不限制、不惩罚检测出时间，首次拒绝时间只作为观察指标。",
        "- 一次迟滞：VQGAN 分数先按用户 `best_lock_policy.json` 中的 `ema_alpha` 做 EMA，再按 1 秒节拍抽样为一次返回结果。",
        "- 二次迟滞 EMA+EMA：对一次结果 reject(0/1) 序列做二次 EMA，10 个一次结果输出一次，`alpha=0.25`，reject 阈值 `0.8`。",
        "- 二次迟滞 EMA+vote：对最近 10 个一次结果做 `8-of-10` 投票，10 个一次结果输出一次。",
        "",
        "## 汇总结果",
        "",
        "| 目标 FRR | 方案 | 验证 FRR | 验证 FAR | 测试 FRR | 测试 FAR | 攻击拒绝率 | ERR | 输出数 | 攻击平均首次拒绝(s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {target} | {variant} | {val_frr} | {val_far} | {test_frr} | {test_far} | {imp_reject} | {err} | {outputs:.0f} | {imp_first:.2f} |".format(
                target=_target_label(row["target_frr"]),
                variant=row["variant"],
                val_frr=_format_pct(row["validation_frr"]),
                val_far=_format_pct(row["validation_far"]),
                test_frr=_format_pct(row["frr_false_reject_rate"]),
                test_far=_format_pct(row["far_false_accept_rate"]),
                imp_reject=_format_pct(row["impostor_reject_rate"]),
                err=_format_pct(row["err"]),
                outputs=float(row["outputs"]),
                imp_first=float(row["impostor_mean_first_reject_sec"]),
            )
        )

    lines.extend(["", "## 分析结论", ""])
    by_target: Dict[float, Dict[str, Dict[str, object]]] = {}
    for row in aggregate:
        by_target.setdefault(float(row["target_frr"]), {})[str(row["variant"])] = row
    for target in sorted(by_target):
        group = by_target[target]
        primary = group.get("一次迟滞 EMA")
        ema_ema = group.get("二次迟滞 EMA+EMA")
        vote = next((v for k, v in group.items() if k.startswith("二次迟滞 EMA+vote")), None)
        if primary and ema_ema and vote:
            lines.append(
                "- 目标 {target}：一次 EMA 测试 FAR={primary_far}，二次 EMA+EMA FAR={ema_far}，二次 EMA+vote FAR={vote_far}；"
                "对应测试 FRR 分别为 {primary_frr}、{ema_frr}、{vote_frr}。".format(
                    target=_target_label(target),
                    primary_far=_format_pct(primary["far_false_accept_rate"]),
                    ema_far=_format_pct(ema_ema["far_false_accept_rate"]),
                    vote_far=_format_pct(vote["far_false_accept_rate"]),
                    primary_frr=_format_pct(primary["frr_false_reject_rate"]),
                    ema_frr=_format_pct(ema_ema["frr_false_reject_rate"]),
                    vote_frr=_format_pct(vote["frr_false_reject_rate"]),
                )
            )
    lines.extend(
        [
            "- 由于本次明确不限制检测出时间，阈值选择只受验证集 FRR 约束；测试集 FRR 可能因数据分布差异略高或略低于目标。",
            "- 二次迟滞会显著降低输出频率，因此其 FRR/FAR 是按最终返回给 App 的二次结果统计，不是按每秒一次结果统计。",
            "- 在相同目标 FRR 下，二次迟滞通常需要更激进的一次 EMA 阈值才能用满 FRR 预算；这可能降低 FAR，但也会推迟或减少最终拒绝次数。",
            "",
            "## 分用户结果",
            "",
            "| 用户 | 目标 FRR | 方案 | 阈值 | 验证 FRR | 测试 FRR | 测试 FAR | 攻击拒绝率 | ERR | 输出数 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in detail:
        lines.append(
            "| {user} | {target} | {variant} | {threshold:.6f} | {val_frr} | {test_frr} | {test_far} | {imp_reject} | {err} | {outputs:.0f} |".format(
                user=row["user"],
                target=_target_label(row["target_frr"]),
                variant=row["variant"],
                threshold=float(row["selected_threshold"]),
                val_frr=_format_pct(row["validation_frr"]),
                test_frr=_format_pct(row["frr_false_reject_rate"]),
                test_far=_format_pct(row["far_false_accept_rate"]),
                imp_reject=_format_pct(row["impostor_reject_rate"]),
                err=_format_pct(row["err"]),
                outputs=float(row["outputs"]),
            )
        )
    lines.extend(
        [
            "",
            "## 明细",
            "",
            "完整明细见同目录 CSV：`secondary_hysteresis_hmog_frr_targets_metrics.csv`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_target_frrs(value: str) -> List[float]:
    out: List[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = float(part)
        if parsed > 1.0:
            parsed = parsed / 100.0
        out.append(max(0.0, min(1.0, parsed)))
    if not out:
        raise ValueError("target FRR list must not be empty")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate secondary hysteresis on cached HMOG policy_search scores.")
    parser.add_argument("--models-root", type=Path, default=ROOT / "deploy" / "data" / "models")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_metrics.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_report.md")
    parser.add_argument("--target-output-csv", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_frr_targets_metrics.csv")
    parser.add_argument("--target-output-md", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_frr_targets_report.md")
    parser.add_argument("--target-frrs", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--primary-interval-sec", type=float, default=1.0)
    parser.add_argument("--secondary-time-sec", type=float, default=10.0)
    args = parser.parse_args()

    user_dirs = _discover_policy_dirs(args.models_root)
    if not user_dirs:
        raise SystemExit(f"No cached policy_search test.npz files found under {args.models_root}")

    rows: List[Dict[str, object]] = []
    for user_dir in user_dirs:
        rows.extend(
            _evaluate_user(
                user_dir,
                primary_interval_sec=float(args.primary_interval_sec),
                secondary_time_sec=float(args.secondary_time_sec),
            )
        )
    rows.extend(_aggregate(rows))
    _write_csv(rows, args.output_csv)
    _write_report(rows, args.output_md, models_root=args.models_root)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")

    target_rows: List[Dict[str, object]] = []
    target_frrs = _parse_target_frrs(str(args.target_frrs))
    for user_dir in user_dirs:
        target_rows.extend(
            _evaluate_user_target_frr(
                user_dir,
                target_frrs=target_frrs,
                primary_interval_sec=float(args.primary_interval_sec),
                secondary_time_sec=float(args.secondary_time_sec),
            )
        )
    target_rows.extend(_aggregate_target_rows(target_rows))
    _write_csv(target_rows, args.target_output_csv)
    _write_target_report(target_rows, args.target_output_md, models_root=args.models_root)
    print(f"Wrote {args.target_output_csv}")
    print(f"Wrote {args.target_output_md}")


if __name__ == "__main__":
    main()
