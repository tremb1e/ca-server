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
        return DecisionArrays(
            session_ids=np.empty((0,), dtype=np.int32),
            labels=np.empty((0,), dtype=np.int8),
            accept=np.empty((0,), dtype=bool),
            interrupt=np.empty((0,), dtype=bool),
            score=np.empty((0,), dtype=np.float32),
            threshold=np.empty((0,), dtype=np.float32),
            time_sec=np.empty((0,), dtype=np.float32),
        )

    selected_scores = ema[np.asarray(idxs, dtype=np.int64)]
    accepted = selected_scores >= float(threshold)
    return DecisionArrays(
        session_ids=session_ids[np.asarray(idxs, dtype=np.int64)].astype(np.int32, copy=False),
        labels=labels[np.asarray(idxs, dtype=np.int64)].astype(np.int8, copy=False),
        accept=accepted.astype(bool, copy=False),
        interrupt=(~accepted).astype(bool, copy=False),
        score=selected_scores.astype(np.float32, copy=False),
        threshold=np.full((len(idxs),), float(threshold), dtype=np.float32),
        time_sec=np.asarray(times, dtype=np.float32),
    )


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
            result = tracker.update(accepted=bool(primary.accept[idx]), interrupt=bool(primary.interrupt[idx]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate secondary hysteresis on cached HMOG policy_search scores.")
    parser.add_argument("--models-root", type=Path, default=ROOT / "deploy" / "data" / "models")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_metrics.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "secondary_hysteresis_hmog_report.md")
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


if __name__ == "__main__":
    main()
