from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..ca_config import CAConfig, get_ca_config
from ..utils.ca_train import ensure_ca_train_on_path
from ..utils.policy_paths import serialize_policy_path
from ..utils.runtime import app_root
from .metrics import first_interrupt_times_sec_per_session, p_first_interrupt_le_columns
from .pareto import ParetoPoint, pareto_frontier


logger = logging.getLogger(__name__)

AUTH_METHOD_VQGAN_ONLY = "vqgan-only"
AUTH_METHOD_VQGAN_TRANSFORMER = "vqgan+transformer"

# Prompt Step 12: explicit (N, M) sweep ranges (inclusive).
# - N in [7, 20]
# - For each N, M in [lo, hi]
PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N: Dict[int, Tuple[int, int]] = {
    7: (4, 6),
    8: (5, 6),
    9: (5, 7),
    10: (6, 8),
    11: (6, 8),
    12: (7, 9),
    13: (7, 10),
    14: (8, 10),
    15: (8, 11),
    16: (9, 13),
    17: (9, 13),
    18: (10, 14),
    19: (10, 15),
    20: (11, 16),
}


def _normalize_auth_method(value: str) -> str:
    value = str(value or "").strip().lower()
    aliases = {
        "vqgan": AUTH_METHOD_VQGAN_ONLY,
        "vqgan_only": AUTH_METHOD_VQGAN_ONLY,
        "vqgan-only": AUTH_METHOD_VQGAN_ONLY,
        "vqganonly": AUTH_METHOD_VQGAN_ONLY,
        "transformer": AUTH_METHOD_VQGAN_TRANSFORMER,
        "lm": AUTH_METHOD_VQGAN_TRANSFORMER,
        "vqgan_lm": AUTH_METHOD_VQGAN_TRANSFORMER,
        "vqgan+transformer": AUTH_METHOD_VQGAN_TRANSFORMER,
        "vqgan_transformer": AUTH_METHOD_VQGAN_TRANSFORMER,
        "vqgan+lm": AUTH_METHOD_VQGAN_TRANSFORMER,
    }
    normalized = aliases.get(value, value)
    if normalized not in {AUTH_METHOD_VQGAN_ONLY, AUTH_METHOD_VQGAN_TRANSFORMER}:
        raise ValueError(f"Unsupported auth_method={value!r}; expected vqgan-only or vqgan+transformer.")
    return normalized


def _auth_method_tag(auth_method: str) -> str:
    auth_method = _normalize_auth_method(auth_method)
    if auth_method == AUTH_METHOD_VQGAN_ONLY:
        return "vqgan_only"
    return "vqgan_transformer"


def _server_root() -> Path:
    return app_root()


def _ensure_ca_train_on_path() -> Path:
    return ensure_ca_train_on_path()


def _default_dataset_root(server_root: Path) -> Path:
    return server_root / "data_storage" / "processed_data" / "window"


def _default_models_root(server_root: Path) -> Path:
    return server_root / "data_storage" / "models"


def _ckpt_paths(models_root: Path, user: str, window_size: float) -> Tuple[Path, Path]:
    ckpt_dir = models_root / user / "checkpoints"
    return (
        ckpt_dir / f"vqgan_user_{user}_ws_{window_size:.1f}.pt",
        ckpt_dir / f"token_gpt_user_{user}_ws_{window_size:.1f}.pt",
    )


def _file_fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "path": str(path),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "size": int(st.st_size),
    }


@dataclass(frozen=True)
class ScoreArrays:
    session_ids: np.ndarray  # int32
    labels: np.ndarray  # int8
    scores: np.ndarray  # float32


def _load_or_score_split(
    *,
    user: str,
    split: str,
    auth_method: str,
    window_size: float,
    overlap: float,
    target_width: int,
    csv_path: Path,
    vqgan_ckpt: Path,
    lm_ckpt: Path,
    device: str,
    token_batch_size: int,
    use_amp: bool,
    cache_dir: Path,
) -> ScoreArrays:
    auth_method = _normalize_auth_method(auth_method)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{split}.npz"
    meta_path = cache_dir / f"{split}.meta.json"

    expected = {
        "user": str(user),
        "split": str(split),
        "auth_method": str(auth_method),
        "window_size": float(window_size),
        "overlap": float(overlap),
        "target_width": int(target_width),
        "csv": _file_fingerprint(csv_path),
        "vqgan_ckpt": _file_fingerprint(vqgan_ckpt),
    }
    if auth_method == AUTH_METHOD_VQGAN_TRANSFORMER:
        expected["lm_ckpt"] = _file_fingerprint(lm_ckpt)

    if npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if meta == expected:
            payload = np.load(npz_path)
            return ScoreArrays(
                session_ids=payload["session_ids"].astype(np.int32, copy=False),
                labels=payload["labels"].astype(np.int8, copy=False),
                scores=payload["scores"].astype(np.float32, copy=False),
            )

    arrays = _score_split_inprocess(
        csv_path=csv_path,
        target_user=str(user),
        auth_method=str(auth_method),
        window_size=float(window_size),
        target_width=int(target_width),
        vqgan_ckpt=vqgan_ckpt,
        lm_ckpt=lm_ckpt,
        device=str(device),
        token_batch_size=int(token_batch_size),
        use_amp=bool(use_amp),
    )

    np.savez_compressed(
        npz_path,
        session_ids=arrays.session_ids,
        labels=arrays.labels,
        scores=arrays.scores,
    )
    meta_path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
    return arrays


def _score_split_inprocess(
    *,
    csv_path: Path,
    target_user: str,
    auth_method: str,
    window_size: float,
    target_width: int,
    vqgan_ckpt: Path,
    lm_ckpt: Path,
    device: str,
    token_batch_size: int,
    use_amp: bool,
) -> ScoreArrays:
    auth_method = _normalize_auth_method(auth_method)
    if auth_method == AUTH_METHOD_VQGAN_ONLY:
        return _score_split_vqgan_only_inprocess(
            csv_path=csv_path,
            target_user=target_user,
            window_size=window_size,
            target_width=target_width,
            vqgan_ckpt=vqgan_ckpt,
            device=device,
            batch_size=int(token_batch_size),
            use_amp=bool(use_amp),
        )

    return _score_split_vqgan_transformer_inprocess(
        csv_path=csv_path,
        target_user=target_user,
        window_size=window_size,
        target_width=target_width,
        vqgan_ckpt=vqgan_ckpt,
        lm_ckpt=lm_ckpt,
        device=device,
        token_batch_size=int(token_batch_size),
        use_amp=bool(use_amp),
    )


def _score_split_vqgan_transformer_inprocess(
    *,
    csv_path: Path,
    target_user: str,
    window_size: float,
    target_width: int,
    vqgan_ckpt: Path,
    lm_ckpt: Path,
    device: str,
    token_batch_size: int,
    use_amp: bool,
) -> ScoreArrays:
    _ensure_ca_train_on_path()

    import torch  # local import to keep policy_search import light

    from ..utils.accelerator import autocast_context, resolve_torch_device
    from hmog_token_auth_inference import load_lm, load_vqgan
    from hmog_data import iter_windows_from_csv_unlabeled_with_session
    from hmog_tokenizer import encode_windows_to_tokens

    torch_device = resolve_torch_device(device)
    vqgan = load_vqgan(Path(vqgan_ckpt), device=torch_device, cfg_path=None)
    lm = load_lm(Path(lm_ckpt), device=torch_device, cfg_path=None)

    batch_windows: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_session_ids: List[int] = []

    scores_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    sessions_chunks: List[np.ndarray] = []

    cur_key: Optional[Tuple[str, str]] = None
    session_id = 0
    target_user_norm = str(target_user).strip()

    def _flush() -> None:
        if not batch_windows:
            return
        windows_np = np.stack(batch_windows, axis=0).astype(np.float32, copy=False)
        tok = encode_windows_to_tokens(
            vqgan,
            windows_np,
            batch_size=int(token_batch_size),
            device=torch_device,
            use_amp=bool(use_amp),
        )
        tokens = torch.from_numpy(tok.tokens).to(device=torch_device, dtype=torch.long, non_blocking=True)
        with autocast_context(torch_device, enabled=bool(use_amp)):
            scores = lm.score(tokens).detach().cpu().numpy().astype(np.float32, copy=False)
        scores_chunks.append(scores)
        labels_chunks.append(np.asarray(batch_labels, dtype=np.int8))
        sessions_chunks.append(np.asarray(batch_session_ids, dtype=np.int32))
        batch_windows.clear()
        batch_labels.clear()
        batch_session_ids.clear()

    for _, subject, session, window in iter_windows_from_csv_unlabeled_with_session(
        Path(csv_path),
        window_size_sec=float(window_size),
        target_width=int(target_width),
    ):
        subject_s = str(subject).strip()
        session_s = str(session).strip()
        key = (subject_s, session_s)
        if cur_key is None:
            cur_key = key
            session_id = 0
        elif key != cur_key:
            session_id += 1
            cur_key = key
        label = 1 if subject_s == target_user_norm else 0
        batch_windows.append(window)
        batch_labels.append(int(label))
        batch_session_ids.append(int(session_id))
        if len(batch_windows) >= int(token_batch_size):
            _flush()

    _flush()
    if not scores_chunks:
        raise ValueError(f"No valid windows scored from {csv_path}")

    scores_all = np.concatenate(scores_chunks, axis=0)
    labels_all = np.concatenate(labels_chunks, axis=0)
    sessions_all = np.concatenate(sessions_chunks, axis=0)
    return ScoreArrays(session_ids=sessions_all, labels=labels_all, scores=scores_all)


def _score_split_vqgan_only_inprocess(
    *,
    csv_path: Path,
    target_user: str,
    window_size: float,
    target_width: int,
    vqgan_ckpt: Path,
    device: str,
    batch_size: int,
    use_amp: bool,
) -> ScoreArrays:
    _ensure_ca_train_on_path()

    import torch  # local import to keep policy_search import light

    from ..utils.accelerator import autocast_context, resolve_torch_device
    from hmog_token_auth_inference import load_vqgan
    from hmog_data import iter_windows_from_csv_unlabeled_with_session

    torch_device = resolve_torch_device(device)
    vqgan = load_vqgan(Path(vqgan_ckpt), device=torch_device, cfg_path=None)

    batch_windows: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_session_ids: List[int] = []

    scores_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    sessions_chunks: List[np.ndarray] = []

    cur_key: Optional[Tuple[str, str]] = None
    session_id = 0
    target_user_norm = str(target_user).strip()

    def _flush() -> None:
        if not batch_windows:
            return
        windows_np = np.stack(batch_windows, axis=0).astype(np.float32, copy=False)
        batch = torch.from_numpy(windows_np).to(device=torch_device, dtype=torch.float32, non_blocking=True)
        with autocast_context(torch_device, enabled=bool(use_amp)):
            decoded, _, _ = vqgan(batch)
            errors = torch.mean((batch - decoded) ** 2, dim=(1, 2, 3))
        scores = (-errors).detach().cpu().numpy().astype(np.float32, copy=False)
        scores_chunks.append(scores)
        labels_chunks.append(np.asarray(batch_labels, dtype=np.int8))
        sessions_chunks.append(np.asarray(batch_session_ids, dtype=np.int32))
        batch_windows.clear()
        batch_labels.clear()
        batch_session_ids.clear()

    for _, subject, session, window in iter_windows_from_csv_unlabeled_with_session(
        Path(csv_path),
        window_size_sec=float(window_size),
        target_width=int(target_width),
    ):
        subject_s = str(subject).strip()
        session_s = str(session).strip()
        key = (subject_s, session_s)
        if cur_key is None:
            cur_key = key
            session_id = 0
        elif key != cur_key:
            session_id += 1
            cur_key = key
        label = 1 if subject_s == target_user_norm else 0
        batch_windows.append(window)
        batch_labels.append(int(label))
        batch_session_ids.append(int(session_id))
        if len(batch_windows) >= int(batch_size):
            _flush()

    _flush()
    if not scores_chunks:
        raise ValueError(f"No valid windows scored from {csv_path}")

    scores_all = np.concatenate(scores_chunks, axis=0)
    labels_all = np.concatenate(labels_chunks, axis=0)
    sessions_all = np.concatenate(sessions_chunks, axis=0)
    return ScoreArrays(session_ids=sessions_all, labels=labels_all, scores=scores_all)


def _resolve_target_width(*checkpoint_paths: Path, default: int = 50) -> int:
    for ckpt in checkpoint_paths:
        cfg_path = Path(ckpt).with_suffix(".json")
        if not cfg_path.exists():
            continue
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("target_width", payload.get("input_width", default))
        try:
            return int(value)
        except Exception:
            continue
    return int(default)


@dataclass(frozen=True)
class GridSearchConfig:
    window_sizes: Sequence[float]
    overlap: float
    vote_window_sizes: Sequence[int]
    target_window_frr_candidates: Sequence[float]
    max_decision_time_sec: float
    decision_strategies: Sequence[str] = ("ema", "vote")
    ema_alpha_candidates: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50)
    # If set, use the prompt-specified M ranges for each N. When None (or when
    # N is missing from the map), fall back to `vote_min_rejects_ratio_range`.
    vote_min_rejects_range_by_n: Optional[Dict[int, Tuple[int, int]]] = field(
        default_factory=lambda: dict(PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N)
    )
    vote_min_rejects_ratio_range: Optional[Tuple[float, float]] = None
    token_batch_size: int = 512
    use_amp: bool = True
    # Additional reporting: compute p(first_interrupt <= T) for multiple T (sec).
    # When None/empty, defaults to [max_decision_time_sec].
    decision_time_thresholds_sec: Optional[Sequence[float]] = None
    # Prompt Step 12: write one file per (auth_method, t, N, M).
    write_per_combo: bool = True


def _resolve_vote_min_rejects_range(cfg: GridSearchConfig, n: int) -> Tuple[int, int]:
    n_i = int(n)
    if n_i <= 0:
        return 1, 1

    if cfg.vote_min_rejects_range_by_n:
        rng = cfg.vote_min_rejects_range_by_n.get(n_i)
        if rng is not None:
            lo, hi = rng
            lo_i = max(1, int(lo))
            hi_i = min(n_i, int(hi))
            if hi_i < lo_i:
                hi_i = lo_i
            return lo_i, hi_i

    if cfg.vote_min_rejects_ratio_range is None:
        r_min, r_max = (0.5, 0.7)
    else:
        r_min, r_max = cfg.vote_min_rejects_ratio_range
    m_lo = max(1, int(np.ceil(float(r_min) * n_i)))
    m_hi = min(n_i, int(np.floor(float(r_max) * n_i)))
    if m_hi < m_lo:
        # Ensure at least one candidate
        m_lo = max(1, min(n_i, int(round(float(r_min) * n_i))))
        m_hi = m_lo
    return m_lo, m_hi


def _normalize_decision_strategy(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in {"ema", "ewma"}:
        return "ema"
    if value in {"vote", "yofx", "y-of-x"}:
        return "vote"
    if value in {"k", "consecutive"}:
        return "k"
    return value


def _ema_scores_by_session(
    session_ids: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    session_ids_arr = np.asarray(session_ids)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if session_ids_arr.shape != scores_arr.shape:
        raise ValueError("session_ids/scores must have the same shape")

    alpha_f = min(1.0, max(0.0, float(alpha)))
    out = np.empty(scores_arr.shape, dtype=np.float64)
    cur_sid: Optional[int] = None
    ema = 0.0
    for i, (sid, score) in enumerate(zip(session_ids_arr.tolist(), scores_arr.tolist())):
        sid_i = int(sid)
        score_f = float(score)
        if cur_sid is None or sid_i != int(cur_sid):
            cur_sid = sid_i
            ema = score_f
        else:
            ema = alpha_f * score_f + (1.0 - alpha_f) * ema
        out[i] = float(ema)
    return out.astype(np.float32, copy=False)


def _ema_first_interrupt_times_sec(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    ema_scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    window_size_sec: float,
    overlap: float,
    target_label: int,
) -> List[Optional[float]]:
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(ema_scores, dtype=np.float64)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    stride_sec = float(window_size_sec) * (1.0 - float(overlap))
    out: List[Optional[float]] = []
    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_first: Optional[int] = None
    cur_index = 0

    def _flush() -> None:
        nonlocal cur_sid, cur_label, cur_first, cur_index
        if cur_sid is None or cur_label is None:
            return
        if int(cur_label) == int(target_label):
            if cur_first is None:
                out.append(None)
            else:
                out.append(float(window_size_sec + (int(cur_first) - 1) * stride_sec))
        cur_sid = None
        cur_label = None
        cur_first = None
        cur_index = 0

    for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        label_i = int(label)
        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
            cur_index = 0
        elif sid_i != int(cur_sid) or (cur_label is not None and label_i != int(cur_label)):
            _flush()
            cur_sid = sid_i
            cur_label = label_i
            cur_index = 0
        cur_index += 1
        if cur_first is None and float(score) < float(threshold):
            cur_first = int(cur_index)

    _flush()
    return out


def _compute_ema_window_metrics_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    ema_scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(ema_scores, dtype=np.float64)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    pred_genuine = scores_arr >= float(threshold)
    genuine = labels_arr == 1
    impostor = labels_arr == 0
    pos = int(genuine.sum())
    neg = int(impostor.sum())
    tp = int((pred_genuine & genuine).sum())
    fn = int(((~pred_genuine) & genuine).sum())
    fp = int((pred_genuine & impostor).sum())
    tn = int(((~pred_genuine) & impostor).sum())

    genuine_sessions = impostor_sessions = 0
    genuine_first_sum = impostor_first_sum = 0
    genuine_first_n = impostor_first_n = 0
    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_first: Optional[int] = None
    cur_index = 0

    def _flush() -> None:
        nonlocal genuine_sessions, impostor_sessions
        nonlocal genuine_first_sum, impostor_first_sum, genuine_first_n, impostor_first_n
        nonlocal cur_sid, cur_label, cur_first, cur_index
        if cur_sid is None or cur_label is None:
            return
        if int(cur_label) == 1:
            genuine_sessions += 1
            if cur_first is not None:
                genuine_first_sum += int(cur_first)
                genuine_first_n += 1
        else:
            impostor_sessions += 1
            if cur_first is not None:
                impostor_first_sum += int(cur_first)
                impostor_first_n += 1
        cur_sid = None
        cur_label = None
        cur_first = None
        cur_index = 0

    for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        label_i = int(label)
        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
            cur_index = 0
        elif sid_i != int(cur_sid) or (cur_label is not None and label_i != int(cur_label)):
            _flush()
            cur_sid = sid_i
            cur_label = label_i
            cur_index = 0
        cur_index += 1
        if cur_first is None and float(score) < float(threshold):
            cur_first = int(cur_index)
    _flush()

    return {
        "threshold": float(threshold),
        "far": float(fp / max(neg, 1)),
        "frr": float(fn / max(pos, 1)),
        "err": float((fp + fn) / max(pos + neg, 1)),
        "pos": float(pos),
        "neg": float(neg),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_first_interrupt_n": float(genuine_first_n),
        "impostor_first_interrupt_n": float(impostor_first_n),
        "genuine_mean_first_interrupt_window": float(genuine_first_sum / genuine_first_n) if genuine_first_n else 0.0,
        "impostor_mean_first_interrupt_window": float(impostor_first_sum / impostor_first_n) if impostor_first_n else 0.0,
    }


def _select_threshold_by_ema_window_frr_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    ema_scores: Sequence[float] | np.ndarray,
    *,
    target_window_frr: float,
) -> Tuple[float, Dict[str, float]]:
    scores_arr = np.asarray(ema_scores, dtype=np.float64)
    candidates = np.unique(scores_arr)
    if candidates.size == 0:
        raise ValueError("No score candidates found for threshold selection")
    candidates.sort()

    target_window_frr = min(1.0, max(0.0, float(target_window_frr)))
    best_thr = float(candidates[0])
    best_metrics = _compute_ema_window_metrics_from_arrays(
        session_ids,
        labels,
        ema_scores,
        threshold=best_thr,
    )

    lo = 0
    hi = int(candidates.size - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        thr = float(candidates[mid])
        metrics = _compute_ema_window_metrics_from_arrays(session_ids, labels, ema_scores, threshold=thr)
        if float(metrics.get("frr", 0.0)) <= target_window_frr:
            best_thr = thr
            best_metrics = metrics
            lo = mid + 1
        else:
            hi = mid - 1
    return float(best_thr), best_metrics


def run_policy_grid_search(
    user: str,
    *,
    device: str = "auto",
    auth_method: str = AUTH_METHOD_VQGAN_ONLY,
    cfg: Optional[GridSearchConfig] = None,
    ca_cfg: Optional[CAConfig] = None,
    dataset_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
    write_best_policy: bool = True,
) -> Dict[str, Path]:
    """
    Run offline policy grid search for a trained user and write tables under models/<user>/policy_search/.

    Returns a dict of output paths.
    """
    ca_cfg = ca_cfg or get_ca_config()
    from ..utils.accelerator import normalize_device

    server_root = _server_root()
    dataset_root = Path(dataset_root) if dataset_root is not None else _default_dataset_root(server_root)
    models_root = Path(models_root) if models_root is not None else _default_models_root(server_root)
    resolved_device = normalize_device(device)

    if cfg is None:
        preferred_strategy = _normalize_decision_strategy(str(ca_cfg.auth.decision_strategy))
        decision_strategies = [preferred_strategy]
        for strategy in ("ema", "vote"):
            if strategy not in decision_strategies:
                decision_strategies.append(strategy)
        cfg = GridSearchConfig(
            window_sizes=list(ca_cfg.windows.sizes),
            overlap=float(ca_cfg.windows.overlap),
            decision_strategies=decision_strategies,
            ema_alpha_candidates=list(ca_cfg.auth.ema_alpha_candidates or [ca_cfg.auth.ema_alpha]),
            vote_window_sizes=list(range(7, 21)),
            target_window_frr_candidates=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, float(ca_cfg.auth.target_window_frr)],
            max_decision_time_sec=float(ca_cfg.auth.max_decision_time_sec),
            token_batch_size=512,
            use_amp=True,
            write_per_combo=True,
        )

    out_dir = models_root / user / "policy_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    auth_method = _normalize_auth_method(auth_method)
    method_tag = _auth_method_tag(auth_method)
    best_json = models_root / user / "best_lock_policy.json"
    if auth_method == AUTH_METHOD_VQGAN_TRANSFORMER:
        grid_csv = out_dir / "grid_results.csv"
        pareto_csv = out_dir / "pareto_frontier.csv"
    else:
        grid_csv = out_dir / "grid_results_vqgan_only.csv"
        pareto_csv = out_dir / "pareto_frontier_vqgan_only.csv"

    _ensure_ca_train_on_path()
    from hmog_consecutive_rejects import (  # type: ignore[import-not-found]
        compute_vote_reject_window_metrics_from_arrays,
        select_threshold_by_vote_window_frr_from_arrays,
    )

    rows: List[Dict[str, object]] = []
    genuine_stats_by_ws: Dict[float, dict] = {}

    max_t = float(cfg.max_decision_time_sec)
    overlap = float(cfg.overlap)

    for ws in cfg.window_sizes:
        ws_f = float(ws)
        vqgan_ckpt, lm_ckpt = _ckpt_paths(models_root, user, ws_f)
        if not vqgan_ckpt.exists():
            raise FileNotFoundError(f"Missing VQGAN checkpoint for user={user} ws={ws_f:.1f}: {vqgan_ckpt}")
        if auth_method == AUTH_METHOD_VQGAN_TRANSFORMER and not lm_ckpt.exists():
            raise FileNotFoundError(f"Missing transformer checkpoint for user={user} ws={ws_f:.1f}: {lm_ckpt}")

        target_width = _resolve_target_width(lm_ckpt, vqgan_ckpt, default=50)

        ws_tag = f"{ws_f:.1f}"
        user_dir = dataset_root / ws_tag / user
        val_csv_path = user_dir / "val.csv"
        test_csv_path = user_dir / "test.csv"
        if not val_csv_path.exists() or not test_csv_path.exists():
            raise FileNotFoundError(f"Missing window CSVs for user={user} ws={ws_tag}: {val_csv_path} / {test_csv_path}")

        cache_dir = out_dir / "cache" / method_tag / f"ws_{ws_tag}"
        val_arrays = _load_or_score_split(
            user=user,
            split="val",
            auth_method=auth_method,
            window_size=ws_f,
            overlap=overlap,
            target_width=target_width,
            csv_path=val_csv_path,
            vqgan_ckpt=vqgan_ckpt,
            lm_ckpt=lm_ckpt,
            device=resolved_device,
            token_batch_size=int(cfg.token_batch_size),
            use_amp=bool(cfg.use_amp),
            cache_dir=cache_dir,
        )
        test_arrays = _load_or_score_split(
            user=user,
            split="test",
            auth_method=auth_method,
            window_size=ws_f,
            overlap=overlap,
            target_width=target_width,
            csv_path=test_csv_path,
            vqgan_ckpt=vqgan_ckpt,
            lm_ckpt=lm_ckpt,
            device=resolved_device,
            token_batch_size=int(cfg.token_batch_size),
            use_amp=bool(cfg.use_amp),
            cache_dir=cache_dir,
        )

        # 落盘真实用户(genuine)分数统计，供认证启动时做阈值越界校验。
        stats = _genuine_score_stats(val_arrays.scores, val_arrays.labels)
        if stats is not None:
            genuine_stats_by_ws[float(ws_f)] = stats

        stride_sec = float(ws_f) * (1.0 - float(overlap))
        strategies = {_normalize_decision_strategy(s) for s in (getattr(cfg, "decision_strategies", []) or ["ema", "vote"])}

        if "ema" in strategies:
            for alpha in getattr(cfg, "ema_alpha_candidates", []) or [float(ca_cfg.auth.ema_alpha)]:
                alpha_f = float(alpha)
                val_ema_scores = _ema_scores_by_session(val_arrays.session_ids, val_arrays.scores, alpha=alpha_f)
                test_ema_scores = _ema_scores_by_session(test_arrays.session_ids, test_arrays.scores, alpha=alpha_f)

                for tw_frr in cfg.target_window_frr_candidates:
                    tw_frr_f = float(tw_frr)
                    threshold, val_metrics = _select_threshold_by_ema_window_frr_from_arrays(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_ema_scores,
                        target_window_frr=tw_frr_f,
                    )
                    test_metrics = _compute_ema_window_metrics_from_arrays(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_ema_scores,
                        threshold=float(threshold),
                    )

                    def _mean_window_to_sec(mean_window: float) -> float:
                        mean_window = float(mean_window)
                        if mean_window <= 0.0:
                            return 0.0
                        return float(ws_f + (mean_window - 1.0) * stride_sec)

                    val_impostor_mean_sec = _mean_window_to_sec(float(val_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0))
                    val_genuine_mean_sec = _mean_window_to_sec(float(val_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0))
                    test_impostor_mean_sec = _mean_window_to_sec(float(test_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0))
                    test_genuine_mean_sec = _mean_window_to_sec(float(test_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0))

                    thresholds_sec = list(getattr(cfg, "decision_time_thresholds_sec", []) or [])
                    if not thresholds_sec:
                        thresholds_sec = [1.0]
                        if max_t > 0.0:
                            upper = min(10, int(math.floor(max_t)))
                            for sec in range(2, upper + 1):
                                thresholds_sec.append(float(sec))
                            thresholds_sec.append(float(max_t))
                    thresholds_sec = sorted({float(x) for x in thresholds_sec if float(x) > 0.0})
                    if 1.0 not in thresholds_sec:
                        thresholds_sec = sorted({1.0, *thresholds_sec})
                    if max_t > 0.0 and all(abs(float(x) - float(max_t)) > 1e-9 for x in thresholds_sec):
                        thresholds_sec = sorted({*thresholds_sec, float(max_t)})

                    val_imp_times = _ema_first_interrupt_times_sec(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_ema_scores,
                        threshold=float(threshold),
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=0,
                    )
                    val_gen_times = _ema_first_interrupt_times_sec(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_ema_scores,
                        threshold=float(threshold),
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=1,
                    )
                    test_imp_times = _ema_first_interrupt_times_sec(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_ema_scores,
                        threshold=float(threshold),
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=0,
                    )
                    test_gen_times = _ema_first_interrupt_times_sec(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_ema_scores,
                        threshold=float(threshold),
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=1,
                    )

                    row: Dict[str, object] = {
                        "user": str(user),
                        "auth_method": str(auth_method),
                        "decision_strategy": "ema",
                        "interrupt_rule": "ema",
                        "window_size_sec": float(ws_f),
                        "overlap": float(overlap),
                        "stride_sec": float(stride_sec),
                        "target_width": int(target_width),
                        "ema_alpha": float(alpha_f),
                        "vote_window_size": 0,
                        "vote_min_rejects": 0,
                        "target_window_frr": float(tw_frr_f),
                        "threshold": float(threshold),
                        "interrupt_time_sec_min": float(ws_f),
                        "feasible_within_max_decision_time": int(max_t <= 0.0 or ws_f <= max_t + 1e-9),
                        "vqgan_checkpoint": str(vqgan_ckpt),
                        "lm_checkpoint": str(lm_ckpt) if lm_ckpt.exists() else "",
                        "val_far": float(val_metrics.get("far", 0.0) or 0.0),
                        "val_frr": float(val_metrics.get("frr", 0.0) or 0.0),
                        "val_err": float(val_metrics.get("err", 0.0) or 0.0),
                        "val_impostor_mean_first_interrupt_sec": float(val_impostor_mean_sec),
                        "val_genuine_mean_first_interrupt_sec": float(val_genuine_mean_sec),
                        "test_far": float(test_metrics.get("far", 0.0) or 0.0),
                        "test_frr": float(test_metrics.get("frr", 0.0) or 0.0),
                        "test_err": float(test_metrics.get("err", 0.0) or 0.0),
                        "test_impostor_mean_first_interrupt_sec": float(test_impostor_mean_sec),
                        "test_genuine_mean_first_interrupt_sec": float(test_genuine_mean_sec),
                        "val_impostor_sessions": float(val_metrics.get("impostor_sessions", 0.0) or 0.0),
                        "val_impostor_interrupted_sessions": float(val_metrics.get("impostor_first_interrupt_n", 0.0) or 0.0),
                        "val_genuine_sessions": float(val_metrics.get("genuine_sessions", 0.0) or 0.0),
                        "val_genuine_interrupted_sessions": float(val_metrics.get("genuine_first_interrupt_n", 0.0) or 0.0),
                        "test_impostor_sessions": float(test_metrics.get("impostor_sessions", 0.0) or 0.0),
                        "test_impostor_interrupted_sessions": float(test_metrics.get("impostor_first_interrupt_n", 0.0) or 0.0),
                        "test_genuine_sessions": float(test_metrics.get("genuine_sessions", 0.0) or 0.0),
                        "test_genuine_interrupted_sessions": float(test_metrics.get("genuine_first_interrupt_n", 0.0) or 0.0),
                        "val_impostor_mean_first_interrupt_window": float(val_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0),
                        "val_genuine_mean_first_interrupt_window": float(val_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0),
                        "test_impostor_mean_first_interrupt_window": float(test_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0),
                        "test_genuine_mean_first_interrupt_window": float(test_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0),
                    }
                    row.update(p_first_interrupt_le_columns("val_impostor", val_imp_times, thresholds_sec))
                    row.update(p_first_interrupt_le_columns("val_genuine", val_gen_times, thresholds_sec))
                    row.update(p_first_interrupt_le_columns("test_impostor", test_imp_times, thresholds_sec))
                    row.update(p_first_interrupt_le_columns("test_genuine", test_gen_times, thresholds_sec))
                    rows.append(row)

        for n in cfg.vote_window_sizes if "vote" in strategies else []:
            n_i = int(n)
            if n_i <= 0:
                continue

            # Theoretical earliest decision time under this vote rule.
            interrupt_time_sec = float(ws_f + (n_i - 1) * stride_sec)
            feasible_1s = bool(max_t > 0.0 and interrupt_time_sec <= max_t + 1e-9)

            m_lo, m_hi = _resolve_vote_min_rejects_range(cfg, n_i)

            for m in range(m_lo, m_hi + 1):
                m_i = int(m)
                if m_i <= 0 or m_i > n_i:
                    continue
                for tw_frr in cfg.target_window_frr_candidates:
                    tw_frr_f = float(tw_frr)

                    threshold, val_metrics = select_threshold_by_vote_window_frr_from_arrays(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_arrays.scores,
                        window_size=n_i,
                        min_rejects=m_i,
                        target_window_frr=tw_frr_f,
                    )
                    test_metrics = compute_vote_reject_window_metrics_from_arrays(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_arrays.scores,
                        threshold=float(threshold),
                        window_size=n_i,
                        min_rejects=m_i,
                    )

                    # Add time conversions (mean first interrupt window -> seconds).
                    def _mean_window_to_sec(mean_window: float) -> float:
                        mean_window = float(mean_window)
                        if mean_window <= 0.0:
                            return 0.0
                        return float(ws_f + (mean_window - 1.0) * stride_sec)

                    val_impostor_mean_sec = _mean_window_to_sec(float(val_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0))
                    val_genuine_mean_sec = _mean_window_to_sec(float(val_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0))
                    test_impostor_mean_sec = _mean_window_to_sec(float(test_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0))
                    test_genuine_mean_sec = _mean_window_to_sec(float(test_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0))

                    # Session-level first-interrupt stats.
                    #
                    # We always output the base session stats (sessions, interrupted sessions,
                    # mean first-interrupt time) and additionally output p(first_interrupt<=T)
                    # for multiple T values (decision_time_thresholds_sec).
                    thresholds_sec = list(getattr(cfg, "decision_time_thresholds_sec", []) or [])
                    if not thresholds_sec:
                        # Default report thresholds:
                        # - Always include 1s for backward compatibility
                        # - Include integer seconds up to min(10, floor(max_t))
                        # - Always include max_t itself (even if >10 or non-integer)
                        thresholds_sec = [1.0]
                        if max_t > 0.0:
                            upper = min(10, int(math.floor(max_t)))
                            for sec in range(2, upper + 1):
                                thresholds_sec.append(float(sec))
                            thresholds_sec.append(float(max_t))
                    # Keep a stable, sorted, unique list.
                    thresholds_sec = sorted({float(x) for x in thresholds_sec if float(x) > 0.0})

                    # Back-compat: keep the original <=1s columns by always
                    # including 1.0 in the computed thresholds.
                    if 1.0 not in thresholds_sec:
                        thresholds_sec = sorted({1.0, *thresholds_sec})

                    # Ensure the configured max_t column always exists so that best-policy
                    # selection can reference it safely.
                    if max_t > 0.0 and all(abs(float(x) - float(max_t)) > 1e-9 for x in thresholds_sec):
                        thresholds_sec = sorted({*thresholds_sec, float(max_t)})

                    val_imp_times = first_interrupt_times_sec_per_session(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_arrays.scores,
                        threshold=float(threshold),
                        vote_window_size=n_i,
                        vote_min_rejects=m_i,
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=0,
                    )
                    val_gen_times = first_interrupt_times_sec_per_session(
                        val_arrays.session_ids,
                        val_arrays.labels,
                        val_arrays.scores,
                        threshold=float(threshold),
                        vote_window_size=n_i,
                        vote_min_rejects=m_i,
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=1,
                    )
                    test_imp_times = first_interrupt_times_sec_per_session(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_arrays.scores,
                        threshold=float(threshold),
                        vote_window_size=n_i,
                        vote_min_rejects=m_i,
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=0,
                    )
                    test_gen_times = first_interrupt_times_sec_per_session(
                        test_arrays.session_ids,
                        test_arrays.labels,
                        test_arrays.scores,
                        threshold=float(threshold),
                        vote_window_size=n_i,
                        vote_min_rejects=m_i,
                        window_size_sec=ws_f,
                        overlap=float(overlap),
                        target_label=1,
                    )

                    val_imp_pcols = p_first_interrupt_le_columns("val_impostor", val_imp_times, thresholds_sec)
                    val_gen_pcols = p_first_interrupt_le_columns("val_genuine", val_gen_times, thresholds_sec)
                    test_imp_pcols = p_first_interrupt_le_columns("test_impostor", test_imp_times, thresholds_sec)
                    test_gen_pcols = p_first_interrupt_le_columns("test_genuine", test_gen_times, thresholds_sec)

                    row: Dict[str, object] = {
                        "user": str(user),
                        "auth_method": str(auth_method),
                        "decision_strategy": "vote",
                        "interrupt_rule": "vote",
                        "window_size_sec": float(ws_f),
                        "overlap": float(overlap),
                        "stride_sec": float(stride_sec),
                        "target_width": int(target_width),
                        "ema_alpha": "",
                        "vote_window_size": int(n_i),
                        "vote_min_rejects": int(m_i),
                        "target_window_frr": float(tw_frr_f),
                        "threshold": float(threshold),
                        "interrupt_time_sec_min": float(interrupt_time_sec),
                        "feasible_within_max_decision_time": int(feasible_1s),
                        "vqgan_checkpoint": str(vqgan_ckpt),
                        "lm_checkpoint": str(lm_ckpt),
                        # Window-level metrics
                        "val_far": float(val_metrics.get("far", 0.0) or 0.0),
                        "val_frr": float(val_metrics.get("frr", 0.0) or 0.0),
                        "val_err": float(val_metrics.get("err", 0.0) or 0.0),
                        "val_impostor_mean_first_interrupt_sec": float(val_impostor_mean_sec),
                        "val_genuine_mean_first_interrupt_sec": float(val_genuine_mean_sec),
                        "test_far": float(test_metrics.get("far", 0.0) or 0.0),
                        "test_frr": float(test_metrics.get("frr", 0.0) or 0.0),
                        "test_err": float(test_metrics.get("err", 0.0) or 0.0),
                        "test_impostor_mean_first_interrupt_sec": float(test_impostor_mean_sec),
                        "test_genuine_mean_first_interrupt_sec": float(test_genuine_mean_sec),
                        # Session counts from vote metrics (at least one interrupt at any time).
                        "val_impostor_sessions": float(val_metrics.get("impostor_sessions", 0.0) or 0.0),
                        "val_impostor_interrupted_sessions": float(val_metrics.get("impostor_first_interrupt_n", 0.0) or 0.0),
                        "val_genuine_sessions": float(val_metrics.get("genuine_sessions", 0.0) or 0.0),
                        "val_genuine_interrupted_sessions": float(val_metrics.get("genuine_first_interrupt_n", 0.0) or 0.0),
                        "test_impostor_sessions": float(test_metrics.get("impostor_sessions", 0.0) or 0.0),
                        "test_impostor_interrupted_sessions": float(test_metrics.get("impostor_first_interrupt_n", 0.0) or 0.0),
                        "test_genuine_sessions": float(test_metrics.get("genuine_sessions", 0.0) or 0.0),
                        "test_genuine_interrupted_sessions": float(test_metrics.get("genuine_first_interrupt_n", 0.0) or 0.0),
                    }
                    # Keep the mean-first-interrupt window values that are already
                    # computed inside the vote-metric evaluation (same definition).
                    row.update(
                        {
                            "val_impostor_mean_first_interrupt_window": float(val_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0),
                            "val_genuine_mean_first_interrupt_window": float(val_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0),
                            "test_impostor_mean_first_interrupt_window": float(test_metrics.get("impostor_mean_first_interrupt_window", 0.0) or 0.0),
                            "test_genuine_mean_first_interrupt_window": float(test_metrics.get("genuine_mean_first_interrupt_window", 0.0) or 0.0),
                        }
                    )
                    row.update(val_imp_pcols)
                    row.update(val_gen_pcols)
                    row.update(test_imp_pcols)
                    row.update(test_gen_pcols)
                    rows.append(row)

    if not rows:
        raise ValueError("No grid rows produced; check inputs.")

    _write_csv(rows, grid_csv)

    if bool(getattr(cfg, "write_per_combo", False)):
        per_combo_root = out_dir / "per_combo" / method_tag
        _write_per_combo_csvs(rows, per_combo_root)

    # Pareto (on validation) over feasible policies only.
    def _time_suffix(sec: float) -> str:
        sec_f = float(sec)
        if abs(sec_f - round(sec_f)) < 1e-9:
            return f"{int(round(sec_f))}s"
        text = f"{sec_f:g}"
        text = text.replace(".", "p")
        return f"{text}s"

    time_suffix = _time_suffix(max_t) if max_t > 0.0 else "1s"
    pareto_gen_key = f"val_genuine_p_first_interrupt_le_{time_suffix}"
    feasible_rows = [r for r in rows if int(r.get("feasible_within_max_decision_time", 0) or 0) == 1]
    points: List[ParetoPoint] = []
    for r in feasible_rows:
        points.append(
            ParetoPoint(
                objectives=(
                    float(r.get("val_far", 1.0) or 1.0),
                    float(r.get("val_impostor_mean_first_interrupt_sec", float("inf")) or float("inf")),
                    float(r.get(pareto_gen_key, r.get("val_genuine_p_first_interrupt_le_1s", 1.0)) or 1.0),
                ),
                payload=r,
            )
        )
    frontier = pareto_frontier(points)
    pareto_rows = [p.payload for p in frontier]
    _write_csv(pareto_rows, pareto_csv)

    best = _pick_best_policy(ca_cfg, feasible_rows)
    if best and write_best_policy:
        policy_dir = best_json.parent
        strategy = _normalize_decision_strategy(str(best.get("decision_strategy", best.get("interrupt_rule", "vote"))))
        vqgan_checkpoint = Path(str(best["vqgan_checkpoint"]))
        vqgan_config = vqgan_checkpoint.with_suffix(".json")
        lm_checkpoint_s = str(best.get("lm_checkpoint", "") or "")
        policy_payload = {
            "user": str(user),
            "auth_method": str(auth_method),
            "window": float(best["window_size_sec"]),
            "overlap": float(best["overlap"]),
            "target_width": int(best.get("target_width", _resolve_target_width(vqgan_checkpoint, default=50)) or 50),
            "interrupt_rule": strategy,
            "decision_strategy": strategy,
            "threshold_strategy": "interrupt_window_frr",
            "target_window_frr": float(best["target_window_frr"]),
            "k_rejects": 0,
            "vote_window_size": int(best.get("vote_window_size", 0) or 0) if strategy == "vote" else 0,
            "vote_min_rejects": int(best.get("vote_min_rejects", 0) or 0) if strategy == "vote" else 0,
            "ema_alpha": float(best.get("ema_alpha", ca_cfg.auth.ema_alpha) or ca_cfg.auth.ema_alpha) if strategy == "ema" else float(ca_cfg.auth.ema_alpha),
            "max_decision_time_sec": float(max_t),
            "k_rejects_mode": str(ca_cfg.auth.k_rejects_mode),
            "threshold": float(best["threshold"]),
            "interrupt_time_sec": float(best["interrupt_time_sec_min"]),
            "vqgan_checkpoint": serialize_policy_path(vqgan_checkpoint, relative_to=policy_dir),
            "vqgan_config": serialize_policy_path(vqgan_config, relative_to=policy_dir),
            "model_version": vqgan_checkpoint.name,
            "policy_search_completed": True,
            "policy_status": "ready",
            "score_metric": "mse",
            "score_scale": "negative_reconstruction_error",
            "genuine_score_stats": genuine_stats_by_ws.get(float(best["window_size_sec"])),
            "grid_search": {
                "grid_results_csv": serialize_policy_path(grid_csv, relative_to=policy_dir),
                "pareto_frontier_csv": serialize_policy_path(pareto_csv, relative_to=policy_dir),
            },
        }
        if lm_checkpoint_s:
            policy_payload["lm_checkpoint"] = serialize_policy_path(Path(lm_checkpoint_s), relative_to=policy_dir)
        policy = {
            str(user): policy_payload
        }
        _atomic_write_json(best_json, policy)
        method_best_json = out_dir / f"best_lock_policy_{method_tag}.json"
        if method_best_json != best_json:
            _atomic_write_json(method_best_json, policy)

    return {"grid_csv": grid_csv, "pareto_csv": pareto_csv, "best_policy_json": best_json}


def _pick_best_policy(ca_cfg: CAConfig, rows: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    max_t = float(getattr(ca_cfg.auth, "max_decision_time_sec", 0.0) or 0.0)
    eps = 1e-9

    # Prefer policies that actually satisfy the configured window FRR bound on val.
    target_frr = float(getattr(ca_cfg.auth, "target_window_frr", 0.0) or 0.0)
    candidates = [r for r in rows if float(r.get("val_frr", 1.0) or 1.0) <= target_frr + eps]
    if not candidates:
        candidates = list(rows)

    def _time_suffix(sec: float) -> str:
        sec_f = float(sec)
        if abs(sec_f - round(sec_f)) < 1e-9:
            return f"{int(round(sec_f))}s"
        text = f"{sec_f:g}"
        text = text.replace(".", "p")
        return f"{text}s"

    time_suffix = _time_suffix(max_t) if max_t > 0.0 else "1s"
    imp_key = f"val_impostor_p_first_interrupt_le_{time_suffix}"
    gen_key = f"val_genuine_p_first_interrupt_le_{time_suffix}"

    # UX constraint: limit the probability of interrupting genuine sessions
    # within the configured decision time budget.
    #
    # Prefer an explicit `max_genuine_first_interrupt_p` if present in config;
    # otherwise, fall back to the same bound as `target_window_frr`.
    max_gen_p = float(getattr(ca_cfg.auth, "max_genuine_first_interrupt_p", target_frr) or target_frr)
    max_gen_p = min(1.0, max(0.0, max_gen_p))
    constrained = [
        r
        for r in candidates
        if float(r.get(gen_key, r.get("val_genuine_p_first_interrupt_le_1s", 1.0)) or 1.0) <= max_gen_p + eps
    ]
    if constrained:
        candidates = constrained

    preferred_strategy = _normalize_decision_strategy(str(getattr(ca_cfg.auth, "decision_strategy", "") or ""))
    if preferred_strategy:
        preferred = [r for r in candidates if _normalize_decision_strategy(str(r.get("decision_strategy", ""))) == preferred_strategy]
        if preferred:
            candidates = preferred

    def _key(r: Dict[str, object]) -> Tuple[float, float, float, float, float, float, float]:
        imp_p = float(r.get(imp_key, r.get("val_impostor_p_first_interrupt_le_1s", 0.0)) or 0.0)
        gen_p = float(r.get(gen_key, r.get("val_genuine_p_first_interrupt_le_1s", 0.0)) or 0.0)
        val_far = float(r.get("val_far", 1.0) or 1.0)
        val_err = float(r.get("val_err", 1.0) or 1.0)
        val_imp_mean = float(r.get("val_impostor_mean_first_interrupt_sec", float("inf")) or float("inf"))
        val_frr = float(r.get("val_frr", 1.0) or 1.0)
        t_min = float(r.get("interrupt_time_sec_min", float("inf")) or float("inf"))

        miss = 1.0 - imp_p
        return (miss, gen_p, val_far, val_imp_mean, val_err, val_frr, t_min)

    # If max_decision_time_sec is set, prioritize those with theoretical earliest time within it.
    if max_t > 0.0:
        within = [r for r in candidates if float(r.get("interrupt_time_sec_min", float("inf")) or float("inf")) <= max_t + eps]
        if within:
            candidates = within

    return sorted(candidates, key=_key)[0]


def _atomic_write_json(path: Path, obj) -> None:
    """Atomically write JSON: write to a sibling .tmp file, flush+fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _genuine_score_stats(scores, labels) -> Optional[dict]:
    """Summary statistics of genuine (label==1) per-window scores.

    Scores are negative reconstruction error (higher = more genuine). Used by
    authentication startup to reject thresholds that fall far outside the
    training/validation genuine score band.
    """
    g = np.asarray(scores)[np.asarray(labels) == 1]
    if g.size == 0:
        return None
    quantile_keys = ["0.001", "0.01", "0.05", "0.5", "0.95", "0.99", "0.999"]
    return {
        "source": "val",
        "count": int(g.size),
        "min": float(g.min()),
        "max": float(g.max()),
        "mean": float(g.mean()),
        "std": float(g.std()),
        "quantiles": {q: float(np.quantile(g, float(q))) for q in quantile_keys},
    }


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _write_per_combo_csvs(rows: Sequence[Dict[str, object]], root: Path) -> None:
    """
    Prompt Step 12: write independent files for each (t, N, M) combination.

    We group over (window_size_sec, vote_window_size, vote_min_rejects) and keep
    multiple target_window_frr candidates in the same file.
    """
    grouped: Dict[Tuple[float, int, int], List[Dict[str, object]]] = {}
    for r in rows:
        try:
            ws = float(r.get("window_size_sec", 0.0) or 0.0)
            n = int(r.get("vote_window_size", 0) or 0)
            m = int(r.get("vote_min_rejects", 0) or 0)
        except Exception:
            continue
        grouped.setdefault((ws, n, m), []).append(r)

    for (ws, n, m), group_rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        ws_tag = f"{float(ws):.1f}"
        target = root / f"t_{ws_tag}" / f"N_{int(n)}_M_{int(m)}.csv"
        group_rows = sorted(group_rows, key=lambda r: float(r.get("target_window_frr", 0.0) or 0.0))
        _write_csv(group_rows, target)
