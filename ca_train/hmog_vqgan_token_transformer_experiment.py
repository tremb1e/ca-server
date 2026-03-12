import argparse
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hmog_consecutive_rejects import (
    compute_vote_reject_window_metrics_from_arrays,
    k_from_interrupt_time,
    select_threshold_by_vote_window_frr_from_arrays,
)
from hmog_data import (
    DEFAULT_OVERLAP,
    WINDOW_SIZES,
    WindowedHMOGDataset,
    format_window_dir_name,
    iter_windows_from_csv_unlabeled_with_session,
    load_user_train_windows,
    list_available_users,
    prepare_user_datasets,
)
from hmog_metrics import compute_metrics
from hmog_token_transformer import build_token_lm
from hmog_tokenizer import encode_windows_to_tokens
from runtime_paths import dataset_root as default_dataset_root, results_dir as default_results_dir, token_cache_dir as default_token_cache_dir
from vqgan import VQGAN


logger = logging.getLogger(__name__)


def set_seed(seed: int, cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: Path) -> Tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hmog_vqgan_token_transformer.log"
    metrics_txt = log_dir / "hmog_vqgan_token_transformer_metrics.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file, metrics_txt


def save_jsonl(log_path: Path, payload: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_text_log(text_path: Path, payload: Dict) -> None:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = payload.get("metrics", {}) or {}
    thr_strategy = str(payload.get("threshold_strategy", ""))
    target_far = payload.get("target_far", None)
    target_frr = payload.get("target_frr", None)
    target_window_frr = payload.get("target_window_frr", None)
    target_far_str = "" if target_far is None else f"{float(target_far):.6f}"
    target_frr_str = "" if target_frr is None else f"{float(target_frr):.6f}"
    target_window_frr_str = "" if target_window_frr is None else f"{float(target_window_frr):.6f}"
    interrupt_after_sec = payload.get("interrupt_after_sec", None)
    interrupt_stride_sec = payload.get("interrupt_stride_sec", None)
    interrupt_time_sec = payload.get("interrupt_time_sec", None)
    vote_window_size = payload.get("vote_window_size", None)
    vote_min_rejects = payload.get("vote_min_rejects", None)
    interrupt_after_sec_str = "" if interrupt_after_sec is None else f"{float(interrupt_after_sec):.6f}"
    interrupt_stride_sec_str = "" if interrupt_stride_sec is None else f"{float(interrupt_stride_sec):.6f}"
    interrupt_time_sec_str = "" if interrupt_time_sec is None else f"{float(interrupt_time_sec):.6f}"
    vote_window_size_str = "" if vote_window_size is None else f"{int(vote_window_size)}"
    vote_min_rejects_str = "" if vote_min_rejects is None else f"{int(vote_min_rejects)}"
    line = (
        f"{ts}\tstage={payload.get('stage','')}\tuser={payload.get('user','')}"
        f"\twindow={payload.get('window','')}\tepoch={payload.get('epoch','')}"
        f"\tauc={metrics.get('auc',0):.6f}"
        f"\tfar={metrics.get('far',0):.6f}"
        f"\tfrr={metrics.get('frr',0):.6f}"
        f"\terr={metrics.get('err',0):.6f}"
        f"\teer={metrics.get('eer',0):.6f}"
        f"\tf1={metrics.get('f1',0):.6f}"
        f"\tprecision={metrics.get('precision',0):.6f}"
        f"\trecall={metrics.get('recall',0):.6f}"
        f"\timpostor_f1={metrics.get('impostor_f1',0):.6f}"
        f"\tpos={int(metrics.get('pos',0))}"
        f"\tneg={int(metrics.get('neg',0))}"
        f"\ttp={int(metrics.get('tp',0))}"
        f"\tfp={int(metrics.get('fp',0))}"
        f"\ttn={int(metrics.get('tn',0))}"
        f"\tfn={int(metrics.get('fn',0))}"
        f"\tk_rejects={int(payload.get('k_rejects',0))}"
        f"\tsession_far={metrics.get('session_far',0):.6f}"
        f"\tsession_frr={metrics.get('session_frr',0):.6f}"
        f"\tsession_tpr={metrics.get('session_tpr',0):.6f}"
        f"\tgenuine_sessions={int(metrics.get('genuine_sessions',0))}"
        f"\timpostor_sessions={int(metrics.get('impostor_sessions',0))}"
        f"\tgenuine_interrupted_sessions={int(metrics.get('genuine_interrupted_sessions',0))}"
        f"\timpostor_detected_sessions={int(metrics.get('impostor_detected_sessions',0))}"
        f"\tgenuine_avg_interrupts_per_session={metrics.get('genuine_avg_interrupts_per_session',0):.6f}"
        f"\timpostor_avg_interrupts_per_session={metrics.get('impostor_avg_interrupts_per_session',0):.6f}"
        f"\tthreshold={metrics.get('threshold',0):.6f}"
        f"\teer_threshold={metrics.get('eer_threshold',0):.6f}"
        f"\tthr_strategy={thr_strategy}"
        f"\ttarget_far={target_far_str}"
        f"\ttarget_frr={target_frr_str}"
        f"\ttarget_window_frr={target_window_frr_str}"
        f"\tinterrupt_after_sec={interrupt_after_sec_str}"
        f"\tinterrupt_stride_sec={interrupt_stride_sec_str}"
        f"\tinterrupt_time_sec={interrupt_time_sec_str}"
        f"\tvote_window_size={vote_window_size_str}"
        f"\tvote_min_rejects={vote_min_rejects_str}"
        f"\tlatency={payload.get('latency',0):.6f}"
    )
    with text_path.open("a") as f:
        f.write(line + "\n")


def build_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = WindowedHMOGDataset(train_x, train_y)
    val_dataset = WindowedHMOGDataset(val_x, val_y)
    test_dataset = WindowedHMOGDataset(test_x, test_y)

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def vqgan_reconstruction_step(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    *,
    q_loss_weight: float = 1.0,
    input_noise_std: float = 0.0,
    rec_loss_metric: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = batch.to(device, non_blocking=True)
    if input_noise_std and input_noise_std > 0:
        batch = batch + torch.randn_like(batch) * float(input_noise_std)
    with amp.autocast(device_type=device.type, enabled=use_amp):
        decoded, _, q_loss = model(batch)
        if rec_loss_metric == "mse":
            rec_loss = torch.mean((batch - decoded) ** 2)
        else:
            rec_loss = torch.mean(torch.abs(batch - decoded))
        q_loss = torch.mean(q_loss)
        loss = rec_loss + float(q_loss_weight) * q_loss
    return loss, rec_loss.detach(), q_loss.detach()


@torch.no_grad()
def evaluate_vqgan(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    *,
    threshold: Optional[float] = None,
    score_metric: str = "mse",
    threshold_strategy: str = "eer",
    target_far: Optional[float] = None,
    target_frr: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    start = time.time()
    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels_np = labels.cpu().numpy()
        with amp.autocast(device_type=device.type, enabled=use_amp):
            decoded, _, _ = model(batch)
            if score_metric == "l1":
                errors = torch.mean(torch.abs(batch - decoded), dim=(1, 2, 3))
            else:
                errors = torch.mean((batch - decoded) ** 2, dim=(1, 2, 3))
        scores = (-errors).detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels_np)
    end = time.time()

    scores_np = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    metrics = compute_metrics(
        labels_np,
        scores_np,
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        target_far=target_far,
        target_frr=target_frr,
    )
    latency = float((end - start) / max(int(labels_np.size), 1))
    return metrics, latency


class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, labels: Optional[np.ndarray] = None):
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens as (N, L), got {tokens.shape}")
        self.tokens = torch.from_numpy(tokens).to(dtype=torch.long, copy=False)
        self.labels = None if labels is None else torch.from_numpy(labels).to(dtype=torch.long, copy=False)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.tokens[idx]
        return self.tokens[idx], self.labels[idx]


@torch.no_grad()
def evaluate_token_lm(
    model: nn.Module,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    threshold: Optional[float] = None,
    threshold_strategy: str = "eer",
    target_far: Optional[float] = None,
    target_frr: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    start = time.time()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    n = int(tokens.shape[0])
    for i in range(0, n, batch_size):
        batch_tokens = torch.from_numpy(tokens[i : i + batch_size]).to(device=device, dtype=torch.long, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            # score = -NLL; larger -> more genuine
            scores = model.score(batch_tokens).detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels[i : i + batch_size])
    end = time.time()
    scores_np = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    metrics = compute_metrics(
        labels_np,
        scores_np,
        threshold=threshold,
        threshold_strategy=threshold_strategy,
        target_far=target_far,
        target_frr=target_frr,
    )
    latency = float((end - start) / max(int(labels_np.size), 1))
    return metrics, latency


def _iter_eval_windows_from_csv_with_caps(
    csv_path: Path,
    *,
    window_size_sec: float,
    target_width: int,
    target_user: str,
    max_negative: Optional[int],
    max_total: Optional[int],
) -> Iterator[Tuple[str, str, int, np.ndarray]]:
    """
    Iterate windows from a server CSV in chronological order, applying eval caps.

    Yields: (subject, session, label, window)
      - label: 1 if subject == target_user else 0.

    Capping policy (order-preserving):
      - keep all genuine windows until we start seeing impostors;
      - then keep at most `max_negative` impostor windows and stop.
      - if `max_total` is set, stop after keeping `max_total` windows (once impostors started).
    """
    target_user_norm = str(target_user).strip()
    neg_cap = None if max_negative is None else max(0, int(max_negative))
    total_cap = None if max_total is None else max(0, int(max_total))

    neg_kept = 0
    total_kept = 0
    seen_negative = False

    for _, subject, session, window in iter_windows_from_csv_unlabeled_with_session(
        csv_path,
        window_size_sec=float(window_size_sec),
        target_width=int(target_width),
    ):
        subject = str(subject).strip()
        session = str(session).strip()
        label = 1 if subject == target_user_norm else 0

        if label == 0:
            seen_negative = True
            if neg_cap is not None and neg_kept >= neg_cap:
                break

        if total_cap is not None and seen_negative and total_kept >= total_cap:
            break

        if label == 0:
            neg_kept += 1
        total_kept += 1
        yield subject, session, label, window


@torch.no_grad()
def score_lm_from_csv_to_arrays(
    vqgan: nn.Module,
    lm: nn.Module,
    *,
    csv_path: Path,
    target_user: str,
    window_size_sec: float,
    target_width: int,
    token_batch_size: int,
    device: torch.device,
    use_amp: bool,
    max_negative: Optional[int] = None,
    max_total: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Stream windows from CSV -> VQGAN tokens -> LM score, producing aligned arrays:
      - session_ids: int32, increments when (subject, session) changes
      - labels: int8, 1=genuine (subject==target_user), 0=impostor
      - scores: float32, -NLL per window (larger => more genuine)

    If max_negative/max_total are provided, applies the same capping policy as
    `_iter_eval_windows_from_csv_with_caps`. If they are None, consumes the full CSV.
    """
    target_user_norm = str(target_user).strip()

    batch_windows: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_session_ids: List[int] = []

    scores_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    session_chunks: List[np.ndarray] = []

    cur_key: Optional[Tuple[str, str]] = None
    session_id = 0

    start = time.time()

    def _flush() -> None:
        if not batch_windows:
            return
        windows_np = np.stack(batch_windows, axis=0).astype(np.float32, copy=False)
        tok = encode_windows_to_tokens(
            vqgan, windows_np, batch_size=int(token_batch_size), device=device, use_amp=use_amp
        )
        tokens = torch.from_numpy(tok.tokens).to(device=device, dtype=torch.long, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            scores = lm.score(tokens).detach().cpu().numpy().astype(np.float32, copy=False)
        scores_chunks.append(scores)
        labels_chunks.append(np.asarray(batch_labels, dtype=np.int8))
        session_chunks.append(np.asarray(batch_session_ids, dtype=np.int32))
        batch_windows.clear()
        batch_labels.clear()
        batch_session_ids.clear()

    if max_negative is None and max_total is None:
        window_iter = (
            (subject, session, window)
            for _, subject, session, window in iter_windows_from_csv_unlabeled_with_session(
                csv_path,
                window_size_sec=float(window_size_sec),
                target_width=int(target_width),
            )
        )
        neg_cap = None
        total_cap = None
    else:
        window_iter = (
            (subject, session, window)
            for subject, session, _, window in _iter_eval_windows_from_csv_with_caps(
                csv_path,
                window_size_sec=float(window_size_sec),
                target_width=int(target_width),
                target_user=str(target_user_norm),
                max_negative=max_negative,
                max_total=max_total,
            )
        )
        neg_cap = max_negative
        total_cap = max_total
        del neg_cap, total_cap  # kept for debugging parity

    # Note: We compute labels ourselves to support both capped and full iterators.
    for subject, session, window in window_iter:
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
    sessions_all = np.concatenate(session_chunks, axis=0)
    end = time.time()
    latency = float((end - start) / max(int(scores_all.size), 1))
    return sessions_all, labels_all, scores_all, latency


@torch.no_grad()
def collect_lm_scores_from_csv(
    vqgan: nn.Module,
    lm: nn.Module,
    *,
    csv_path: Path,
    target_user: str,
    window_size_sec: float,
    target_width: int,
    max_negative: Optional[int],
    max_total: Optional[int],
    token_batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[List[Tuple[str, str, int, float]], float]:
    """
    Stream windows from CSV -> VQGAN tokens -> LM score, keeping (subject, session, label, score).

    Returns:
      - scored_windows: list of tuples in chronological order.
      - latency_sec_per_window: end-to-end average time per window for scoring.
    """
    batch_windows: List[np.ndarray] = []
    batch_meta: List[Tuple[str, str, int]] = []
    scored: List[Tuple[str, str, int, float]] = []

    start = time.time()

    def _flush() -> None:
        if not batch_windows:
            return
        windows_np = np.stack(batch_windows, axis=0).astype(np.float32, copy=False)
        tok = encode_windows_to_tokens(
            vqgan, windows_np, batch_size=int(token_batch_size), device=device, use_amp=use_amp
        )
        tokens = torch.from_numpy(tok.tokens).to(device=device, dtype=torch.long, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            scores = lm.score(tokens).detach().cpu().numpy()
        for (subject, session, label), score in zip(batch_meta, scores):
            scored.append((subject, session, int(label), float(score)))
        batch_windows.clear()
        batch_meta.clear()

    for subject, session, label, window in _iter_eval_windows_from_csv_with_caps(
        csv_path,
        window_size_sec=float(window_size_sec),
        target_width=int(target_width),
        target_user=str(target_user),
        max_negative=max_negative,
        max_total=max_total,
    ):
        batch_windows.append(window)
        batch_meta.append((subject, session, int(label)))
        if len(batch_windows) >= int(token_batch_size):
            _flush()

    _flush()
    end = time.time()
    latency = float((end - start) / max(len(scored), 1))
    return scored, latency


def select_threshold_by_k_session_frr(
    scored_windows: Sequence[Tuple[str, str, int, float]],
    *,
    k: int,
    target_session_frr: float,
    reset_on_interrupt: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Select the *highest* threshold such that session_FRR <= target_session_frr on the given scored windows.

    session_FRR is computed under the K-consecutive-rejects interrupt rule.
    """
    if not scored_windows:
        raise ValueError("Empty scored_windows for threshold selection")
    k = int(k)
    if k <= 0:
        # If K is disabled, the interrupt rule never triggers; return the strictest reasonable threshold.
        scores = np.array([s for _, _, _, s in scored_windows], dtype=np.float64)
        thr = float(np.max(scores))
        metrics = compute_k_consecutive_reject_session_metrics(
            scored_windows, threshold=thr, k=k, reset_on_interrupt=reset_on_interrupt
        )
        return thr, metrics

    scores = np.array([s for _, _, _, s in scored_windows], dtype=np.float64)
    candidates = np.unique(scores)
    if candidates.size == 0:
        raise ValueError("No score candidates found for threshold selection")
    candidates.sort()  # ascending

    target_session_frr = float(target_session_frr)
    if target_session_frr < 0.0:
        target_session_frr = 0.0
    if target_session_frr > 1.0:
        target_session_frr = 1.0

    best_thr = float(candidates[0])
    best_metrics = compute_k_consecutive_reject_session_metrics(
        scored_windows, threshold=best_thr, k=k, reset_on_interrupt=reset_on_interrupt
    )

    lo = 0
    hi = int(candidates.size - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        thr = float(candidates[mid])
        metrics = compute_k_consecutive_reject_session_metrics(
            scored_windows, threshold=thr, k=k, reset_on_interrupt=reset_on_interrupt
        )
        if float(metrics.get("session_frr", 0.0)) <= target_session_frr:
            best_thr = thr
            best_metrics = metrics
            lo = mid + 1
        else:
            hi = mid - 1

    return float(best_thr), best_metrics


def _scan_subjects(csv_path: Path, *, max_rows: int, max_subjects: int = 20) -> List[str]:
    """
    Lightweight sanity scan: read up to `max_rows` CSV rows and return unique subject values (sorted).
    """
    if max_rows <= 0:
        return []
    subjects = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        header = [str(h).strip().lstrip("\ufeff") for h in header]
        lookup = {name.lower(): i for i, name in enumerate(header)}
        idx_subject = lookup.get("subject", None)
        if idx_subject is None:
            return []
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            subjects.add(str(row[idx_subject]).strip())
            if len(subjects) >= max_subjects:
                break
            if i >= max_rows:
                break
    return sorted(subjects)


def _token_cache_path(token_cache_dir: Path, *, user_id: str, window_size: float, split: str) -> Path:
    token_cache_dir.mkdir(parents=True, exist_ok=True)
    return token_cache_dir / f"tokens_user_{user_id}_ws_{window_size:.1f}_{split}.npz"


def load_token_cache(
    cache_path: Path,
    *,
    expected_vqgan_ckpt: str,
    expected_vqgan_ckpt_mtime_ns: int,
    expected_vqgan_ckpt_size: int,
    expected_num_codebook_vectors: int,
    expected_target_width: int,
    expected_seed: int,
    expected_num_windows: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            vqgan_ckpt = str(data["vqgan_ckpt"].item())
            if vqgan_ckpt != expected_vqgan_ckpt:
                return None
            cached_mtime_ns = int(data["vqgan_ckpt_mtime_ns"].item()) if "vqgan_ckpt_mtime_ns" in data else -1
            cached_size = int(data["vqgan_ckpt_size"].item()) if "vqgan_ckpt_size" in data else -1
            if int(cached_mtime_ns) != int(expected_vqgan_ckpt_mtime_ns):
                return None
            if int(cached_size) != int(expected_vqgan_ckpt_size):
                return None
            if int(data["num_codebook_vectors"].item()) != int(expected_num_codebook_vectors):
                return None
            if int(data["target_width"].item()) != int(expected_target_width):
                return None
            cached_seed = int(data["seed"].item()) if "seed" in data else -1
            if int(cached_seed) != int(expected_seed):
                return None
            cached_num_windows = int(data["num_windows"].item()) if "num_windows" in data else -1
            if int(cached_num_windows) != int(expected_num_windows):
                return None
            tokens = data["tokens"].astype(np.int64, copy=False)
            if int(tokens.shape[0]) != int(expected_num_windows):
                return None
            codebook_hw = tuple(int(x) for x in data["codebook_hw"].tolist())
            return tokens, (int(codebook_hw[0]), int(codebook_hw[1]))
    except Exception:
        return None


def save_token_cache(
    cache_path: Path,
    *,
    tokens: np.ndarray,
    codebook_hw: Tuple[int, int],
    vqgan_ckpt: str,
    vqgan_ckpt_mtime_ns: int,
    vqgan_ckpt_size: int,
    num_codebook_vectors: int,
    target_width: int,
    seed: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        tokens=tokens.astype(np.int64, copy=False),
        codebook_hw=np.asarray(codebook_hw, dtype=np.int32),
        vqgan_ckpt=str(vqgan_ckpt),
        vqgan_ckpt_mtime_ns=int(vqgan_ckpt_mtime_ns),
        vqgan_ckpt_size=int(vqgan_ckpt_size),
        num_codebook_vectors=int(num_codebook_vectors),
        target_width=int(target_width),
        seed=int(seed),
        num_windows=int(tokens.shape[0]),
    )


def run_single_window(
    args: argparse.Namespace,
    *,
    user_id: str,
    window_size: float,
    device: torch.device,
    json_log_path: Path,
    text_log_path: Path,
) -> Dict:
    prep_start = time.time()
    target_width = int(args.target_width if args.target_width > 0 else int(round(window_size * 100)))
    full_eval = bool(getattr(args, "full_eval", False))

    vote_window_size = int(getattr(args, "vote_window_size", 0) or 0)
    vote_min_rejects = int(getattr(args, "vote_min_rejects", 0) or 0)
    vote_enabled = vote_window_size > 0 or vote_min_rejects > 0
    if vote_enabled:
        if vote_window_size <= 0 or vote_min_rejects <= 0:
            raise ValueError("--vote-window-size and --vote-min-rejects must both be > 0 when vote is enabled.")
        if vote_min_rejects > vote_window_size:
            raise ValueError("--vote-min-rejects must be <= --vote-window-size.")
        if int(getattr(args, "k_rejects", 0)) > 0:
            raise ValueError("Vote mode is mutually exclusive with --k-rejects.")
        if getattr(args, "interrupt_after_sec", None) is not None and float(args.interrupt_after_sec) > 0.0:
            raise ValueError("Vote mode is mutually exclusive with --interrupt-after-sec.")
        if getattr(args, "interrupt_after_sec_base", None) is not None:
            raise ValueError("Vote mode is mutually exclusive with --interrupt-after-sec-base/--interrupt-after-sec-scale.")

    stride_sec = float(window_size) * (1.0 - float(args.overlap))
    if stride_sec <= 0.0:
        raise ValueError(f"Invalid stride: window={window_size} overlap={args.overlap}")

    # Continuous-auth "interrupt" policy:
    # - legacy: fixed K consecutive rejects
    # - requested: fixed interrupt time (seconds), converted to K via stride
    # - usability: per-window dynamic interrupt time (seconds), computed as:
    #       T = interrupt_after_sec_base + interrupt_after_sec_scale * window_size
    if vote_enabled:
        interrupt_after_sec = None
        interrupt_after_sec_base = None
        interrupt_after_sec_scale = 0.0
        interrupt_min_k = 0
        k_rejects_effective = 0
        interrupt_time_sec = float(window_size + (vote_window_size - 1) * stride_sec)
    else:
        interrupt_after_sec = getattr(args, "interrupt_after_sec", None)
        if interrupt_after_sec is not None and float(interrupt_after_sec) <= 0.0:
            interrupt_after_sec = None

        interrupt_after_sec_base = getattr(args, "interrupt_after_sec_base", None)
        interrupt_after_sec_scale = getattr(args, "interrupt_after_sec_scale", 0.0)
        interrupt_min_k = int(getattr(args, "interrupt_min_k", 0) or 0)

        if interrupt_after_sec_base is None and float(interrupt_after_sec_scale or 0.0) != 0.0:
            raise ValueError("--interrupt-after-sec-scale requires --interrupt-after-sec-base.")

        if interrupt_after_sec_base is not None:
            if interrupt_after_sec is not None:
                raise ValueError(
                    "Use either --interrupt-after-sec or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both."
                )
            if int(getattr(args, "k_rejects", 0)) > 0:
                raise ValueError("Use either --k-rejects or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both.")
            interrupt_after_sec = float(interrupt_after_sec_base) + float(interrupt_after_sec_scale) * float(window_size)
            if interrupt_after_sec <= 0.0:
                raise ValueError(
                    f"Computed interrupt_after_sec={interrupt_after_sec} must be >0 (base={interrupt_after_sec_base}, "
                    f"scale={interrupt_after_sec_scale}, window={window_size})"
                )

        if interrupt_after_sec is not None and int(getattr(args, "k_rejects", 0)) > 0:
            raise ValueError("Use either --k-rejects or --interrupt-after-sec (including dynamic), not both.")
        k_rejects_effective = int(getattr(args, "k_rejects", 0))
        if interrupt_after_sec is not None:
            k_rejects_effective = k_from_interrupt_time(
                float(interrupt_after_sec), window_size_sec=float(window_size), overlap=float(args.overlap)
            )
        if interrupt_min_k > 0:
            k_rejects_effective = max(int(k_rejects_effective), int(interrupt_min_k))
        interrupt_time_sec = float(window_size + (k_rejects_effective - 1) * stride_sec) if k_rejects_effective > 0 else 0.0

    if full_eval:
        train_x, train_y = load_user_train_windows(
            target_user=user_id,
            window_size_sec=window_size,
            target_width=target_width,
            base_path=Path(args.dataset_path),
            seed=args.seed,
            max_train_windows=args.max_train_per_user,
        )
        val_x = np.empty((0, 1, 12, target_width), dtype=np.float32)
        val_y = np.empty((0,), dtype=np.int64)
        test_x = np.empty((0, 1, 12, target_width), dtype=np.float32)
        test_y = np.empty((0,), dtype=np.int64)
    else:
        train_x, train_y, val_x, val_y, test_x, test_y = prepare_user_datasets(
            target_user=user_id,
            window_size_sec=window_size,
            overlap=args.overlap,
            target_width=target_width,
            prep_workers=args.prep_workers,
            max_negative_per_split=args.max_negative_per_split,
            max_eval_per_split=args.max_eval_per_split,
            base_path=Path(args.dataset_path),
            full_scan_eval=args.full_scan_eval,
            seed=args.seed,
        )
    prep_dur = time.time() - prep_start
    logger.info(
        "[PREP] user=%s ws=%.1f target_width=%d train=%d val(pos=%d,neg=%d) test(pos=%d,neg=%d) full_eval=%s time=%.2fs",
        user_id,
        window_size,
        target_width,
        len(train_y),
        int((val_y == 1).sum()),
        int((val_y == 0).sum()),
        int((test_y == 1).sum()),
        int((test_y == 0).sum()),
        str(full_eval),
        prep_dur,
    )

    # Sanity checks: labels are defined by `subject == target_user` (server pipeline contract).
    if train_y.size and not np.all(train_y == 1):
        raise ValueError(
            f"[DATA] train split for user={user_id} ws={window_size:.1f} must be genuine-only; "
            f"got labels={np.unique(train_y)}"
        )
    for split_name, split_y in (("val", val_y), ("test", test_y)):
        uniq = np.unique(split_y)
        if split_y.size and len(uniq) < 2:
            raise ValueError(
                f"[DATA] {split_name} split for user={user_id} ws={window_size:.1f} must contain both classes "
                f"(subject=={user_id} -> 1, attackers -> 0); got labels={uniq}"
            )

    if int(getattr(args, "subject_scan_rows", 0) or 0) > 0:
        ws_dir = Path(args.dataset_path) / f"{window_size:.1f}" / user_id
        for split in ("train", "val", "test"):
            csv_path = ws_dir / f"{split}.csv"
            subjects = _scan_subjects(csv_path, max_rows=int(args.subject_scan_rows))
            if subjects:
                logger.info("[SUBJECTS] user=%s ws=%.1f split=%s subjects=%s", user_id, window_size, split, subjects)

    if args.max_train_per_user and len(train_x) > args.max_train_per_user:
        idx = np.random.default_rng(args.seed).choice(len(train_x), size=int(args.max_train_per_user), replace=False)
        train_x = train_x[idx]
        train_y = train_y[idx]

    args.input_height = int(train_x.shape[2]) if train_x.size else 12
    args.input_width = int(train_x.shape[3]) if train_x.size else int(target_width)
    args.use_nonlocal = not getattr(args, "no_nonlocal", False)

    train_loader, val_loader, test_loader = build_loaders(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vqgan_ckpt_path = ckpt_dir / f"vqgan_user_{user_id}_ws_{window_size:.1f}.pt"
    lm_ckpt_path = ckpt_dir / f"token_gpt_user_{user_id}_ws_{window_size:.1f}.pt"

    # -----------------------------
    # Stage 1: Train (or load) VQGAN
    # -----------------------------
    vqgan_trained_epochs = 0
    vqgan_best_epoch = 0
    vqgan = VQGAN(args).to(device)
    if args.reuse_vqgan and vqgan_ckpt_path.exists():
        vqgan.load_state_dict(torch.load(vqgan_ckpt_path, map_location=device))
        logger.info("[VQGAN] reuse checkpoint %s", vqgan_ckpt_path)
        cfg_path = vqgan_ckpt_path.with_suffix(".json")
        if cfg_path.exists():
            try:
                cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                vqgan_trained_epochs = int(cfg_payload.get("trained_epochs", 0) or 0)
                vqgan_best_epoch = int(cfg_payload.get("best_epoch", 0) or 0)
            except Exception:
                pass
        else:
            # Keep inference/grid-search usable even when old checkpoints were saved
            # without a companion JSON config.
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_channels": int(args.base_channels),
                        "latent_dim": int(args.latent_dim),
                        "num_codebook_vectors": int(args.num_codebook_vectors),
                        "beta": float(args.beta),
                        "image_channels": int(args.image_channels),
                        "use_nonlocal": bool(args.use_nonlocal),
                        "input_height": int(args.input_height),
                        "input_width": int(args.input_width),
                        "target_width": int(target_width),
                        "window_size": float(window_size),
                        "trained_epochs": int(vqgan_trained_epochs),
                        "best_epoch": int(vqgan_best_epoch),
                        "early_stop_patience": int(getattr(args, "early_stop_patience", 0) or 0),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
    else:
        optimizer = torch.optim.AdamW(
            params=vqgan.parameters(),
            lr=args.vqgan_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=float(args.vqgan_weight_decay),
        )
        scaler = amp.GradScaler(device.type, enabled=args.use_amp)
        early_stop_patience = int(getattr(args, "early_stop_patience", 0) or 0)
        early_stop_enabled = early_stop_patience > 0
        best_auc = float("-inf")
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        no_improve = 0
        vqgan_trained_epochs = 0

        for epoch in range(int(args.vqgan_epochs)):
            vqgan.train()
            pbar = tqdm(train_loader, desc=f"VQGAN user={user_id} ws={window_size:.1f} epoch={epoch+1}/{args.vqgan_epochs}")
            for batch, _ in pbar:
                optimizer.zero_grad(set_to_none=True)
                loss, rec_loss, q_loss = vqgan_reconstruction_step(
                    vqgan,
                    batch,
                    device,
                    args.use_amp,
                    q_loss_weight=args.q_loss_weight,
                    input_noise_std=args.input_noise_std,
                    rec_loss_metric=args.train_rec_loss,
                )
                scaler.scale(loss).backward()
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vqgan.parameters(), float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
                pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", q=f"{q_loss.item():.4f}")

            vqgan_trained_epochs = int(epoch + 1)

            should_eval = bool(early_stop_enabled) or bool(args.vqgan_val_interval and (epoch + 1) % int(args.vqgan_val_interval) == 0)
            if should_eval:
                val_metrics, val_latency = evaluate_vqgan(vqgan, val_loader, device, args.use_amp, score_metric=args.score_metric)
                payload = {
                    "stage": "vqgan-val",
                    "user": user_id,
                    "window": window_size,
                    "epoch": epoch + 1,
                    "metrics": val_metrics,
                    "latency": val_latency,
                }
                save_jsonl(json_log_path, payload)
                save_text_log(text_log_path, payload)
                logger.info("[VQGAN][VAL] user=%s ws=%.1f epoch=%d metrics=%s", user_id, window_size, epoch + 1, val_metrics)

                if early_stop_enabled:
                    auc = float(val_metrics.get("auc", 0.0) or 0.0)
                    if auc > best_auc + 1e-6:
                        best_auc = auc
                        best_epoch = int(epoch + 1)
                        # Store a CPU snapshot to avoid doubling GPU memory usage.
                        best_state = {k: v.detach().cpu().clone() for k, v in vqgan.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= int(early_stop_patience):
                        logger.info(
                            "[VQGAN][EARLY-STOP] user=%s ws=%.1f stop at epoch=%d (best_epoch=%d best_auc=%.6f)",
                            user_id,
                            window_size,
                            int(epoch + 1),
                            int(best_epoch),
                            float(best_auc),
                        )
                        break

        if early_stop_enabled and best_state is not None:
            vqgan.load_state_dict({k: v.to(device=device) for k, v in best_state.items()})
            logger.info(
                "[VQGAN][EARLY-STOP] user=%s ws=%.1f restore best_epoch=%d best_auc=%.6f (trained_epochs=%d)",
                user_id,
                window_size,
                int(best_epoch),
                float(best_auc),
                int(vqgan_trained_epochs),
            )
            vqgan_best_epoch = int(best_epoch)
        else:
            vqgan_best_epoch = int(vqgan_trained_epochs)

        torch.save(vqgan.state_dict(), vqgan_ckpt_path)
        with (vqgan_ckpt_path.with_suffix(".json")).open("w") as f:
            json.dump(
                {
                    "base_channels": int(args.base_channels),
                    "latent_dim": int(args.latent_dim),
                    "num_codebook_vectors": int(args.num_codebook_vectors),
                    "beta": float(args.beta),
                    "image_channels": int(args.image_channels),
                    "use_nonlocal": bool(args.use_nonlocal),
                    "input_height": int(args.input_height),
                    "input_width": int(args.input_width),
                    "target_width": int(target_width),
                    "window_size": float(window_size),
                    "trained_epochs": int(vqgan_trained_epochs),
                    "best_epoch": int(vqgan_best_epoch),
                    "early_stop_patience": int(early_stop_patience),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("[VQGAN] saved checkpoint %s", vqgan_ckpt_path)

    # Final VQGAN baseline metrics
    #
    # In full-eval mode we avoid loading full val/test window tensors in memory.
    # We therefore skip baseline metrics here (pos/neg will be 0, clearly signaling "not evaluated").
    if full_eval:
        vqgan_val_metrics = compute_metrics(np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float64))
        vqgan_test_metrics = compute_metrics(np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float64))
        vqgan_val_latency = 0.0
        vqgan_test_latency = 0.0
        logger.info("[VQGAN][SKIP] full_eval enabled; skip baseline val/test metrics")
    else:
        vqgan_val_metrics, vqgan_val_latency = evaluate_vqgan(
            vqgan, val_loader, device, args.use_amp, score_metric=args.score_metric
        )
        vqgan_test_metrics, vqgan_test_latency = evaluate_vqgan(
            vqgan,
            test_loader,
            device,
            args.use_amp,
            threshold=float(vqgan_val_metrics.get("threshold", 0.0)),
            score_metric=args.score_metric,
        )

    # -----------------------------
    # Stage 2: Tokenize splits
    # -----------------------------
    token_cache_dir = Path(args.token_cache_dir)
    expected_vqgan_ckpt = str(vqgan_ckpt_path)
    ckpt_stat = vqgan_ckpt_path.stat()
    expected_vqgan_ckpt_mtime_ns = int(getattr(ckpt_stat, "st_mtime_ns", int(ckpt_stat.st_mtime * 1e9)))
    expected_vqgan_ckpt_size = int(ckpt_stat.st_size)
    cached = load_token_cache(
        _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="train"),
        expected_vqgan_ckpt=expected_vqgan_ckpt,
        expected_vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
        expected_vqgan_ckpt_size=expected_vqgan_ckpt_size,
        expected_num_codebook_vectors=args.num_codebook_vectors,
        expected_target_width=target_width,
        expected_seed=args.seed,
        expected_num_windows=int(train_x.shape[0]),
    )
    if cached is None:
        train_tok = encode_windows_to_tokens(
            vqgan,
            train_x,
            batch_size=args.token_batch_size,
            device=device,
            use_amp=args.use_amp,
            desc=f"user={user_id} ws={window_size:.1f} split=train",
        )
        save_token_cache(
            _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="train"),
            tokens=train_tok.tokens,
            codebook_hw=train_tok.codebook_hw,
            vqgan_ckpt=expected_vqgan_ckpt,
            vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
            vqgan_ckpt_size=expected_vqgan_ckpt_size,
            num_codebook_vectors=args.num_codebook_vectors,
            target_width=target_width,
            seed=args.seed,
        )
        train_tokens = train_tok.tokens
        codebook_hw = train_tok.codebook_hw
    else:
        train_tokens, codebook_hw = cached

    if full_eval:
        # In full-eval mode we score val/test directly from CSV to avoid materializing huge
        # window tensors (and token caches) in memory/disk.
        token_len = int(train_tokens.shape[1]) if train_tokens.ndim == 2 else 0
        val_tokens = np.empty((0, token_len), dtype=np.int64)
        test_tokens = np.empty((0, token_len), dtype=np.int64)
    else:
        val_cached = load_token_cache(
            _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="val"),
            expected_vqgan_ckpt=expected_vqgan_ckpt,
            expected_vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
            expected_vqgan_ckpt_size=expected_vqgan_ckpt_size,
            expected_num_codebook_vectors=args.num_codebook_vectors,
            expected_target_width=target_width,
            expected_seed=args.seed,
            expected_num_windows=int(val_x.shape[0]),
        )
        if val_cached is None:
            val_tok = encode_windows_to_tokens(
                vqgan,
                val_x,
                batch_size=args.token_batch_size,
                device=device,
                use_amp=args.use_amp,
                desc=f"user={user_id} ws={window_size:.1f} split=val",
            )
            save_token_cache(
                _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="val"),
                tokens=val_tok.tokens,
                codebook_hw=val_tok.codebook_hw,
                vqgan_ckpt=expected_vqgan_ckpt,
                vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
                vqgan_ckpt_size=expected_vqgan_ckpt_size,
                num_codebook_vectors=args.num_codebook_vectors,
                target_width=target_width,
                seed=args.seed,
            )
            val_tokens = val_tok.tokens
        else:
            val_tokens, _ = val_cached

        test_cached = load_token_cache(
            _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="test"),
            expected_vqgan_ckpt=expected_vqgan_ckpt,
            expected_vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
            expected_vqgan_ckpt_size=expected_vqgan_ckpt_size,
            expected_num_codebook_vectors=args.num_codebook_vectors,
            expected_target_width=target_width,
            expected_seed=args.seed,
            expected_num_windows=int(test_x.shape[0]),
        )
        if test_cached is None:
            test_tok = encode_windows_to_tokens(
                vqgan,
                test_x,
                batch_size=args.token_batch_size,
                device=device,
                use_amp=args.use_amp,
                desc=f"user={user_id} ws={window_size:.1f} split=test",
            )
            save_token_cache(
                _token_cache_path(token_cache_dir, user_id=user_id, window_size=window_size, split="test"),
                tokens=test_tok.tokens,
                codebook_hw=test_tok.codebook_hw,
                vqgan_ckpt=expected_vqgan_ckpt,
                vqgan_ckpt_mtime_ns=expected_vqgan_ckpt_mtime_ns,
                vqgan_ckpt_size=expected_vqgan_ckpt_size,
                num_codebook_vectors=args.num_codebook_vectors,
                target_width=target_width,
                seed=args.seed,
            )
            test_tokens = test_tok.tokens
        else:
            test_tokens, _ = test_cached

    if train_tokens.size == 0:
        raise ValueError(f"Empty train tokens for user={user_id} ws={window_size:.1f}")

    token_len = int(train_tokens.shape[1])
    logger.info("[TOKENS] user=%s ws=%.1f codebook_hw=%s token_len=%d", user_id, window_size, codebook_hw, token_len)
    try:
        flat = train_tokens.reshape(-1)
        counts = np.bincount(flat, minlength=int(args.num_codebook_vectors))
        probs = counts[counts > 0] / max(int(counts.sum()), 1)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else 0.0
        perplexity = float(np.exp(entropy))
        unique_tokens = int((counts > 0).sum())
        logger.info(
            "[TOKENS][USAGE] user=%s ws=%.1f unique=%d/%d perplexity=%.2f",
            user_id,
            window_size,
            unique_tokens,
            int(args.num_codebook_vectors),
            perplexity,
        )
    except Exception:
        # Token stats are best-effort (do not fail the experiment).
        pass

    # -----------------------------
    # Stage 3: Train (or load) token LM
    # -----------------------------
    lm, lm_cfg = build_token_lm(
        num_codebook_vectors=args.num_codebook_vectors,
        block_size=token_len,
        n_layer=args.lm_n_layer,
        n_head=args.lm_n_head,
        n_embd=args.lm_n_embd,
        dropout=args.lm_dropout,
        sos_token=None,
    )
    lm = lm.to(device)

    lm_trained_epochs = 0
    lm_best_epoch = 0
    reuse_lm = bool(args.reuse_lm and lm_ckpt_path.exists())
    if reuse_lm:
        cfg_path = lm_ckpt_path.with_suffix(".json")
        if not cfg_path.exists():
            reuse_lm = False
        else:
            try:
                cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg_payload = {}
            cfg_vqgan_ckpt = str(cfg_payload.get("vqgan_ckpt", ""))
            cfg_mtime = int(cfg_payload.get("vqgan_ckpt_mtime_ns", -1))
            cfg_size = int(cfg_payload.get("vqgan_ckpt_size", -1))
            if cfg_vqgan_ckpt != expected_vqgan_ckpt:
                reuse_lm = False
            if cfg_mtime != int(expected_vqgan_ckpt_mtime_ns) or cfg_size != int(expected_vqgan_ckpt_size):
                reuse_lm = False
            if reuse_lm:
                try:
                    lm_trained_epochs = int(cfg_payload.get("trained_epochs", 0) or 0)
                    lm_best_epoch = int(cfg_payload.get("best_epoch", 0) or 0)
                except Exception:
                    pass

    if reuse_lm:
        lm.load_state_dict(torch.load(lm_ckpt_path, map_location=device))
        logger.info("[LM] reuse checkpoint %s", lm_ckpt_path)
    else:
        train_token_loader = DataLoader(
            TokenDataset(train_tokens),
            batch_size=int(args.lm_batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=True,
            persistent_workers=bool(args.num_workers),
        )
        lm_optim = torch.optim.AdamW(
            lm.parameters(),
            lr=float(args.lm_lr),
            betas=(0.9, 0.95),
            weight_decay=float(args.lm_weight_decay),
        )
        lm_scaler = amp.GradScaler(device.type, enabled=args.use_amp)
        early_stop_patience = int(getattr(args, "early_stop_patience", 0) or 0)
        early_stop_enabled = early_stop_patience > 0
        best_auc = float("-inf")
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        no_improve = 0
        lm_trained_epochs = 0

        for epoch in range(int(args.lm_epochs)):
            lm.train()
            pbar = tqdm(train_token_loader, desc=f"LM user={user_id} ws={window_size:.1f} epoch={epoch+1}/{args.lm_epochs}")
            for batch_tokens in pbar:
                lm_optim.zero_grad(set_to_none=True)
                batch_tokens = batch_tokens.to(device=device, dtype=torch.long, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=args.use_amp):
                    loss = lm.loss(batch_tokens)
                lm_scaler.scale(loss).backward()
                if args.lm_grad_clip_norm and args.lm_grad_clip_norm > 0:
                    lm_scaler.unscale_(lm_optim)
                    torch.nn.utils.clip_grad_norm_(lm.parameters(), float(args.lm_grad_clip_norm))
                lm_scaler.step(lm_optim)
                lm_scaler.update()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            lm_trained_epochs = int(epoch + 1)

            if early_stop_enabled:
                # Use quick, in-memory val AUC as the early-stop signal.
                # (Final lock metrics still use the vote rule on streamed CSV.)
                val_metrics, val_latency = evaluate_token_lm(
                    lm,
                    val_tokens,
                    val_y,
                    batch_size=int(args.lm_eval_batch_size),
                    device=device,
                    use_amp=bool(args.use_amp),
                    threshold=None,
                    threshold_strategy="eer",
                )
                auc = float(val_metrics.get("auc", 0.0) or 0.0)
                logger.info(
                    "[LM][VAL] user=%s ws=%.1f epoch=%d auc=%.6f eer=%.6f",
                    user_id,
                    window_size,
                    int(epoch + 1),
                    float(auc),
                    float(val_metrics.get("eer", 0.0) or 0.0),
                )
                if auc > best_auc + 1e-6:
                    best_auc = auc
                    best_epoch = int(epoch + 1)
                    best_state = {k: v.detach().cpu().clone() for k, v in lm.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= int(early_stop_patience):
                    logger.info(
                        "[LM][EARLY-STOP] user=%s ws=%.1f stop at epoch=%d (best_epoch=%d best_auc=%.6f)",
                        user_id,
                        window_size,
                        int(epoch + 1),
                        int(best_epoch),
                        float(best_auc),
                    )
                    break

        if early_stop_enabled and best_state is not None:
            lm.load_state_dict({k: v.to(device=device) for k, v in best_state.items()})
            logger.info(
                "[LM][EARLY-STOP] user=%s ws=%.1f restore best_epoch=%d best_auc=%.6f (trained_epochs=%d)",
                user_id,
                window_size,
                int(best_epoch),
                float(best_auc),
                int(lm_trained_epochs),
            )
            lm_best_epoch = int(best_epoch)
        else:
            lm_best_epoch = int(lm_trained_epochs)

        torch.save(lm.state_dict(), lm_ckpt_path)
        with (lm_ckpt_path.with_suffix(".json")).open("w") as f:
            payload = dict(lm_cfg.__dict__)
            payload.update(
                {
                    "vqgan_ckpt": str(expected_vqgan_ckpt),
                    "vqgan_ckpt_mtime_ns": int(expected_vqgan_ckpt_mtime_ns),
                    "vqgan_ckpt_size": int(expected_vqgan_ckpt_size),
                    "target_width": int(target_width),
                    "window_size": float(window_size),
                    "trained_epochs": int(lm_trained_epochs),
                    "best_epoch": int(lm_best_epoch),
                    "early_stop_patience": int(early_stop_patience),
                }
            )
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("[LM] saved checkpoint %s", lm_ckpt_path)

    lm_scored_val: Optional[List[Tuple[str, str, int, float]]] = None
    lm_scored_test: Optional[List[Tuple[str, str, int, float]]] = None
    lmseq_val_metrics: Optional[Dict[str, float]] = None
    lmseq_test_metrics: Optional[Dict[str, float]] = None
    lmseq_val_latency: Optional[float] = None
    lmseq_test_latency: Optional[float] = None

    strategy = str(args.threshold_strategy).lower().strip()
    k_rejects = int(k_rejects_effective)
    if full_eval or vote_enabled or strategy in {"k_session_frr", "interrupt_window_frr"}:
        if strategy in {"k_session_frr", "interrupt_window_frr"} and not vote_enabled:
            raise ValueError("--threshold-strategy interrupt_window_frr requires vote mode (--vote-window-size/--vote-min-rejects).")

        ws_tag = format_window_dir_name(window_size)
        base_path = Path(args.dataset_path)
        user_dir = base_path / ws_tag / user_id
        val_csv = user_dir / "val.csv"
        test_csv = user_dir / "test.csv"

        max_negative: Optional[int]
        max_total: Optional[int]
        if full_eval:
            max_negative = None
            max_total = None
        else:
            max_negative = int(args.max_negative_per_split) if int(args.max_negative_per_split) > 0 else None
            max_total = int(args.max_eval_per_split) if int(args.max_eval_per_split or 0) > 0 else None

        val_session_ids, val_labels_raw, val_scores_raw, lm_val_latency = score_lm_from_csv_to_arrays(
            vqgan,
            lm,
            csv_path=val_csv,
            target_user=user_id,
            window_size_sec=float(window_size),
            target_width=int(target_width),
            token_batch_size=int(args.token_batch_size),
            device=device,
            use_amp=bool(args.use_amp),
            max_negative=max_negative,
            max_total=max_total,
        )
        test_session_ids, test_labels_raw, test_scores_raw, lm_test_latency = score_lm_from_csv_to_arrays(
            vqgan,
            lm,
            csv_path=test_csv,
            target_user=user_id,
            window_size_sec=float(window_size),
            target_width=int(target_width),
            token_batch_size=int(args.token_batch_size),
            device=device,
            use_amp=bool(args.use_amp),
            max_negative=max_negative,
            max_total=max_total,
        )

        val_labels = val_labels_raw.astype(np.int32, copy=False)
        val_scores = val_scores_raw.astype(np.float64, copy=False)
        test_labels = test_labels_raw.astype(np.int32, copy=False)
        test_scores = test_scores_raw.astype(np.float64, copy=False)

        selected_thr: float
        if strategy in {"k_session_frr", "interrupt_window_frr"}:
            selected_thr, lmseq_val_metrics = select_threshold_by_vote_window_frr_from_arrays(
                val_session_ids,
                val_labels_raw,
                val_scores_raw,
                window_size=int(vote_window_size),
                min_rejects=int(vote_min_rejects),
                target_window_frr=float(getattr(args, "target_window_frr", 0.0) or 0.0),
            )
            lm_val_metrics = compute_metrics(val_labels, val_scores, threshold=float(selected_thr))
            lm_test_metrics = compute_metrics(test_labels, test_scores, threshold=float(selected_thr))
        else:
            lm_val_metrics = compute_metrics(
                val_labels,
                val_scores,
                threshold=None,
                threshold_strategy=args.threshold_strategy,
                target_far=args.target_far,
                target_frr=args.target_frr,
            )
            selected_thr = float(lm_val_metrics.get("threshold", 0.0))
            lm_test_metrics = compute_metrics(test_labels, test_scores, threshold=float(selected_thr))

        if vote_enabled:
            lm_threshold = float(selected_thr)
            if lmseq_val_metrics is None:
                lmseq_val_metrics = compute_vote_reject_window_metrics_from_arrays(
                    val_session_ids,
                    val_labels_raw,
                    val_scores_raw,
                    threshold=lm_threshold,
                    window_size=int(vote_window_size),
                    min_rejects=int(vote_min_rejects),
                )
            lmseq_test_metrics = compute_vote_reject_window_metrics_from_arrays(
                test_session_ids,
                test_labels_raw,
                test_scores_raw,
                threshold=lm_threshold,
                window_size=int(vote_window_size),
                min_rejects=int(vote_min_rejects),
            )

            def _mean_window_to_sec(mean_window: float) -> float:
                mean_window = float(mean_window)
                if mean_window <= 0.0:
                    return 0.0
                return float(window_size + (mean_window - 1.0) * stride_sec)

            lmseq_val_metrics["genuine_mean_first_interrupt_sec"] = _mean_window_to_sec(
                float(lmseq_val_metrics.get("genuine_mean_first_interrupt_window", 0.0))
            )
            lmseq_val_metrics["impostor_mean_first_interrupt_sec"] = _mean_window_to_sec(
                float(lmseq_val_metrics.get("impostor_mean_first_interrupt_window", 0.0))
            )
            if lmseq_test_metrics is not None:
                lmseq_test_metrics["genuine_mean_first_interrupt_sec"] = _mean_window_to_sec(
                    float(lmseq_test_metrics.get("genuine_mean_first_interrupt_window", 0.0))
                )
                lmseq_test_metrics["impostor_mean_first_interrupt_sec"] = _mean_window_to_sec(
                    float(lmseq_test_metrics.get("impostor_mean_first_interrupt_window", 0.0))
                )
            lmseq_val_latency = float(lm_val_latency)
            lmseq_test_latency = float(lm_test_latency)
    else:
        lm_val_metrics, lm_val_latency = evaluate_token_lm(
            lm,
            val_tokens,
            val_y,
            batch_size=int(args.lm_eval_batch_size),
            device=device,
            use_amp=args.use_amp,
            threshold=None,
            threshold_strategy=args.threshold_strategy,
            target_far=args.target_far,
            target_frr=args.target_frr,
        )
        lm_test_metrics, lm_test_latency = evaluate_token_lm(
            lm,
            test_tokens,
            test_y,
            batch_size=int(args.lm_eval_batch_size),
            device=device,
            use_amp=args.use_amp,
            threshold=float(lm_val_metrics.get("threshold", 0.0)),
            threshold_strategy="eer",  # fixed threshold passed above
        )

    stages: List[Tuple[str, Dict[str, float], float]] = [
        ("vqgan-val", vqgan_val_metrics, vqgan_val_latency),
        ("vqgan-test", vqgan_test_metrics, vqgan_test_latency),
        ("lm-val", lm_val_metrics, lm_val_latency),
        ("lm-test", lm_test_metrics, lm_test_latency),
    ]
    if lmseq_val_metrics is not None and lmseq_val_latency is not None:
        stages.append(("lock-val", lmseq_val_metrics, float(lmseq_val_latency)))
    if lmseq_test_metrics is not None and lmseq_test_latency is not None:
        stages.append(("lock-test", lmseq_test_metrics, float(lmseq_test_latency)))

    for stage, metrics, latency in stages:
        stage_thr_strategy = "eer" if stage.startswith("vqgan") else str(args.threshold_strategy)
        stage_epoch = int(vqgan_trained_epochs) if stage.startswith("vqgan") else int(lm_trained_epochs)
        payload = {
            "stage": stage,
            "user": user_id,
            "window": window_size,
            "epoch": stage_epoch,
            "metrics": metrics,
            "latency": latency,
            "threshold_strategy": stage_thr_strategy,
            "target_far": args.target_far,
            "target_frr": args.target_frr,
            "target_window_frr": float(getattr(args, "target_window_frr", 0.0) or 0.0),
            "k_rejects": int(k_rejects_effective),
            "vote_window_size": int(vote_window_size) if vote_enabled else None,
            "vote_min_rejects": int(vote_min_rejects) if vote_enabled else None,
            "interrupt_after_sec": None if interrupt_after_sec is None else float(interrupt_after_sec),
            "interrupt_after_sec_base": None if interrupt_after_sec_base is None else float(interrupt_after_sec_base),
            "interrupt_after_sec_scale": float(interrupt_after_sec_scale or 0.0),
            "interrupt_min_k": int(interrupt_min_k),
            "interrupt_stride_sec": float(stride_sec),
            "interrupt_time_sec": float(interrupt_time_sec),
        }
        save_jsonl(json_log_path, payload)
        save_text_log(text_log_path, payload)

    return {
        "user": user_id,
        "window": window_size,
        "target_width": target_width,
        "interrupt_rule": "vote" if vote_enabled else "k",
        "vote_window_size": int(vote_window_size) if vote_enabled else 0,
        "vote_min_rejects": int(vote_min_rejects) if vote_enabled else 0,
        "interrupt_after_sec": None if interrupt_after_sec is None else float(interrupt_after_sec),
        "interrupt_after_sec_base": None if interrupt_after_sec_base is None else float(interrupt_after_sec_base),
        "interrupt_after_sec_scale": float(interrupt_after_sec_scale or 0.0),
        "interrupt_min_k": int(interrupt_min_k),
        "interrupt_stride_sec": float(stride_sec),
        "interrupt_time_sec": float(interrupt_time_sec),
        "vqgan": {
            "val": vqgan_val_metrics,
            "test": vqgan_test_metrics,
            "checkpoint": str(vqgan_ckpt_path),
            "trained_epochs": int(vqgan_trained_epochs),
            "best_epoch": int(vqgan_best_epoch),
        },
        "lm": {
            "config": lm_cfg.__dict__,
            "val": lm_val_metrics,
            "test": lm_test_metrics,
            "checkpoint": str(lm_ckpt_path),
            "trained_epochs": int(lm_trained_epochs),
            "best_epoch": int(lm_best_epoch),
        },
        "lmseq": {
            "k_rejects": int(k_rejects_effective),
            "vote_window_size": int(vote_window_size) if vote_enabled else 0,
            "vote_min_rejects": int(vote_min_rejects) if vote_enabled else 0,
            "val": lmseq_val_metrics,
            "test": lmseq_test_metrics,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMOG VQGAN -> Token -> Transformer(LM) window sweep")
    parser.add_argument("--dataset-path", type=str, default=default_dataset_root())
    parser.add_argument("--users", nargs="*", help="仅训练指定用户 id，默认遍历全部目录。")
    parser.add_argument("--window-sizes", nargs="*", type=float, default=list(WINDOW_SIZES))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--target-width", type=int, default=50, help="重采样后的时间轴长度；设为 0 则使用该 window 的原始点数 (t*100Hz)")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--prep-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Eval caps (important for huge CSVs)
    parser.add_argument("--max-negative-per-split", type=int, default=20000)
    parser.add_argument("--max-eval-per-split", type=int, default=40000)
    parser.add_argument("--max-train-per-user", type=int, default=None, help="可选的训练集子采样上限")
    parser.add_argument("--full-scan-eval", action="store_true")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help=(
            "使用完整 val/test CSV 做评估（不做窗口数截断；可能很慢）。"
            "开启后会跳过把 val/test 窗口张量与 token cache 物化到内存/磁盘。"
        ),
    )

    # VQGAN config
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-codebook-vectors", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--no-nonlocal", action="store_true")
    parser.add_argument("--q-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-rec-loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--score-metric", choices=["mse", "l1"], default="mse")
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--vqgan-epochs", type=int, default=1, help="VQGAN 最大训练 epoch 数（会受 --early-stop-patience 影响提前停止）")
    parser.add_argument("--vqgan-val-interval", type=int, default=0)
    parser.add_argument("--vqgan-lr", type=float, default=2.5e-4)
    parser.add_argument("--vqgan-weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--reuse-vqgan", action="store_true", help="如果存在 checkpoint 则跳过 VQGAN 训练")

    # Tokenization
    parser.add_argument("--token-cache-dir", type=str, default=default_token_cache_dir())
    parser.add_argument("--token-batch-size", type=int, default=512)

    # Token LM config
    parser.add_argument("--lm-epochs", type=int, default=3, help="LM 最大训练 epoch 数（会受 --early-stop-patience 影响提前停止）")
    parser.add_argument("--lm-batch-size", type=int, default=256)
    parser.add_argument("--lm-eval-batch-size", type=int, default=2048)
    parser.add_argument("--lm-lr", type=float, default=3e-4)
    parser.add_argument("--lm-weight-decay", type=float, default=0.01)
    parser.add_argument("--lm-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lm-n-layer", type=int, default=6)
    parser.add_argument("--lm-n-head", type=int, default=6)
    parser.add_argument("--lm-n-embd", type=int, default=384)
    parser.add_argument("--lm-dropout", type=float, default=0.1)
    parser.add_argument("--reuse-lm", action="store_true", help="如果存在 checkpoint 则跳过 LM 训练")

    # Early stopping
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="早停：验证集性能连续 P 次不提升则停止（0=关闭；同时作用于 VQGAN/LM）。",
    )

    # Threshold selection (on val), applied to test.
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        default="eer",
        choices=["eer", "f1", "far", "frr", "interrupt_window_frr", "k_session_frr"],
        help=(
            "在 val 上选择阈值的方法："
            "eer(默认) / f1(最大化F1) / far(约束FAR<=target_far并最大化TPR) / "
            "frr(约束FRR<=target_frr并尽量降低FAR) / "
            "interrupt_window_frr(基于'投票打断规则'的窗口级 false interrupt rate(FRR) 约束选择阈值；会排除每个 session 最后 N-1 个窗口)"
        ),
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.05,
        help="threshold-strategy=far 时使用的目标 FAR (0~1)，例如 0.05 表示 FAR=0.05",
    )
    parser.add_argument(
        "--target-frr",
        type=float,
        default=0.001,
        help="threshold-strategy=frr 时使用的目标 FRR (0~1)，例如 0.001 表示 FRR=0.001（更偏可用性）",
    )
    parser.add_argument(
        "--k-rejects",
        type=int,
        default=0,
        help="连续拒绝 K 次才触发一次打断（0=关闭；>0 会额外输出 lmseq-val/lmseq-test 序列指标）",
    )
    parser.add_argument(
        "--vote-window-size",
        type=int,
        default=0,
        help="投票窗口长度 N（最近 N 个窗口）；与 --k-rejects/--interrupt-after-sec* 互斥，0=关闭。",
    )
    parser.add_argument(
        "--vote-min-rejects",
        type=int,
        default=0,
        help="投票触发阈值 M：最近 N 个窗口中 reject>=M 触发 interrupt（需配合 --vote-window-size；0=关闭）。",
    )
    parser.add_argument(
        "--interrupt-after-sec",
        type=float,
        default=None,
        help=(
            "固定打断时间（秒）。会按 stride=window*(1-overlap) 自动换算 K=ceil(T/stride)，"
            "并用于 lmseq 以及 threshold-strategy=k_session_frr 的阈值选择。"
            "与 --k-rejects 互斥。"
        ),
    )
    parser.add_argument(
        "--interrupt-after-sec-base",
        type=float,
        default=None,
        help=(
            "动态打断时间（秒）的 base：T = base + scale*window_size；"
            "会按 stride=window*(1-overlap) 自动换算 K=ceil(T/stride)，并用于 lmseq 与 k_session_frr。"
            "与 --k-rejects / --interrupt-after-sec 互斥。"
        ),
    )
    parser.add_argument(
        "--interrupt-after-sec-scale",
        type=float,
        default=0.0,
        help="动态打断时间的 scale：T = base + scale*window_size（需配合 --interrupt-after-sec-base）。",
    )
    parser.add_argument(
        "--interrupt-min-k",
        type=int,
        default=0,
        help="可选：对换算出来的 K 再做下限约束（K=max(K, interrupt_min_k)），用于进一步降低误报触发。",
    )
    parser.add_argument(
        "--target-window-frr",
        type=float,
        default=0.0,
        help="threshold-strategy=interrupt_window_frr 时使用：窗口级 false interrupt rate(FRR) 上限（0~1，建议先尝试 0.10）。",
    )
    # Backward-compatible alias (deprecated): maps to --target-window-frr if provided.
    parser.add_argument(
        "--target-session-frr",
        type=float,
        default=None,
        help="(deprecated) 请改用 --target-window-frr；保留仅用于兼容旧调用。",
    )
    parser.add_argument(
        "--subject-scan-rows",
        type=int,
        default=0,
        help="可选：每个 split 读取前 N 行，打印出现的 subject 列取值用于校验(0 关闭)。",
    )

    # Runtime
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)

    parser.add_argument("--output-dir", type=str, default=default_results_dir())
    parser.add_argument("--log-dir", type=str, default=str(Path(default_results_dir()) / "experiment_logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "target_session_frr", None) is not None and float(args.target_window_frr or 0.0) == 0.0:
        args.target_window_frr = float(args.target_session_frr)
    _, metrics_txt = setup_logging(Path(args.log_dir))
    json_log_path = Path(args.log_dir) / "hmog_vqgan_token_transformer.jsonl"
    text_log_path = Path(metrics_txt)

    torch.set_num_threads(int(args.cpu_threads))
    torch.backends.cudnn.benchmark = True
    set_seed(int(args.seed), cuda=torch.cuda.is_available())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("[START] device=%s cuda_available=%s", device, torch.cuda.is_available())

    vote_window_size = int(getattr(args, "vote_window_size", 0) or 0)
    vote_min_rejects = int(getattr(args, "vote_min_rejects", 0) or 0)
    vote_enabled = vote_window_size > 0 or vote_min_rejects > 0
    if vote_enabled:
        if vote_window_size <= 0 or vote_min_rejects <= 0:
            raise ValueError("--vote-window-size and --vote-min-rejects must both be > 0 when vote is enabled.")
        if vote_min_rejects > vote_window_size:
            raise ValueError("--vote-min-rejects must be <= --vote-window-size.")
        if int(getattr(args, "k_rejects", 0)) > 0:
            raise ValueError("Vote mode is mutually exclusive with --k-rejects.")
        if getattr(args, "interrupt_after_sec", None) is not None and float(args.interrupt_after_sec) > 0.0:
            raise ValueError("Vote mode is mutually exclusive with --interrupt-after-sec.")
        if getattr(args, "interrupt_after_sec_base", None) is not None:
            raise ValueError("Vote mode is mutually exclusive with --interrupt-after-sec-base/--interrupt-after-sec-scale.")

    if getattr(args, "interrupt_after_sec", None) is not None and float(args.interrupt_after_sec) > 0.0:
        if int(getattr(args, "k_rejects", 0)) > 0:
            raise ValueError("Use either --k-rejects or --interrupt-after-sec, not both.")
        if getattr(args, "interrupt_after_sec_base", None) is not None:
            raise ValueError("Use either --interrupt-after-sec or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both.")

    if getattr(args, "interrupt_after_sec_base", None) is not None:
        if int(getattr(args, "k_rejects", 0)) > 0:
            raise ValueError("Use either --k-rejects or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both.")

    base_path = Path(args.dataset_path)
    users = args.users or list_available_users(base_path)
    if not users:
        raise ValueError(f"No users found under {base_path}")

    all_results: List[Dict] = []
    for user_id in users:
        for ws in args.window_sizes:
            # Avoid identical randomness across windows.
            set_seed(int(args.seed) + int(float(ws) * 10), cuda=device.type.startswith("cuda"))
            res = run_single_window(
                args,
                user_id=user_id,
                window_size=float(ws),
                device=device,
                json_log_path=json_log_path,
                text_log_path=text_log_path,
            )
            all_results.append(res)
            logger.info(
                "[DONE] user=%s ws=%.1f vqgan_auc=%.4f lm_auc=%.4f",
                user_id,
                float(ws),
                float(res["vqgan"]["test"]["auc"]),
                float(res["lm"]["test"]["auc"]),
            )

    summary_path = Path(args.log_dir) / "hmog_vqgan_token_transformer_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("[SUMMARY] saved %s", summary_path)

    # Choose a "best" window/policy per user.
    # Objective (window-level, vote-based lock):
    #   1) minimize false interrupts on genuine windows (frr)
    #   2) minimize misses on impostor windows (far)
    #   3) minimize time-to-first-lock on impostors
    def _pick_best_policy(results: List[Dict]) -> Optional[Dict]:
        candidates: List[Dict] = []
        for res in results:
            lmseq_val = ((res.get("lmseq") or {}).get("val") or None)
            if lmseq_val is None:
                continue
            candidates.append(res)
        if not candidates:
            return None

        def _key(res: Dict) -> Tuple[float, float, float, float]:
            lmseq_val = (res.get("lmseq") or {}).get("val") or {}
            impostor_mean_sec = float(lmseq_val.get("impostor_mean_first_interrupt_sec", 0.0) or 0.0)
            if float(lmseq_val.get("impostor_first_interrupt_n", 0.0) or 0.0) <= 0.0:
                impostor_mean_sec = float("inf")
            window_frr = float(lmseq_val.get("frr", 1.0))
            window_far = float(lmseq_val.get("far", 1.0))
            interrupt_time = float(res.get("interrupt_time_sec", 0.0) or 0.0)
            return (window_frr, window_far, impostor_mean_sec, interrupt_time)

        return sorted(candidates, key=_key)[0]

    best_by_user: Dict[str, Dict] = {}
    for user_id in users:
        picked = _pick_best_policy([r for r in all_results if r.get("user") == user_id])
        if not picked:
            continue
        best_by_user[user_id] = {
            "user": picked.get("user"),
            "window": float(picked.get("window", 0.0)),
            "target_width": int(picked.get("target_width", 0)),
            "overlap": float(args.overlap),
            "threshold_strategy": str(args.threshold_strategy),
            "target_window_frr": float(getattr(args, "target_window_frr", 0.0) or 0.0),
            "threshold": float((((picked.get("lm") or {}).get("val") or {}).get("threshold") or 0.0)),
            "k_rejects": int(((picked.get("lmseq") or {}).get("k_rejects") or 0)),
            "vote_window_size": int(((picked.get("lmseq") or {}).get("vote_window_size") or 0)),
            "vote_min_rejects": int(((picked.get("lmseq") or {}).get("vote_min_rejects") or 0)),
            "interrupt_time_sec": float(picked.get("interrupt_time_sec", 0.0) or 0.0),
            "interrupt_after_sec": picked.get("interrupt_after_sec"),
            "interrupt_after_sec_base": picked.get("interrupt_after_sec_base"),
            "interrupt_after_sec_scale": picked.get("interrupt_after_sec_scale"),
            "interrupt_min_k": picked.get("interrupt_min_k"),
            "lmseq_val": (picked.get("lmseq") or {}).get("val"),
            "lmseq_test": (picked.get("lmseq") or {}).get("test"),
            "vqgan_checkpoint": ((picked.get("vqgan") or {}).get("checkpoint")),
            "lm_checkpoint": ((picked.get("lm") or {}).get("checkpoint")),
        }

    if best_by_user:
        best_path = Path(args.log_dir) / "best_lock_policy.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(best_by_user, f, indent=2, ensure_ascii=False)
        logger.info("[BEST] saved %s", best_path)

    # Print a compact table for quick comparison.
    print("\n=== Window sweep summary (test) ===")
    has_lmseq = any(((r.get("lmseq") or {}).get("test") is not None) for r in all_results)
    if has_lmseq:
        print(
            "window\tK\tvote_N\tvote_M\tinterrupt_time_sec\tvqgan_auc\tvqgan_eer\tlm_auc\tlm_eer\t"
            "lm_thr\tlm_test_frr\tlm_test_far\t"
            "lmseq_test_frr\tlmseq_test_far\tlmseq_test_err\tlmseq_test_impostor_mean_first_interrupt_sec"
        )
    else:
        print("window\tvqgan_auc\tvqgan_eer\tlm_auc\tlm_eer")
    for res in sorted(all_results, key=lambda x: float(x["window"])):
        if has_lmseq:
            lm_thr = float(((res.get("lm") or {}).get("val") or {}).get("threshold", 0.0))
            lm_test_frr = float(((res.get("lm") or {}).get("test") or {}).get("frr", 0.0))
            lm_test_far = float(((res.get("lm") or {}).get("test") or {}).get("far", 0.0))
            lmseq_test = (res.get("lmseq") or {}).get("test") or {}
            k_eff = int(((res.get("lmseq") or {}).get("k_rejects") or 0))
            v_n = int(((res.get("lmseq") or {}).get("vote_window_size") or 0))
            v_m = int(((res.get("lmseq") or {}).get("vote_min_rejects") or 0))
            t_eff = float(res.get("interrupt_time_sec", 0.0) or 0.0)
            lmseq_test_frr = float(lmseq_test.get("frr", 0.0))
            lmseq_test_far = float(lmseq_test.get("far", 0.0))
            lmseq_test_err = float(lmseq_test.get("err", 0.0))
            lmseq_test_imp_sec = float(lmseq_test.get("impostor_mean_first_interrupt_sec", 0.0) or 0.0)
            print(
                f"{res['window']:.1f}\t{k_eff}\t{v_n}\t{v_m}\t{t_eff:.3f}\t"
                f"{res['vqgan']['test']['auc']:.4f}\t{res['vqgan']['test']['eer']:.4f}\t"
                f"{res['lm']['test']['auc']:.4f}\t{res['lm']['test']['eer']:.4f}\t"
                f"{lm_thr:.6f}\t{lm_test_frr:.6f}\t{lm_test_far:.6f}\t"
                f"{lmseq_test_frr:.6f}\t{lmseq_test_far:.6f}\t{lmseq_test_err:.6f}\t{lmseq_test_imp_sec:.3f}"
            )
        else:
            print(
                f"{res['window']:.1f}\t"
                f"{res['vqgan']['test']['auc']:.4f}\t{res['vqgan']['test']['eer']:.4f}\t"
                f"{res['lm']['test']['auc']:.4f}\t{res['lm']['test']['eer']:.4f}"
            )


if __name__ == "__main__":
    main()
