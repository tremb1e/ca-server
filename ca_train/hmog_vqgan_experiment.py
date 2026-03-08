import argparse
import json
import logging
import os
import random
import time
import multiprocessing as mp
import resource
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerator import (
    autocast_context,
    device_count,
    make_grad_scaler,
    normalize_device,
    resolve_torch_device,
    seed_all,
    set_visible_device,
)
from hmog_data import (
    DEFAULT_OVERLAP,
    WINDOW_SIZES,
    WindowedHMOGDataset,
    list_available_users,
    prepare_user_datasets,
    precompute_all_user_windows,
)
from vqgan import VQGAN


def set_seed(seed: int, device_type: Optional[str] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    seed_all(seed, device_type=device_type)


def setup_logging(log_dir: Path) -> Tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hmog_vqgan.log"
    metrics_txt = log_dir / "hmog_metrics.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file, metrics_txt


def build_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = WindowedHMOGDataset(train_x, train_y)
    val_dataset = WindowedHMOGDataset(val_x, val_y)
    test_dataset = WindowedHMOGDataset(test_x, test_y)

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def _safe_worker_count(requested: int) -> int:
    """Clamp dataloader workers to avoid hitting open file limits."""
    if requested <= 0:
        return 0
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        # 每个 worker 可能需要若干管道/文件句柄，这里粗略按 64 做预算，留出 40% 富余
        budget = int(soft_limit * 0.6) if soft_limit > 0 else requested
        safe = max(1, min(requested, max(1, budget // 64)))
        return safe
    except Exception:
        return min(requested, 8)


def reconstruction_step(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    *,
    q_loss_weight: float = 1.0,
    input_noise_std: float = 0.0,
    rec_loss_metric: str = "l1",
):
    batch = batch.to(device, non_blocking=True)
    if input_noise_std and input_noise_std > 0:
        batch = batch + torch.randn_like(batch) * float(input_noise_std)
    with autocast_context(device, enabled=bool(use_amp)):
        decoded, _, q_loss = model(batch)
        if rec_loss_metric == "mse":
            rec_loss = torch.mean((batch - decoded) ** 2)
        else:
            rec_loss = torch.mean(torch.abs(batch - decoded))
        q_loss = torch.mean(q_loss)
        loss = rec_loss + float(q_loss_weight) * q_loss
    return loss, rec_loss.detach(), q_loss.detach()


def _eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Return (eer, eer_threshold) where threshold is on the `scores` scale."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = float(interp1d(fpr, thresholds)(eer))
    if np.isnan(eer_threshold):
        eer_threshold = float(thresholds[0]) if len(thresholds) else 0.0
    return float(eer), eer_threshold


def _far_frr_f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    preds = (scores >= threshold).astype(np.int32)
    neg = labels == 0
    pos = labels == 1
    far = float(((preds == 1) & neg).sum() / max(int(neg.sum()), 1))
    frr = float(((preds == 0) & pos).sum() / max(int(pos.sum()), 1))
    tp = float(((preds == 1) & pos).sum())
    fp = float(((preds == 1) & neg).sum())
    fn = float(((preds == 0) & pos).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return far, frr, float(f1)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: Optional[float] = None) -> Dict[str, float]:
    # 中文注释：使用 ROC 曲线求解 EER；阈值在 val 上确定后应固定用于 test 的 FAR/FRR 评估。
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    eer, eer_threshold = _eer_and_threshold(labels, scores)
    selected_threshold = eer_threshold if threshold is None else float(threshold)
    far, frr, f1 = _far_frr_f1_at_threshold(labels, scores, selected_threshold)

    return {
        "auc": float(auc),
        "far": far,
        "frr": frr,
        "eer": float(eer),
        "f1": f1,
        "threshold": float(selected_threshold),
        "eer_threshold": float(eer_threshold),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    *,
    threshold: Optional[float] = None,
    score_metric: str = "mse",
) -> Tuple[Dict[str, float], float]:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[float] = []
    all_errors: List[float] = []
    start = time.time()
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device, non_blocking=True)
            labels = labels.cpu().numpy()
            with autocast_context(device, enabled=bool(use_amp)):
                decoded, _, _ = model(batch)
                # 传感器重建误差越小代表越接近合法用户
                if score_metric == "l1":
                    errors = torch.mean(torch.abs(batch - decoded), dim=(1, 2, 3))
                else:
                    errors = torch.mean((batch - decoded) ** 2, dim=(1, 2, 3))
            scores = (-errors).cpu().numpy()  # 分数越大越可信
            all_errors.append(errors.detach().cpu().numpy())
            all_scores.append(scores)
            all_labels.append(labels)
    end = time.time()
    latency = float((end - start) / max(len(loader.dataset), 1))
    scores_np = np.concatenate(all_scores, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    errors_np = np.concatenate(all_errors, axis=0)
    metrics = compute_metrics(labels_np, scores_np, threshold=threshold)
    # Extra debugging stats for anomaly detection directionality.
    if labels_np.size:
        pos_mask = labels_np == 1
        neg_mask = labels_np == 0
        if pos_mask.any():
            metrics["pos_error_mean"] = float(errors_np[pos_mask].mean())
        if neg_mask.any():
            metrics["neg_error_mean"] = float(errors_np[neg_mask].mean())
    return metrics, latency


def save_jsonl(log_path: Path, payload: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_text_log(text_path: Path, payload: Dict) -> None:
    """
    将指标写入可读性好的 txt 日志，所有指标保持小数形式（例如 FAR=0.12 而不是 12%）
    """
    text_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{ts}\tstage={payload.get('stage','')}\tuser={payload.get('user','')}"
        f"\twindow={payload.get('window','')}\tepoch={payload.get('epoch','')}"
        f"\tauc={payload.get('metrics',{}).get('auc',0):.6f}"
        f"\tfar={payload.get('metrics',{}).get('far',0):.6f}"
        f"\tfrr={payload.get('metrics',{}).get('frr',0):.6f}"
        f"\teer={payload.get('metrics',{}).get('eer',0):.6f}"
        f"\tf1={payload.get('metrics',{}).get('f1',0):.6f}"
        f"\tthreshold={payload.get('metrics',{}).get('threshold',0):.6f}"
        f"\tpos_err_mean={payload.get('metrics',{}).get('pos_error_mean',0):.6f}"
        f"\tneg_err_mean={payload.get('metrics',{}).get('neg_error_mean',0):.6f}"
        f"\tlatency={payload.get('latency',0):.6f}"
    )
    with text_path.open("a") as f:
        f.write(line + "\n")


def _check_split_class_separation(
    split_x: np.ndarray,
    split_y: np.ndarray,
    *,
    split_name: str,
    user_id: str,
    window_size: float,
    seed: int,
    eps: float = 1e-10,
) -> None:
    """
    Quick guardrail: verify positive/negative windows are not numerically identical.

    If this check fails, training will inevitably collapse to AUC≈0.5 with threshold=inf.
    """
    if split_x.size == 0 or split_y.size == 0:
        return
    pos = split_x[split_y == 1]
    neg = split_x[split_y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return

    rng = np.random.default_rng(int(seed))
    k = min(256, len(pos), len(neg))
    idx_pos = rng.choice(len(pos), size=k, replace=False)
    idx_neg = rng.choice(len(neg), size=k, replace=False)
    pair_mse = float(np.mean((pos[idx_pos] - neg[idx_neg]) ** 2))
    if (not np.isfinite(pair_mse)) or pair_mse <= float(eps):
        raise ValueError(
            f"[DATA] {split_name} split for user={user_id} ws={window_size:.1f} has degenerate class separation: "
            f"sampled pos/neg mean_pair_mse={pair_mse}. "
            f"Check window construction/loader for aliasing or mislabeled samples."
        )
    logging.info(
        "[DATA] %s user=%s ws=%.1f sampled_pos_neg_mse=%.6f",
        split_name,
        user_id,
        window_size,
        pair_mse,
    )


def train_single_window(
    args: argparse.Namespace,
    user_id: str,
    window_size: float,
    device: torch.device,
    num_workers: int,
    epochs: int,
    json_log_path: Path,
    text_log_path: Path,
    cache: Dict[str, Dict[str, List[np.ndarray]]] = None,
    window_cache_dir: Path = None,
    all_users: Sequence[str] = None,
    base_path: Path = None,
) -> Dict:
    json_log_path = Path(json_log_path)
    text_log_path = Path(text_log_path)
    loader_workers = _safe_worker_count(num_workers)
    if loader_workers < num_workers:
        logging.info(
            f"[DL] Clamp dataloader workers from {num_workers} to {loader_workers} "
            f"to avoid RLIMIT_NOFILE exhaustion"
        )
    prep_start = time.time()
    train_x, train_y, val_x, val_y, test_x, test_y = prepare_user_datasets(
        target_user=user_id,
        window_size_sec=window_size,
        cache=cache,
        overlap=args.overlap,
        target_width=args.target_width if args.target_width > 0 else int(round(window_size * 100)),
        prep_workers=args.prep_workers,
        max_negative_per_split=args.max_negative_per_split,
        max_eval_per_split=args.max_eval_per_split,
        negative_users=all_users,
        window_cache_dir=window_cache_dir,
        base_path=base_path,
        full_scan_eval=args.full_scan_eval,
        seed=args.seed,
    )

    # Sanity checks: the task defines labels by `subject == target_user`.
    # - train must be genuine-only (all 1s)
    # - val/test must contain both genuine (1) and attackers (0)
    if train_y.size and not np.all(train_y == 1):
        raise ValueError(
            f"[DATA] train split for user={user_id} ws={window_size:.1f} must be genuine-only "
            f"(subject=={user_id} -> label 1); got labels={np.unique(train_y)}"
        )
    for split_name, split_y in (("val", val_y), ("test", test_y)):
        uniq = np.unique(split_y)
        if split_y.size and len(uniq) < 2:
            raise ValueError(
                f"[DATA] {split_name} split for user={user_id} ws={window_size:.1f} must contain both classes "
                f"(subject=={user_id} -> 1, attackers -> 0); got labels={uniq}. "
                f"Check {args.dataset_path}/{window_size:.1f}/{user_id}/{split_name}.csv subject column."
            )
    _check_split_class_separation(
        val_x,
        val_y,
        split_name="val",
        user_id=user_id,
        window_size=window_size,
        seed=args.seed + 11,
    )
    _check_split_class_separation(
        test_x,
        test_y,
        split_name="test",
        user_id=user_id,
        window_size=window_size,
        seed=args.seed + 23,
    )
    prep_dur = time.time() - prep_start
    # 预处理阶段也输出日志，便于观察长时间无输出时的进度（数据量 = 正样本 + 负样本）
    logging.info(
        f"[PREP] user={user_id} window={window_size:.1f} "
        f"train={len(train_x)} val_pos={(val_y==1).sum()} val_neg={(val_y==0).sum()} "
        f"test_pos={(test_y==1).sum()} test_neg={(test_y==0).sum()} prep_time={prep_dur:.2f}s"
    )

    if args.max_train_per_user and len(train_x) > args.max_train_per_user:
        idx = np.random.choice(len(train_x), size=args.max_train_per_user, replace=False)
        train_x = train_x[idx]
        train_y = train_y[idx]

    loaders = build_loaders(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        args.batch_size,
        loader_workers,
        pin_memory=(device.type != "cpu"),
    )
    train_loader, val_loader, test_loader = loaders

    # Let the model know the expected spatial size (used by the decoder upsampling plan).
    if train_x.size:
        args.input_height = int(train_x.shape[2])
        args.input_width = int(train_x.shape[3])
    else:
        args.input_height = 12
        args.input_width = int(args.target_width) if args.target_width > 0 else int(round(window_size * 100))

    # VQGAN blocks read `use_nonlocal`; CLI exposes `--no-nonlocal` for convenience.
    args.use_nonlocal = not getattr(args, "no_nonlocal", False)

    model = VQGAN(args).to(device)
    use_dp = (device.type == "cuda") and (torch.cuda.device_count() > 1) and (not args.no_data_parallel)
    if use_dp:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
    )
    scaler = make_grad_scaler(device, enabled=bool(args.use_amp))

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"User {user_id} | ws {window_size:.1f} | epoch {epoch+1}/{epochs}")
        for batch, _ in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, rec_loss, q_loss = reconstruction_step(
                model,
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rec=f"{rec_loss.item():.4f}",
                q=f"{q_loss.item():.4f}",
            )

        if (epoch + 1) % args.val_interval == 0:
            val_metrics, val_latency = evaluate_model(
                model, val_loader, device, args.use_amp, score_metric=args.score_metric
            )
            payload = {
                "stage": "val",
                "user": user_id,
                "window": window_size,
                "epoch": epoch + 1,
                "metrics": val_metrics,
                "latency": val_latency,
            }
            save_jsonl(json_log_path, payload)
            save_text_log(text_log_path, payload)
            logging.info(f"[VAL] user={user_id} ws={window_size:.1f} epoch={epoch+1} metrics={val_metrics} latency={val_latency:.6f}")

    # Final evaluation: fix the decision threshold on val, then report test FAR/FRR at that threshold.
    val_metrics, val_latency = evaluate_model(model, val_loader, device, args.use_amp, score_metric=args.score_metric)
    test_metrics, test_latency = evaluate_model(
        model,
        test_loader,
        device,
        args.use_amp,
        threshold=float(val_metrics.get("threshold", 0.0)),
        score_metric=args.score_metric,
    )

    payload = {
        "stage": "test",
        "user": user_id,
        "window": window_size,
        "epoch": epochs,
        "metrics": test_metrics,
        "latency": test_latency,
    }
    save_jsonl(json_log_path, payload)
    save_text_log(text_log_path, payload)
    logging.info(f"[TEST] user={user_id} ws={window_size:.1f} metrics={test_metrics} latency={test_latency:.6f}")

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"vqgan_user_{user_id}_ws_{window_size:.1f}.pt"
    torch.save(model_to_save.state_dict(), ckpt_path)

    return {
        "user": user_id,
        "window": window_size,
        "val": val_metrics,
        "test": test_metrics,
        "val_latency": val_latency,
        "test_latency": test_latency,
        "checkpoint": str(ckpt_path),
    }


def _train_worker_entry(
    args_dict: Dict,
    user_id: str,
    window_size: float,
    epochs: int,
    gpu_id: Optional[int],
    json_log_path: str,
    text_log_path: str,
    users: Sequence[str],
) -> Dict:
    args = argparse.Namespace(**args_dict)
    if not logging.getLogger().handlers:
        setup_logging(Path(args.log_dir))

    backend = str(getattr(args, "backend", "cpu"))
    if gpu_id is not None and backend in {"npu", "cuda"}:
        set_visible_device(backend, gpu_id)

    if gpu_id is None:
        device_str = normalize_device(args.device)
    else:
        device_str = f"{backend}:0"

    device = resolve_torch_device(device_str)
    npu_available = bool(getattr(torch, "npu", None) and torch.npu.is_available()) if hasattr(torch, "npu") else False
    logging.info(
        f"[WORKER] pid={os.getpid()} start_method={mp.get_start_method()} "
        f"device_id={gpu_id} device={device} cuda_available={torch.cuda.is_available()} npu_available={npu_available}"
    )
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    # 使用窗口大小扰动种子，避免完全相同的随机序列
    set_seed(args.seed + int(window_size * 10), device_type=device.type)

    base_path = Path(args.dataset_path)
    return train_single_window(
        args=args,
        cache=None,
        user_id=user_id,
        window_size=window_size,
        device=device,
        num_workers=args.num_workers,
        epochs=epochs,
        json_log_path=Path(json_log_path),
        text_log_path=Path(text_log_path),
        window_cache_dir=Path(args.window_cache_dir),
        all_users=list(users),
        base_path=base_path,
    )


def dispatch_training_tasks(
    args: argparse.Namespace,
    tasks: Sequence[Tuple[str, float]],
    epochs: int,
    log_path: Path,
    text_log_path: Path,
) -> Tuple[List[Dict], List[str]]:
    """Launch parallel training jobs for (user, window) pairs."""
    if not tasks:
        return [], []

    backend = str(getattr(args, "backend", "cpu"))
    available_gpus = args.gpu_ids
    if backend in {"npu", "cuda"}:
        if available_gpus is None:
            available_gpus = list(range(device_count(backend)))
        if not available_gpus:
            available_gpus = [None]
    else:
        available_gpus = [None]

    max_workers = args.max_parallel_train or len(available_gpus) or 1
    if available_gpus != [None]:
        max_workers = min(max_workers, len(available_gpus))
    max_workers = min(max_workers, len(tasks))

    args_dict = vars(args)
    job_specs: List[Tuple[str, float, Optional[int]]] = []
    for idx, (user_id, window_size) in enumerate(tasks):
        gpu_id = available_gpus[idx % len(available_gpus)]
        job_specs.append((user_id, window_size, gpu_id))

    user_list = list({user_id for user_id, _ in tasks})

    results: List[Dict] = []
    errors: List[str] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        future_map = {
            ex.submit(
                _train_worker_entry,
                args_dict,
                user_id,
                window_size,
                epochs,
                gpu_id,
                str(log_path),
                str(text_log_path),
                user_list,
            ): (user_id, window_size, gpu_id)
            for user_id, window_size, gpu_id in job_specs
        }

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="train-tasks"):
            job_user, ws, gpu_id = future_map[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                msg = f"user={job_user} ws={ws:.1f} gpu={gpu_id}: {exc}"
                errors.append(msg)
                logging.exception(f"[ERROR] {msg}")

    return results, errors


def run_experiments(args: argparse.Namespace, text_log_path: Path) -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已经设置过启动方式时直接跳过
        pass

    logging.info(f"[MP] start_method={mp.get_start_method()} mp_context_spawn_available={mp.get_all_start_methods()}")

    args.device = normalize_device(args.device)
    args.backend = "cpu" if args.device == "cpu" else str(args.device).split(":", 1)[0]
    device = resolve_torch_device(args.device)
    if device.type == "cpu":
        logging.warning("未检测到 NPU/CUDA，训练将退回到 CPU，可能很慢。")
    elif device.type == "npu":
        logging.info("[NPU] device_count=%d use_dp=%s", device_count("npu"), False)
    else:
        logging.info("[CUDA] device_count=%d use_dp=%s", torch.cuda.device_count(), (not args.no_data_parallel))

    # 仅设定 CPU 随机种子，避免在主进程提前初始化设备上下文影响后续 spawn
    set_seed(args.seed, device_type=None)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base_path = Path(args.dataset_path)
    users = args.users or list_available_users(base_path)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hmog_metrics.jsonl"
    best_windows: Dict[str, Dict] = {}

    logging.info(f"[START] users={users} window_sizes={args.window_sizes} batch={args.batch_size} "
                 f"num_workers={args.num_workers} prep_workers={args.prep_workers} "
                 f"amp={args.use_amp} device={device} cache_dir={args.window_cache_dir}")

    if not args.skip_cache_build:
        cache_jobs = precompute_all_user_windows(
            base_path=base_path,
            cache_dir=Path(args.window_cache_dir),
            users=users,
            window_sizes=args.window_sizes,
            overlap=args.overlap,
            target_width=args.target_width,
            prep_workers=args.prep_workers,
            process_workers=args.cache_processes,
        )
        logging.info(f"[CACHE] 已为 {len(cache_jobs)} 组 (user,window) 构建/检查缓存 -> {args.window_cache_dir}")
    else:
        logging.info("[CACHE] 已按参数跳过预处理缓存构建")

    sweep_tasks = [(user_id, ws) for user_id in users for ws in args.window_sizes]
    sweep_results, sweep_errors = dispatch_training_tasks(
        args=args,
        tasks=sweep_tasks,
        epochs=args.sweep_epochs,
        log_path=log_path,
        text_log_path=text_log_path,
    )

    missing_users: List[str] = []
    for user_id in users:
        user_results = [r for r in sweep_results if r and r.get("user") == user_id]
        if not user_results:
            logging.warning(f"[WARN] 未获得 user={user_id} 的 sweep 结果，跳过挑选最佳窗口")
            missing_users.append(user_id)
            continue
        best = max(user_results, key=lambda r: r["val"]["auc"])
        best_windows[user_id] = best
        logging.info(f"[SELECT] user={user_id} best_window={best['window']:.1f} val_auc={best['val']['auc']:.4f}")

    if missing_users:
        err_preview = " | ".join(sweep_errors[:3]) if sweep_errors else "no worker error details captured"
        raise RuntimeError(
            f"Sweep failed for users={missing_users}; first errors: {err_preview}. "
            f"See {log_dir / 'hmog_vqgan.log'} for full trace."
        )

    if args.final_epochs > args.sweep_epochs and best_windows:
        logging.info(f"[RETRAIN] 针对最佳窗口重新训练 {len(best_windows)} 位用户，每个 {args.final_epochs} 轮")
        final_tasks = [(user_id, info["window"]) for user_id, info in best_windows.items()]
        final_results, _ = dispatch_training_tasks(
            args=args,
            tasks=final_tasks,
            epochs=args.final_epochs,
            log_path=log_path,
            text_log_path=text_log_path,
        )
        for res in final_results:
            if res:
                best_windows[res["user"]] = res

    summary_path = log_dir / "best_windows.json"
    with summary_path.open("w") as f:
        json.dump(best_windows, f, indent=2, ensure_ascii=False)
    logging.info(f"最佳窗口与性能摘要已保存至 {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMOG VQGAN window sweep")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/code/server/data_storage/processed_data/window",
    )
    parser.add_argument(
        "--window-cache-dir",
        type=str,
        default=str(Path(__file__).parent / "cached_windows"),
        help="已切分窗口的本地缓存目录，默认写入当前项目。",
    )
    parser.add_argument("--users", nargs="*", help="仅训练指定用户 id，默认遍历全部目录。")
    parser.add_argument("--window-sizes", nargs="*", type=float, default=list(WINDOW_SIZES))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--target-width",
        type=int,
        default=50,
        help="重采样后的时间轴长度；设为 0 则使用该 window 的原始点数 (t*100Hz)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=40, help="DataLoader 进程/线程数，尽可能占满 40 个 CPU 核心")
    parser.add_argument("--cpu-threads", type=int, default=40, help="Torch 内部算子使用的 CPU 线程数")
    parser.add_argument("--cache-processes", type=int, default=40, help="缓存预处理的顶层进程数（目标 40 CPU 并行）")
    parser.add_argument("--skip-cache-build", action="store_true", help="跳过运行前的窗口缓存构建")
    parser.add_argument("--device", type=str, default="auto", help="auto / npu:0 / cuda:0 / cpu")
    parser.add_argument(
        "--gpu-ids",
        nargs="*",
        type=int,
        default=None,
        help="并行训练所使用的设备编号（NPU/CUDA），缺省时自动使用全部可用设备。",
    )
    parser.add_argument(
        "--max-parallel-train",
        type=int,
        default=None,
        help="同时运行的训练任务数量，默认与 GPU 数量一致。",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay（0 关闭；建议 1e-4~1e-2）")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument(
        "--base-channels",
        type=int,
        default=96,
        help="VQGAN 编解码器基础通道数（建议 64/96/128；越小越省显存/越快）",
    )
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-codebook-vectors", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--no-nonlocal", action="store_true", help="禁用 NonLocalBlock（更快/更省显存）")
    parser.add_argument("--q-loss-weight", type=float, default=1.0, help="VQ codebook loss 的权重")
    parser.add_argument("--train-rec-loss", choices=["l1", "mse"], default="l1", help="训练时重建损失类型")
    parser.add_argument("--input-noise-std", type=float, default=0.0, help="训练时输入噪声标准差（0 关闭）")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="梯度裁剪阈值（0 关闭）")
    parser.add_argument("--score-metric", choices=["mse", "l1"], default="mse", help="评估打分使用的重建误差类型")
    parser.add_argument("--sweep-epochs", type=int, default=1)
    parser.add_argument("--final-epochs", type=int, default=5)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--prep-workers", type=int, default=40, help="窗口生成的进程数，充分利用 CPU")
    parser.add_argument("--sos-token", type=int, default=0)
    parser.add_argument("--pkeep", type=float, default=0.5)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--no-data-parallel", action="store_true", help="调试时可禁用 DataParallel")
    parser.add_argument("--max-negative-per-split", type=int, default=50000)
    parser.add_argument("--max-train-per-user", type=int, default=None, help="可选的训练集子采样上限")
    parser.add_argument("--max-eval-per-split", type=int, default=None, help="验证/测试集可选采样上限")
    parser.add_argument(
        "--full-scan-eval",
        action="store_true",
        help="完整扫描 val/test CSV 做均匀采样（很慢但更接近无偏评估）；默认按需读取到负样本上限即停止。",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--log-dir", type=str, default="results/experiment_logs", help="指标/文本日志输出目录")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, text_log_path = setup_logging(Path(args.log_dir))
    run_experiments(args, text_log_path)
