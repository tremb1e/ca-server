from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..utils.accelerator import resolve_torch_device
from ..utils.ca_train import ensure_ca_train_on_path
from .vqgan_inference import VQGANPolicy, load_vqgan, score_windows


@dataclass(frozen=True)
class AuthRunConfig:
    user: str
    window_size: float
    target_width: int
    overlap: float
    threshold: float
    interrupt_rule: str
    k_rejects: int
    vote_window_size: int
    vote_min_rejects: int
    vqgan_checkpoint: Path
    vqgan_config: Path
    model_version: str = ""


def _server_root() -> Path:
    # server/src/authentication/runner.py -> server/
    return Path(__file__).resolve().parents[2]


def _default_models_root(server_root: Path) -> Path:
    return server_root / "data_storage" / "models"


def load_best_policy(
    user: str,
    *,
    models_root: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> AuthRunConfig:
    server_root = _server_root()
    models_root = Path(models_root) if models_root is not None else _default_models_root(server_root)
    if policy_path is None:
        policy_path = models_root / user / "best_lock_policy.json"
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy json for user={user}: {policy_path}")
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = payload.get(user) or payload.get(str(user))
    if not isinstance(policy, dict):
        raise ValueError(f"Unexpected best_lock_policy.json format: {policy_path}")

    k_rejects = int(policy.get("k_rejects", 0))
    vote_window_size = int(policy.get("vote_window_size", 0))
    vote_min_rejects = int(policy.get("vote_min_rejects", 0))
    interrupt_rule = str(policy.get("interrupt_rule", "") or "")
    if not interrupt_rule:
        if vote_window_size > 0 and vote_min_rejects > 0:
            interrupt_rule = "vote"
        elif k_rejects > 0:
            interrupt_rule = "k"
        else:
            interrupt_rule = "none"

    vqgan_checkpoint = Path(str(policy.get("vqgan_checkpoint")))
    vqgan_config = Path(str(policy.get("vqgan_config") or vqgan_checkpoint.with_suffix(".json")))
    return AuthRunConfig(
        user=str(policy.get("user", user)),
        window_size=float(policy.get("window", 0.0)),
        target_width=int(policy.get("target_width", 50)),
        overlap=float(policy.get("overlap", 0.5)),
        threshold=float(policy.get("threshold", 0.0)),
        interrupt_rule=interrupt_rule,
        k_rejects=k_rejects,
        vote_window_size=vote_window_size,
        vote_min_rejects=vote_min_rejects,
        vqgan_checkpoint=vqgan_checkpoint,
        vqgan_config=vqgan_config,
        model_version=str(policy.get("model_version", "")),
    )


def run_auth_inference(
    *,
    csv_path: Path,
    policy: AuthRunConfig,
    device: str = "auto",
    output_csv: Optional[Path] = None,
    max_windows: Optional[int] = None,
) -> Tuple[Path, Dict]:
    ensure_ca_train_on_path()
    from hmog_consecutive_rejects import ConsecutiveRejectTracker, VoteRejectTracker  # type: ignore
    from hmog_data import iter_windows_from_csv_unlabeled_with_session  # type: ignore

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    if output_csv is None:
        server_root = _server_root()
        models_root = _default_models_root(server_root)
        out_dir = models_root / policy.user / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / f"infer_ws_{policy.window_size:.1f}.csv"
    else:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    torch_device = resolve_torch_device(device)
    vqgan = load_vqgan(policy.vqgan_checkpoint, device=torch_device, config_path=policy.vqgan_config)

    k_tracker = ConsecutiveRejectTracker()
    vote_tracker = VoteRejectTracker()
    current_session_key: Optional[str] = None

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_id",
                "subject",
                "session",
                "score",
                "accept",
                "interrupt",
                "consecutive_rejects",
                "vote_recent_windows",
                "vote_recent_rejects",
            ]
        )

        windows_batch = []
        meta_batch = []
        count = 0

        def flush_batch() -> None:
            if not windows_batch:
                return
            scores = score_windows(
                vqgan,
                windows=np.stack(windows_batch, axis=0).astype("float32", copy=False),
                device=torch_device,
                use_amp=True,
            )
            for meta, score in zip(meta_batch, scores):
                nonlocal current_session_key
                session_key = f"{meta['subject']}::{meta['session']}"
                if current_session_key is None:
                    current_session_key = session_key
                elif current_session_key != session_key:
                    k_tracker.reset()
                    vote_tracker.reset()
                    current_session_key = session_key

                accept = bool(float(score) >= float(policy.threshold))
                interrupt = False
                consecutive_rejects = 0
                vote_recent_windows = 0
                vote_recent_rejects = 0
                if policy.vote_window_size > 0 and policy.vote_min_rejects > 0:
                    interrupt = vote_tracker.update(
                        rejected=not accept,
                        window_size=int(policy.vote_window_size),
                        min_rejects=int(policy.vote_min_rejects),
                        reset_on_interrupt=True,
                    )
                    vote_recent_windows = int(vote_tracker.recent_windows)
                    vote_recent_rejects = int(vote_tracker.recent_rejects)
                elif policy.k_rejects > 0:
                    interrupt = k_tracker.update(
                        rejected=not accept,
                        k=int(policy.k_rejects),
                        reset_on_interrupt=True,
                    )
                    consecutive_rejects = int(k_tracker.consecutive_rejects)

                writer.writerow(
                    [
                        meta["window_id"],
                        meta["subject"],
                        meta["session"],
                        f"{float(score):.6f}",
                        int(accept),
                        int(interrupt),
                        consecutive_rejects,
                        vote_recent_windows,
                        vote_recent_rejects,
                    ]
                )

            windows_batch.clear()
            meta_batch.clear()

        for idx, (window_id, subject, session, window) in enumerate(
            iter_windows_from_csv_unlabeled_with_session(
                csv_path,
                window_size_sec=float(policy.window_size),
                target_width=int(policy.target_width),
            )
        ):
            windows_batch.append(window)
            meta_batch.append({"window_id": window_id, "subject": subject, "session": session})
            count += 1
            if len(windows_batch) >= 256:
                flush_batch()
            if max_windows is not None and idx >= int(max_windows):
                break

        flush_batch()
        if count == 0:
            raise ValueError(f"No valid windows produced from {csv_path}")

    meta = {
        "user": policy.user,
        "window": float(policy.window_size),
        "overlap": float(policy.overlap),
        "target_width": int(policy.target_width),
        "threshold": float(policy.threshold),
        "interrupt_rule": str(policy.interrupt_rule),
        "k_rejects": int(policy.k_rejects),
        "vote_window_size": int(policy.vote_window_size),
        "vote_min_rejects": int(policy.vote_min_rejects),
        "vqgan_checkpoint": str(policy.vqgan_checkpoint),
        "vqgan_config": str(policy.vqgan_config),
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
        "max_windows": None if max_windows is None else int(max_windows),
    }
    return output_csv, meta
