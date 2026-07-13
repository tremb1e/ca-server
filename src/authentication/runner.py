from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..utils.reject_trackers import ConsecutiveRejectTracker, EMAScoreTracker, VoteRejectTracker
from ..utils.runtime import app_root
from ..utils.ca_train import ensure_ca_train_on_path
from ..utils.path_safety import safe_child_path, validate_storage_id
from ..utils.policy_paths import resolve_policy_path


@dataclass(frozen=True)
class AuthRunConfig:
    user: str
    window_size: float
    target_width: int
    overlap: float
    threshold: float
    interrupt_rule: str
    decision_strategy: str
    k_rejects: int
    vote_window_size: int
    vote_min_rejects: int
    ema_alpha: float
    vqgan_checkpoint: Path
    vqgan_config: Path
    model_version: str = ""
    policy_status: str = ""               # "ready" | "training_fallback" | "legacy" | ""
    policy_search_completed: bool = False
    score_metric: str = "mse"
    score_scale: str = "negative_reconstruction_error"
    threshold_strategy: str = ""
    policy_source: str = ""               # "best" | "training_fallback"
    policy_file: str = ""                 # absolute path of the loaded json
    genuine_score_stats: Optional[Dict[str, Any]] = None
    input_height: int = 6


def _server_root() -> Path:
    return app_root()


def _default_models_root(server_root: Path) -> Path:
    return server_root / "data_storage" / "models"


def _sigmoid(x: float) -> float:
    """Numerically-safe logistic mapping of a raw score to (0, 1)."""
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _infer_policy_readiness(policy: dict, policy_path: Path) -> Tuple[str, bool]:
    """Legacy-compatible readiness shim.

    Returns (policy_status, policy_search_completed). New policies carry these
    fields explicitly; older policies are inferred:
      - explicit fields present                  -> use them
      - training_fallback_policy.json filename    -> ("training_fallback", False)
      - has grid_search / interrupt_window_frr    -> ("ready", True)  (policy_search output)
      - otherwise                                 -> ("legacy", False)
    """
    has_status = "policy_status" in policy
    has_completed = "policy_search_completed" in policy
    if has_status or has_completed:
        completed = (
            bool(policy.get("policy_search_completed"))
            if has_completed
            else (str(policy.get("policy_status")) == "ready")
        )
        status = str(policy.get("policy_status")) if has_status else ("ready" if completed else "unknown")
        return status, completed
    if policy_path.name == "training_fallback_policy.json":
        return "training_fallback", False
    if "grid_search" in policy or policy.get("threshold_strategy") == "interrupt_window_frr":
        return "ready", True
    return "legacy", False


def load_best_policy(
    user: str,
    *,
    models_root: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    allow_training_fallback: bool = False,
) -> AuthRunConfig:
    user = validate_storage_id(user, field_name="user")
    server_root = _server_root()
    models_root = Path(models_root) if models_root is not None else _default_models_root(server_root)
    if policy_path is None:
        best = safe_child_path(models_root, user) / "best_lock_policy.json"
        fallback = safe_child_path(models_root, user) / "training_fallback_policy.json"
        if best.exists():
            policy_path = best
        elif allow_training_fallback and fallback.exists():
            policy_path = fallback
        else:
            # Keep pointing at the authoritative path so error messages are clear.
            policy_path = best
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
    decision_strategy = str(policy.get("decision_strategy", interrupt_rule) or interrupt_rule).strip().lower()
    if not interrupt_rule:
        if vote_window_size > 0 and vote_min_rejects > 0:
            interrupt_rule = "vote"
        elif k_rejects > 0:
            interrupt_rule = "k"
        else:
            interrupt_rule = "none"
    if not decision_strategy:
        decision_strategy = interrupt_rule

    vqgan_checkpoint = resolve_policy_path(
        policy.get("vqgan_checkpoint"),
        policy_path=policy_path,
        server_root=server_root,
        models_root=models_root,
    )
    vqgan_config = resolve_policy_path(
        policy.get("vqgan_config") or vqgan_checkpoint.with_suffix(".json"),
        policy_path=policy_path,
        server_root=server_root,
        models_root=models_root,
    )
    policy_status, policy_search_completed = _infer_policy_readiness(policy, policy_path)
    policy_source = "training_fallback" if policy_path.name == "training_fallback_policy.json" else "best"
    genuine_score_stats = policy.get("genuine_score_stats")
    if not isinstance(genuine_score_stats, dict):
        genuine_score_stats = None
    try:
        model_cfg = json.loads(vqgan_config.read_text(encoding="utf-8"))
        input_height = int(model_cfg.get("input_height", 0))
    except Exception:
        input_height = 0
    return AuthRunConfig(
        user=str(policy.get("user", user)),
        window_size=float(policy.get("window", 0.0)),
        target_width=int(policy.get("target_width", 50)),
        overlap=float(policy.get("overlap", 0.5)),
        threshold=float(policy.get("threshold", 0.0)),
        interrupt_rule=interrupt_rule,
        decision_strategy=decision_strategy,
        k_rejects=k_rejects,
        vote_window_size=vote_window_size,
        vote_min_rejects=vote_min_rejects,
        ema_alpha=float(policy.get("ema_alpha", 0.25)),
        vqgan_checkpoint=vqgan_checkpoint,
        vqgan_config=vqgan_config,
        model_version=str(policy.get("model_version", "")),
        policy_status=policy_status,
        policy_search_completed=policy_search_completed,
        score_metric=str(policy.get("score_metric", "mse")),
        score_scale=str(policy.get("score_scale", "negative_reconstruction_error")),
        threshold_strategy=str(policy.get("threshold_strategy", "")),
        policy_source=policy_source,
        policy_file=str(policy_path),
        genuine_score_stats=genuine_score_stats,
        input_height=input_height,
    )


def run_auth_inference(
    *,
    csv_path: Path,
    policy: AuthRunConfig,
    device: str = "auto",
    output_csv: Optional[Path] = None,
    max_windows: Optional[int] = None,
) -> Tuple[Path, Dict]:
    import numpy as np

    from ..utils.accelerator import resolve_torch_device
    from .vqgan_inference import load_vqgan, score_windows

    ensure_ca_train_on_path()
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
    ema_tracker = EMAScoreTracker()
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
                "ema_score",
                "decision_strategy",
                "raw_accept",
                "decision_accept",
                "accept",
                "interrupt",
                "consecutive_rejects",
                "vote_recent_windows",
                "vote_recent_rejects",
                "raw_threshold",
                "display_score",
                "display_threshold",
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
                    ema_tracker.reset()
                    vote_tracker.reset()
                    current_session_key = session_key

                raw_accept = bool(float(score) >= float(policy.threshold))
                decision_accept = raw_accept
                interrupt = False
                consecutive_rejects = 0
                ema_score = ""
                vote_recent_windows = 0
                vote_recent_rejects = 0
                decision_strategy = str(policy.decision_strategy or policy.interrupt_rule).strip().lower()
                if decision_strategy == "ema":
                    interrupt = ema_tracker.update(
                        float(score),
                        alpha=float(policy.ema_alpha),
                        threshold=float(policy.threshold),
                        reset_on_interrupt=False,
                    )
                    decision_accept = not bool(interrupt)
                    ema_score = f"{float(ema_tracker.ema_score or 0.0):.6f}"
                elif policy.vote_window_size > 0 and policy.vote_min_rejects > 0:
                    interrupt = vote_tracker.update(
                        rejected=not raw_accept,
                        window_size=int(policy.vote_window_size),
                        min_rejects=int(policy.vote_min_rejects),
                        reset_on_interrupt=True,
                    )
                    decision_accept = not bool(interrupt)
                    vote_recent_windows = int(vote_tracker.recent_windows)
                    vote_recent_rejects = int(vote_tracker.recent_rejects)
                elif policy.k_rejects > 0:
                    interrupt = k_tracker.update(
                        rejected=not raw_accept,
                        k=int(policy.k_rejects),
                        reset_on_interrupt=True,
                    )
                    decision_accept = not bool(interrupt)
                    consecutive_rejects = int(k_tracker.consecutive_rejects)

                # 拆分尺度：raw_* 为模型原始分数尺度；display_* 为 0~1 可视化分数。
                raw_threshold = f"{float(policy.threshold):.6f}"
                if decision_strategy == "ema" and ema_tracker.ema_score is not None:
                    display_basis = float(ema_tracker.ema_score)
                else:
                    display_basis = float(score)
                display_score = _sigmoid(display_basis)
                display_threshold = _sigmoid(float(policy.threshold))

                writer.writerow(
                    [
                        meta["window_id"],
                        meta["subject"],
                        meta["session"],
                        f"{float(score):.6f}",
                        ema_score,
                        decision_strategy,
                        int(raw_accept),
                        int(decision_accept),
                        int(decision_accept),
                        int(interrupt),
                        consecutive_rejects,
                        vote_recent_windows,
                        vote_recent_rejects,
                        raw_threshold,
                        f"{display_score:.6f}",
                        f"{display_threshold:.6f}",
                    ]
                )

            windows_batch.clear()
            meta_batch.clear()

        for idx, (window_id, subject, session, window) in enumerate(
            iter_windows_from_csv_unlabeled_with_session(
                csv_path,
                window_size_sec=float(policy.window_size),
                target_width=int(policy.target_width),
                input_height=int(policy.input_height),
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
        "input_height": int(policy.input_height),
        "threshold": float(policy.threshold),
        "interrupt_rule": str(policy.interrupt_rule),
        "decision_strategy": str(policy.decision_strategy),
        "k_rejects": int(policy.k_rejects),
        "vote_window_size": int(policy.vote_window_size),
        "vote_min_rejects": int(policy.vote_min_rejects),
        "ema_alpha": float(policy.ema_alpha),
        "vqgan_checkpoint": str(policy.vqgan_checkpoint),
        "vqgan_config": str(policy.vqgan_config),
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
        "max_windows": None if max_windows is None else int(max_windows),
    }
    return output_csv, meta
