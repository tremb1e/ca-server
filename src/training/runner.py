from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..ca_config import CAConfig, get_ca_config
from ..utils.accelerator import normalize_device
from ..utils.ca_train import ca_train_script, ensure_ca_train_on_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunResult:
    window_size: float
    threshold: float
    log_dir: Path
    output_dir: Path
    summary: Dict


def _server_root() -> Path:
    # server/src/training/runner.py -> server/
    return Path(__file__).resolve().parents[2]


def _default_dataset_path(server_root: Path) -> Path:
    return server_root / "data_storage" / "processed_data" / "window"


def _default_models_root(server_root: Path) -> Path:
    return server_root / "data_storage" / "models"


def _default_ca_train_script() -> Path:
    return ca_train_script("hmog_vqgan_experiment.py")


def _pick_workers() -> Tuple[int, int]:
    cpu = os.cpu_count() or 1
    num_workers = min(8, cpu)
    cpu_threads = min(16, cpu)
    return max(0, int(num_workers)), max(1, int(cpu_threads))


def _vqgan_config(target_width: int, *, base_channels: int, latent_dim: int, codebook_vectors: int, beta: float) -> Dict:
    return {
        "base_channels": int(base_channels),
        "latent_dim": int(latent_dim),
        "num_codebook_vectors": int(codebook_vectors),
        "beta": float(beta),
        "image_channels": 1,
        "input_height": 12,
        "input_width": int(target_width),
        "use_nonlocal": True,
    }


def _write_vqgan_config(cfg: Dict, checkpoint: Path) -> Path:
    config_path = checkpoint.with_suffix(".json")
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def _extract_last_training_error(log_dir: Path) -> Optional[str]:
    log_path = log_dir / "hmog_vqgan.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    patterns = ("[ERROR]", "Traceback", "ValueError:", "RuntimeError:")
    for raw in reversed(lines):
        line = raw.strip()
        if line and any(p in line for p in patterns):
            return line
    return None


def _read_best_window(log_dir: Path, user_id: str) -> Dict:
    summary_path = log_dir / "best_windows.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training finished but summary missing: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected best_windows.json format: {summary_path}")
    record = payload.get(str(user_id))
    if not isinstance(record, dict):
        details: List[str] = []
        if not payload:
            details.append("summary is empty")
        available_users = [str(k) for k, v in payload.items() if isinstance(v, dict)]
        if available_users:
            head = available_users[:3]
            if len(available_users) > 3:
                head.append("...")
            details.append(f"available_users={head}")
        last_error = _extract_last_training_error(log_dir)
        if last_error:
            details.append(f"last_training_error={last_error}")
        suffix = f" ({'; '.join(details)})" if details else ""
        raise ValueError(f"Missing user={user_id} entry in {summary_path}{suffix}")
    return record


def _write_training_summary(user_output_dir: Path, summary: Dict) -> None:
    summary_path = user_output_dir / "training_summary.json"
    existing: List[Dict] = []
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, list):
            existing = [r for r in data if isinstance(r, dict)]

    window = float(summary.get("window", 0.0) or 0.0)
    merged = {float(r.get("window", 0.0) or 0.0): r for r in existing if isinstance(r, dict)}
    if window > 0:
        merged[window] = summary
    ordered = [merged[k] for k in sorted(merged) if k > 0]
    summary_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_vqgan_policy(
    user_output_dir: Path,
    *,
    user_id: str,
    window_size: float,
    overlap: float,
    target_width: int,
    threshold: float,
    k_rejects: int,
    vqgan_checkpoint: Path,
    vqgan_config: Path,
) -> Path:
    policy = {
        "user": str(user_id),
        "window": float(window_size),
        "overlap": float(overlap),
        "target_width": int(target_width),
        "threshold": float(threshold),
        "interrupt_rule": "k",
        "k_rejects": int(k_rejects),
        "vote_window_size": 0,
        "vote_min_rejects": 0,
        "auth_method": "vqgan-only",
        "vqgan_checkpoint": str(vqgan_checkpoint),
        "vqgan_config": str(vqgan_config),
        "model_version": vqgan_checkpoint.name,
    }
    policy_path = user_output_dir / "best_lock_policy.json"
    policy_path.write_text(json.dumps({str(user_id): policy}, indent=2, ensure_ascii=False), encoding="utf-8")
    return policy_path


def run_window_sweep_for_user(
    user_id: str,
    *,
    device: str = "auto",
    window_sizes: Optional[Sequence[float]] = None,
    vqgan_epochs: int = 10,
    batch_size: Optional[int] = None,
    max_train_per_user: Optional[int] = None,
    max_negative_per_split: Optional[int] = None,
    max_eval_per_split: Optional[int] = None,
    reuse_checkpoints: bool = True,
    ca_cfg: Optional[CAConfig] = None,
    dataset_path: Optional[Path] = None,
    models_root: Optional[Path] = None,
    ca_train_script: Optional[Path] = None,
) -> List[TrainingRunResult]:
    ca_cfg = ca_cfg or get_ca_config()
    server_root = _server_root()
    dataset_path = Path(dataset_path) if dataset_path is not None else _default_dataset_path(server_root)
    models_root = Path(models_root) if models_root is not None else _default_models_root(server_root)
    ca_train_script = Path(ca_train_script) if ca_train_script is not None else _default_ca_train_script()

    if not ca_train_script.exists():
        raise FileNotFoundError(f"Missing CA-train script: {ca_train_script}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing window dataset path: {dataset_path}")

    ensure_ca_train_on_path()
    from hmog_consecutive_rejects import k_from_interrupt_time  # type: ignore

    window_sizes = list(window_sizes) if window_sizes is not None else list(ca_cfg.windows.sizes)
    if not window_sizes:
        raise ValueError("No window sizes configured.")

    user_output_dir = models_root / user_id
    user_output_dir.mkdir(parents=True, exist_ok=True)

    num_workers, cpu_threads = _pick_workers()
    results: List[TrainingRunResult] = []

    resolved_device = normalize_device(device)

    for ws in window_sizes:
        ws_f = float(ws)
        target_width = int(round(ws_f * float(ca_cfg.windows.sampling_rate_hz)))
        log_dir = user_output_dir / "logs" / f"ws_{ws_f:.1f}"
        log_dir.mkdir(parents=True, exist_ok=True)

        best_summary_path = log_dir / "best_windows.json"
        summary: Optional[Dict] = None
        reuse_error: Optional[str] = None
        if reuse_checkpoints and best_summary_path.exists():
            try:
                summary = _read_best_window(log_dir, user_id)
            except Exception as exc:
                reuse_error = str(exc)

        if summary is None:
            if reuse_error:
                logger.warning(
                    "Ignoring cached window sweep summary for user=%s ws=%.1f; will retrain. reason=%s",
                    user_id,
                    ws_f,
                    reuse_error,
                )
            cmd: List[str] = [
                sys.executable,
                str(ca_train_script),
                "--dataset-path",
                str(dataset_path),
                "--users",
                str(user_id),
                "--window-sizes",
                f"{ws_f:.1f}",
                "--overlap",
                str(float(ca_cfg.windows.overlap)),
                "--target-width",
                str(int(target_width)),
                "--device",
                str(resolved_device),
                "--batch-size",
                str(int(batch_size) if batch_size is not None else 128),
                "--num-workers",
                str(int(num_workers)),
                "--cpu-threads",
                str(int(cpu_threads)),
                "--sweep-epochs",
                str(int(vqgan_epochs)),
                "--final-epochs",
                str(int(vqgan_epochs)),
                "--output-dir",
                str(user_output_dir),
                "--log-dir",
                str(log_dir),
                "--use-amp",
            ]

            if max_train_per_user is not None:
                cmd.extend(["--max-train-per-user", str(int(max_train_per_user))])
            if max_negative_per_split is not None:
                cmd.extend(["--max-negative-per-split", str(int(max_negative_per_split))])
            if max_eval_per_split is not None:
                cmd.extend(["--max-eval-per-split", str(int(max_eval_per_split))])

            subprocess.run(cmd, check=True)
            summary = _read_best_window(log_dir, user_id)

        if not isinstance(summary, dict):
            raise ValueError(f"Unexpected summary format in {log_dir}")

        vqgan_checkpoint = Path(str(summary.get("checkpoint") or ""))
        if not vqgan_checkpoint.exists():
            raise FileNotFoundError(f"Missing VQGAN checkpoint: {vqgan_checkpoint}")

        vqgan_cfg = _vqgan_config(
            target_width,
            base_channels=int(summary.get("base_channels", 96) or 96),
            latent_dim=int(summary.get("latent_dim", 256) or 256),
            codebook_vectors=int(summary.get("num_codebook_vectors", 512) or 512),
            beta=float(summary.get("beta", 0.25) or 0.25),
        )
        vqgan_config = _write_vqgan_config(vqgan_cfg, vqgan_checkpoint)

        threshold = float(((summary.get("val") or {}).get("threshold") or 0.0))
        k_rejects = int(
            k_from_interrupt_time(
                float(ca_cfg.auth.max_decision_time_sec),
                window_size_sec=ws_f,
                overlap=float(ca_cfg.windows.overlap),
            )
        )

        summary["vqgan_config"] = str(vqgan_config)
        summary["threshold"] = threshold
        _write_training_summary(user_output_dir, summary)
        _write_vqgan_policy(
            user_output_dir,
            user_id=str(user_id),
            window_size=ws_f,
            overlap=float(ca_cfg.windows.overlap),
            target_width=int(target_width),
            threshold=threshold,
            k_rejects=k_rejects,
            vqgan_checkpoint=vqgan_checkpoint,
            vqgan_config=vqgan_config,
        )

        results.append(
            TrainingRunResult(
                window_size=ws_f,
                threshold=threshold,
                log_dir=log_dir,
                output_dir=user_output_dir,
                summary=summary,
            )
        )

    return results
