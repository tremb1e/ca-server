#!/usr/bin/env python3
"""
Continuous authentication inference for the VQGAN->Token->Transformer(LM) pipeline.

Input:
  - a server-formatted window CSV (same schema as processed_data/window/*/*/*.csv)
Output:
  - per-window scores (and optional accept/reject decisions)
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import amp

from hmog_consecutive_rejects import ConsecutiveRejectTracker, VoteRejectTracker, k_from_interrupt_time
from hmog_data import DEFAULT_OVERLAP, iter_windows_from_csv_unlabeled_with_session
from hmog_token_transformer import TokenGPTAuthenticator, TokenLMConfig
from hmog_tokenizer import encode_windows_to_tokens
from vqgan import VQGAN


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_config_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_suffix(".json")


def load_vqgan(vqgan_ckpt: Path, *, device: torch.device, cfg_path: Optional[Path] = None) -> VQGAN:
    cfg_path = cfg_path or _default_config_path(vqgan_ckpt)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing VQGAN config json: {cfg_path}")
    cfg = _load_json(cfg_path)
    args = argparse.Namespace(**cfg)
    # The VQGAN module expects these attribute names.
    args.use_nonlocal = bool(cfg.get("use_nonlocal", True))
    model = VQGAN(args).to(device)
    model.load_state_dict(torch.load(vqgan_ckpt, map_location=device))
    model.eval()
    return model


def load_lm(lm_ckpt: Path, *, device: torch.device, cfg_path: Optional[Path] = None) -> TokenGPTAuthenticator:
    cfg_path = cfg_path or _default_config_path(lm_ckpt)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing LM config json: {cfg_path}")
    cfg = _load_json(cfg_path)
    # The training script may persist extra metadata (e.g., linked VQGAN checkpoint)
    # alongside the TokenLMConfig fields. Filter unknown keys for forward-compat.
    allowed = set(TokenLMConfig.__dataclass_fields__.keys())
    cfg_filtered = {k: v for k, v in cfg.items() if k in allowed}
    missing = [k for k in ("vocab_size", "block_size", "sos_token") if k not in cfg_filtered]
    if missing:
        raise ValueError(f"LM config {cfg_path} missing required keys: {missing}")
    lm_cfg = TokenLMConfig(**cfg_filtered)
    model = TokenGPTAuthenticator(lm_cfg).to(device)
    model.load_state_dict(torch.load(lm_ckpt, map_location=device))
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token-LM continuous authentication inference")
    parser.add_argument("--csv-path", type=str, required=True, help="Input CSV path (server window format)")
    parser.add_argument("--window-size", type=float, required=True, help="Window size in seconds (e.g., 0.8)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Window overlap ratio (server default=0.5)")
    parser.add_argument("--target-width", type=int, default=50, help="Resample width; 0 => round(window_size*100)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--use-amp", dest="use_amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)

    parser.add_argument("--vqgan-checkpoint", type=str, required=True)
    parser.add_argument("--vqgan-config", type=str, default=None)
    parser.add_argument("--lm-checkpoint", type=str, required=True)
    parser.add_argument("--lm-config", type=str, default=None)

    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold on score")
    parser.add_argument(
        "--vote-window-size",
        type=int,
        default=0,
        help="投票窗口长度 N（最近 N 个窗口）；与 --k-rejects/--interrupt-after-sec 互斥，0=关闭。",
    )
    parser.add_argument(
        "--vote-min-rejects",
        type=int,
        default=0,
        help="投票触发阈值 M：最近 N 个窗口中 reject>=M 触发 interrupt（需配合 --vote-window-size）。",
    )
    parser.add_argument("--k-rejects", type=int, default=0, help="连续拒绝 K 次才触发 interrupt（0=关闭）")
    parser.add_argument(
        "--interrupt-after-sec",
        type=float,
        default=None,
        help="固定打断时间（秒）。会按 stride=window*(1-overlap) 自动换算 K=ceil(T/stride)；与 --k-rejects 互斥。",
    )
    parser.add_argument(
        "--interrupt-after-sec-base",
        type=float,
        default=None,
        help=(
            "动态打断时间（秒）的 base：T = base + scale*window_size；"
            "会按 stride=window*(1-overlap) 自动换算 K=ceil(T/stride)。"
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
    parser.add_argument("--reset-on-interrupt", dest="reset_on_interrupt", action="store_true")
    parser.add_argument("--no-reset-on-interrupt", dest="reset_on_interrupt", action="store_false")
    parser.set_defaults(reset_on_interrupt=True)
    parser.add_argument("--max-windows", type=int, default=None, help="Debug: stop after N windows")
    parser.add_argument("--output-csv", type=str, default=None, help="Write results to CSV")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    csv_path = Path(args.csv_path)

    target_width = int(args.target_width) if int(args.target_width) > 0 else int(round(float(args.window_size) * 100))

    vote_window_size = int(getattr(args, "vote_window_size", 0) or 0)
    vote_min_rejects = int(getattr(args, "vote_min_rejects", 0) or 0)
    vote_enabled = vote_window_size > 0 or vote_min_rejects > 0
    if (vote_window_size > 0) != (vote_min_rejects > 0):
        raise ValueError("--vote-window-size and --vote-min-rejects must be set together (or both 0).")

    interrupt_after_sec = getattr(args, "interrupt_after_sec", None)
    if interrupt_after_sec is not None and float(interrupt_after_sec) <= 0.0:
        interrupt_after_sec = None

    interrupt_after_sec_base = getattr(args, "interrupt_after_sec_base", None)
    interrupt_after_sec_scale = getattr(args, "interrupt_after_sec_scale", 0.0)
    interrupt_min_k = int(getattr(args, "interrupt_min_k", 0) or 0)

    if interrupt_after_sec_base is None and float(interrupt_after_sec_scale or 0.0) != 0.0:
        raise ValueError("--interrupt-after-sec-scale requires --interrupt-after-sec-base.")

    if vote_enabled:
        if interrupt_after_sec is not None or interrupt_after_sec_base is not None or float(interrupt_after_sec_scale or 0.0) != 0.0:
            raise ValueError("Vote rule is incompatible with --interrupt-after-sec* options.")
        if int(args.k_rejects) > 0:
            raise ValueError("Vote rule is incompatible with --k-rejects; use only --vote-window-size/--vote-min-rejects.")

    if interrupt_after_sec_base is not None:
        if interrupt_after_sec is not None:
            raise ValueError("Use either --interrupt-after-sec or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both.")
        if int(args.k_rejects) > 0:
            raise ValueError("Use either --k-rejects or --interrupt-after-sec-base/--interrupt-after-sec-scale, not both.")
        interrupt_after_sec = float(interrupt_after_sec_base) + float(interrupt_after_sec_scale) * float(args.window_size)
        if interrupt_after_sec <= 0.0:
            raise ValueError(
                f"Computed interrupt_after_sec={interrupt_after_sec} must be >0 (base={interrupt_after_sec_base}, "
                f"scale={interrupt_after_sec_scale}, window={args.window_size})"
            )

    if interrupt_after_sec is not None and int(args.k_rejects) > 0:
        raise ValueError("Use either --k-rejects or --interrupt-after-sec (including dynamic), not both.")
    k_rejects_effective = int(args.k_rejects)
    if interrupt_after_sec is not None:
        k_rejects_effective = k_from_interrupt_time(
            float(interrupt_after_sec), window_size_sec=float(args.window_size), overlap=float(args.overlap)
        )
    if interrupt_min_k > 0:
        k_rejects_effective = max(int(k_rejects_effective), int(interrupt_min_k))
    if vote_enabled:
        logger.info("[VOTE] window_size=%d min_rejects=%d", vote_window_size, vote_min_rejects)
    elif k_rejects_effective > 0:
        stride_sec = float(args.window_size) * (1.0 - float(args.overlap))
        logger.info(
            "[INTERRUPT] requested_T=%s stride=%.6fs -> K=%d (effective_T=%.3fs)",
            "none" if interrupt_after_sec is None else f"{float(interrupt_after_sec):.3f}s",
            stride_sec,
            k_rejects_effective,
            float(k_rejects_effective) * stride_sec,
        )

    vqgan = load_vqgan(
        Path(args.vqgan_checkpoint),
        device=device,
        cfg_path=Path(args.vqgan_config) if args.vqgan_config else None,
    )
    lm = load_lm(
        Path(args.lm_checkpoint),
        device=device,
        cfg_path=Path(args.lm_config) if args.lm_config else None,
    )

    out_rows: List[Dict] = []
    batch_windows: List[np.ndarray] = []
    batch_meta: List[Dict] = []
    k_tracker = ConsecutiveRejectTracker()
    vote_tracker = VoteRejectTracker()
    current_session_key: Optional[str] = None

    def _flush_batch() -> None:
        nonlocal current_session_key
        if not batch_windows:
            return
        windows_np = np.stack(batch_windows, axis=0).astype(np.float32, copy=False)
        tok = encode_windows_to_tokens(
            vqgan,
            windows_np,
            batch_size=int(args.batch_size),
            device=device,
            use_amp=bool(args.use_amp),
        )
        tokens = torch.from_numpy(tok.tokens).to(device=device, dtype=torch.long, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=bool(args.use_amp)):
            scores = lm.score(tokens).detach().cpu().numpy()
        for meta, score in zip(batch_meta, scores):
            row = dict(**meta, score=float(score))
            if args.threshold is not None:
                thr = float(args.threshold)
                raw_accept = bool(float(score) >= thr)
                row["accept"] = int(raw_accept)

                if vote_enabled:
                    session_key = f"{meta.get('subject','')}\t{meta.get('session','')}"
                    if current_session_key is None:
                        current_session_key = session_key
                    elif session_key != current_session_key:
                        vote_tracker.reset()
                        current_session_key = session_key
                    triggered = vote_tracker.update(
                        rejected=not raw_accept,
                        window_size=vote_window_size,
                        min_rejects=vote_min_rejects,
                        reset_on_interrupt=bool(args.reset_on_interrupt),
                    )
                    row["interrupt"] = int(triggered)
                    row["vote_recent_windows"] = int(vote_tracker.recent_windows)
                    row["vote_recent_rejects"] = int(vote_tracker.recent_rejects)
                elif int(k_rejects_effective) > 0:
                    session_key = f"{meta.get('subject','')}\t{meta.get('session','')}"
                    if current_session_key is None:
                        current_session_key = session_key
                    elif session_key != current_session_key:
                        k_tracker.reset()
                        current_session_key = session_key
                    triggered = k_tracker.update(
                        rejected=not raw_accept,
                        k=int(k_rejects_effective),
                        reset_on_interrupt=bool(args.reset_on_interrupt),
                    )
                    row["interrupt"] = int(triggered)
                    row["consecutive_rejects"] = int(k_tracker.consecutive_rejects)
            out_rows.append(row)
        batch_windows.clear()
        batch_meta.clear()

    for idx, (window_id, subject, session, window) in enumerate(
        iter_windows_from_csv_unlabeled_with_session(
            csv_path,
            window_size_sec=float(args.window_size),
            target_width=target_width,
        ),
        start=1,
    ):
        batch_windows.append(window)
        batch_meta.append({"window_id": window_id, "subject": subject, "session": session})
        if len(batch_windows) >= int(args.batch_size):
            _flush_batch()
        if args.max_windows is not None and idx >= int(args.max_windows):
            break
    _flush_batch()

    if not out_rows:
        raise ValueError(f"No valid windows produced from {csv_path}")

    scores = np.array([r["score"] for r in out_rows], dtype=np.float64)
    logger.info(
        "[DONE] windows=%d score_mean=%.6f score_std=%.6f score_min=%.6f score_max=%.6f",
        len(out_rows),
        float(scores.mean()),
        float(scores.std()),
        float(scores.min()),
        float(scores.max()),
    )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(out_rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        logger.info("[WRITE] %s", out_path)

    # Print a few rows for quick inspection
    extra = ""
    if args.threshold is not None:
        extra = "\taccept"
        if vote_enabled:
            extra += "\tinterrupt\tvote_recent_windows\tvote_recent_rejects"
        elif int(k_rejects_effective) > 0:
            extra += "\tinterrupt\tconsecutive_rejects"
    print("window_id\tsubject\tsession\tscore" + extra)
    for r in out_rows[:10]:
        if args.threshold is None:
            print(f"{r['window_id']}\t{r['subject']}\t{r['session']}\t{r['score']:.6f}")
        else:
            if vote_enabled:
                print(
                    f"{r['window_id']}\t{r['subject']}\t{r['session']}\t{r['score']:.6f}\t{r['accept']}\t{r.get('interrupt',0)}\t{r.get('vote_recent_windows',0)}\t{r.get('vote_recent_rejects',0)}"
                )
            elif int(k_rejects_effective) > 0:
                print(
                    f"{r['window_id']}\t{r['subject']}\t{r['session']}\t{r['score']:.6f}\t{r['accept']}\t{r.get('interrupt',0)}\t{r.get('consecutive_rejects',0)}"
                )
            else:
                print(f"{r['window_id']}\t{r['subject']}\t{r['session']}\t{r['score']:.6f}\t{r['accept']}")


if __name__ == "__main__":
    main()
