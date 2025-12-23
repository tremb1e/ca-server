from __future__ import annotations

import argparse
import logging

from .runner import run_window_sweep_for_user

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQGAN-only models from server window datasets")
    parser.add_argument("--user", required=True, help="Target user/device hash (subject ID) to train.")
    parser.add_argument("--device", default="cuda:0", help="Training device, e.g. cuda:0 / cuda:1 / cpu.")
    parser.add_argument(
        "--window-sizes",
        nargs="*",
        type=float,
        default=None,
        help="Window sizes (sec) to train. Defaults to ca_config.toml [windows].sizes.",
    )
    parser.add_argument("--vqgan-epochs", type=int, default=10, help="Epochs to train the VQGAN model.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for CA-train --batch-size.")
    parser.add_argument("--max-train-per-user", type=int, default=None, help="Optional cap for --max-train-per-user (CPU smoke tests).")
    parser.add_argument("--max-negative-per-split", type=int, default=None, help="Optional override for --max-negative-per-split.")
    parser.add_argument("--max-eval-per-split", type=int, default=None, help="Optional override for --max-eval-per-split.")
    parser.add_argument("--no-reuse", dest="reuse", action="store_false", help="Do not reuse existing checkpoints.")
    parser.set_defaults(reuse=True)
    return parser.parse_args()


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    logger.info("Training user=%s device=%s", args.user, args.device)
    run_window_sweep_for_user(
        args.user,
        device=args.device,
        window_sizes=list(args.window_sizes) if args.window_sizes else None,
        vqgan_epochs=int(args.vqgan_epochs),
        batch_size=int(args.batch_size) if args.batch_size is not None else None,
        max_train_per_user=int(args.max_train_per_user) if args.max_train_per_user is not None else None,
        max_negative_per_split=int(args.max_negative_per_split) if args.max_negative_per_split is not None else None,
        max_eval_per_split=int(args.max_eval_per_split) if args.max_eval_per_split is not None else None,
        reuse_checkpoints=bool(args.reuse),
    )


if __name__ == "__main__":
    main()
