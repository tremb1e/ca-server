from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .runner import load_best_policy, run_auth_inference

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Continuous Authentication inference from a window CSV")
    parser.add_argument("--user", required=True, help="Target user/device hash (subject ID).")
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Optional policy json path; defaults to data_storage/models/<user>/best_lock_policy.json.",
    )
    parser.add_argument("--csv-path", required=True, help="Input window CSV path (server processed_data/window/*/*/*.csv).")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 / cuda:1 / cpu.")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path; defaults under data_storage/models/<user>/inference/.")
    parser.add_argument("--max-windows", type=int, default=None, help="Debug: stop after N windows (avoid scanning huge CSVs).")
    return parser.parse_args()


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    policy = load_best_policy(args.user, policy_path=Path(args.policy_json) if args.policy_json else None)
    out_csv, meta = run_auth_inference(
        csv_path=Path(args.csv_path),
        policy=policy,
        device=str(args.device),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        max_windows=int(args.max_windows) if args.max_windows is not None else None,
    )
    logger.info("Inference done: %s", out_csv)
    logger.info(
        "Policy: window=%.1f threshold=%.6f rule=%s vote=(%d,%d) k_rejects=%d",
        meta["window"],
        meta["threshold"],
        meta.get("interrupt_rule", ""),
        int(meta.get("vote_window_size", 0)),
        int(meta.get("vote_min_rejects", 0)),
        int(meta.get("k_rejects", 0)),
    )


if __name__ == "__main__":
    main()
