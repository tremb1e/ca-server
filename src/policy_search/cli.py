from __future__ import annotations

import argparse
import logging

from ..ca_config import get_ca_config
from .runner import PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N, GridSearchConfig, run_policy_grid_search

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline policy grid search + Pareto frontier for Continuous Authentication")
    parser.add_argument("--user", required=True, help="Target user/device hash (subject ID).")
    parser.add_argument("--device", default="cuda:0", help="Scoring device, e.g. cuda:0 / cuda:1 / cpu.")
    parser.add_argument(
        "--auth-method",
        choices=["vqgan-only", "vqgan+transformer", "both"],
        default="vqgan-only",
        help="Which per-window score to use for offline policy search (default: vqgan-only).",
    )
    parser.add_argument("--no-write-best", dest="write_best", action="store_false", help="Do not overwrite best_lock_policy.json.")
    parser.set_defaults(write_best=True)

    parser.add_argument("--window-sizes", nargs="*", type=float, default=None, help="Window sizes (sec). Defaults to ca_config.toml [windows].sizes.")
    parser.add_argument("--overlap", type=float, default=None, help="Window overlap ratio. Defaults to ca_config.toml [windows].overlap.")
    parser.add_argument("--vote-window-sizes", nargs="*", type=int, default=None, help="Vote N candidates. Defaults to N=7..20 (prompt spec).")
    parser.add_argument(
        "--vote-min-rejects-ratio-min",
        type=float,
        default=None,
        help="Override prompt M ranges: for each N, M starts at ceil(N*min). (Enabled only when both min/max are set.)",
    )
    parser.add_argument(
        "--vote-min-rejects-ratio-max",
        type=float,
        default=None,
        help="Override prompt M ranges: for each N, M ends at floor(N*max). (Enabled only when both min/max are set.)",
    )
    parser.add_argument(
        "--target-window-frr",
        nargs="*",
        type=float,
        default=None,
        help="target_window_frr candidates. Defaults to [0.001,0.002,0.005,0.01,0.02,0.05,ca_config].",
    )
    parser.add_argument("--max-decision-time-sec", type=float, default=None, help="Max decision time (sec). Defaults to ca_config.toml [auth].max_decision_time_sec.")
    parser.add_argument(
        "--decision-time-thresholds-sec",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Extra thresholds (sec) to compute session-level p(first_interrupt<=T). "
            "Defaults to [max_decision_time_sec]; 1s is always included for compatibility."
        ),
    )
    parser.add_argument("--token-batch-size", type=int, default=512)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        "--no-per-combo",
        dest="write_per_combo",
        action="store_false",
        help="Disable writing per-(auth_method,t,N,M) CSV files (keeps grid_results.csv).",
    )
    parser.set_defaults(write_per_combo=True)
    return parser.parse_args()


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    ca_cfg = get_ca_config()

    vote_window_sizes = list(args.vote_window_sizes) if args.vote_window_sizes else list(range(7, 21))
    vote_min_rejects_ratio_range = None
    vote_min_rejects_range_by_n = dict(PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N)
    if args.vote_min_rejects_ratio_min is not None or args.vote_min_rejects_ratio_max is not None:
        if args.vote_min_rejects_ratio_min is None or args.vote_min_rejects_ratio_max is None:
            raise SystemExit("--vote-min-rejects-ratio-min and --vote-min-rejects-ratio-max must be set together")
        vote_min_rejects_ratio_range = (float(args.vote_min_rejects_ratio_min), float(args.vote_min_rejects_ratio_max))
        # Explicitly disable prompt mapping when ratio override is requested.
        vote_min_rejects_range_by_n = None

    cfg = GridSearchConfig(
        window_sizes=list(args.window_sizes) if args.window_sizes else list(ca_cfg.windows.sizes),
        overlap=float(args.overlap) if args.overlap is not None else float(ca_cfg.windows.overlap),
        vote_window_sizes=vote_window_sizes,
        vote_min_rejects_ratio_range=vote_min_rejects_ratio_range,
        vote_min_rejects_range_by_n=vote_min_rejects_range_by_n,
        target_window_frr_candidates=(
            list(args.target_window_frr)
            if args.target_window_frr
            else sorted({0.001, 0.002, 0.005, 0.01, 0.02, 0.05, float(ca_cfg.auth.target_window_frr)})
        ),
        max_decision_time_sec=float(args.max_decision_time_sec) if args.max_decision_time_sec is not None else float(ca_cfg.auth.max_decision_time_sec),
        decision_time_thresholds_sec=list(args.decision_time_thresholds_sec) if args.decision_time_thresholds_sec else None,
        token_batch_size=int(args.token_batch_size),
        use_amp=bool(args.use_amp),
        write_per_combo=bool(args.write_per_combo),
    )
    methods = ["vqgan-only", "vqgan+transformer"] if args.auth_method == "both" else [str(args.auth_method)]
    for method in methods:
        out = run_policy_grid_search(
            args.user,
            device=str(args.device),
            cfg=cfg,
            write_best_policy=bool(args.write_best),
            auth_method=str(method),
        )
        logger.info("[%s] grid_results: %s", method, out["grid_csv"])
        logger.info("[%s] pareto_frontier: %s", method, out["pareto_csv"])
        logger.info("[%s] best_policy: %s", method, out["best_policy_json"])


if __name__ == "__main__":
    main()
