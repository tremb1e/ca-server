from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from src.utils.runtime import append_env_pythonpath

append_env_pythonpath()


def _run_server() -> int:
    from src.main import run_server

    run_server()
    return 0


def _run_processing(argv: list[str]) -> int:
    from src.processing.cli import main

    sys.argv = [sys.argv[0], *argv]
    main()
    return 0


def _run_training(argv: list[str]) -> int:
    from src.training.cli import main

    sys.argv = [sys.argv[0], *argv]
    main()
    return 0


def _run_policy_search(argv: list[str]) -> int:
    from src.policy_search.cli import main

    sys.argv = [sys.argv[0], *argv]
    main()
    return 0


def _run_auth(argv: list[str]) -> int:
    from src.authentication.cli import main

    sys.argv = [sys.argv[0], *argv]
    main()
    return 0


def _run_ca_train_vqgan(argv: list[str]) -> int:
    from hmog_vqgan_experiment import parse_args, run_experiments, setup_logging

    sys.argv = [sys.argv[0], *argv]
    args = parse_args()
    _, text_log_path = setup_logging(Path(args.log_dir))
    run_experiments(args, text_log_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ca-server",
        description="Compiled Continuous Authentication server runtime",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "processing", "training", "policy-search", "auth", "ca-train-vqgan"],
        help="Runtime command to execute.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Frozen PyInstaller workers reuse this same entry binary. Without
    # freeze_support(), spawn/resource_tracker bootstrap args are parsed as
    # regular CLI subcommands and ProcessPoolExecutor crashes immediately.
    mp.freeze_support()

    parser = build_parser()
    parsed = parser.parse_args(argv)
    forwarded = list(parsed.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if parsed.command == "serve":
        return _run_server()
    if parsed.command == "processing":
        return _run_processing(forwarded)
    if parsed.command == "training":
        return _run_training(forwarded)
    if parsed.command == "policy-search":
        return _run_policy_search(forwarded)
    if parsed.command == "auth":
        return _run_auth(forwarded)
    if parsed.command == "ca-train-vqgan":
        return _run_ca_train_vqgan(forwarded)

    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
