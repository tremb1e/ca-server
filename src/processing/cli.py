import argparse
import logging

from ..utils.cli_args import normalize_option_value_args
from .pipeline import build_config, process_all_users, process_user

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = normalize_option_value_args(argv, options={"--user"})
    parser = argparse.ArgumentParser(description="Run dataset processing pipeline")
    parser.add_argument("--user", help="Process a single device_id_hash. When omitted, all devices are processed.")
    return parser.parse_args(argv)


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = build_config()
    if args.user:
        logger.info("Processing user %s", args.user)
        process_user(args.user, cfg)
    else:
        logger.info("Processing all users")
        process_all_users(cfg)


if __name__ == "__main__":
    main()
