import argparse
import logging

from .pipeline import build_config, process_all_users, process_user

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset processing pipeline")
    parser.add_argument("--user", help="Process a single user ID (device hash). When omitted, all users are processed.")
    return parser.parse_args()


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
