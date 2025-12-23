import logging
import sys
import time
from pathlib import Path
from pythonjsonlogger import jsonlogger
from datetime import datetime
import os


def _cleanup_old_logs(log_path: Path, retention_days: int = 30) -> None:
    if retention_days <= 0:
        return
    cutoff = time.time() - (retention_days * 24 * 60 * 60)
    for entry in log_path.iterdir():
        if not entry.is_file() or entry.suffix != ".log":
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
        except FileNotFoundError:
            continue

def setup_logging(log_level: str = "INFO", log_path: Path = Path("./logs"), log_format: str = "json"):
    log_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = log_path / f"server_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized - Level: {log_level}, Format: {log_format}, File: {log_file}")
    _cleanup_old_logs(log_path, retention_days=30)
