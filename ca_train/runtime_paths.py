from __future__ import annotations

import os
from pathlib import Path


def app_root() -> Path:
    configured = os.getenv("APP_ROOT") or os.getenv("CA_APP_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def dataset_root() -> str:
    configured = os.getenv("CA_TRAIN_DATASET_PATH")
    if configured:
        return str(Path(configured).expanduser())
    return str(app_root() / "data_storage" / "processed_data" / "window")


def window_cache_dir() -> str:
    configured = os.getenv("CA_TRAIN_WINDOW_CACHE_DIR")
    if configured:
        return str(Path(configured).expanduser())
    return str(app_root() / "runtime" / "ca_train" / "cached_windows")


def token_cache_dir() -> str:
    configured = os.getenv("CA_TRAIN_TOKEN_CACHE_DIR")
    if configured:
        return str(Path(configured).expanduser())
    return str(app_root() / "runtime" / "ca_train" / "token_caches")


def results_dir() -> str:
    configured = os.getenv("CA_RESULTS_PATH")
    if configured:
        return str(Path(configured).expanduser())
    return str(app_root() / "results")
