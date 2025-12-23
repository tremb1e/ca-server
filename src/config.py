from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "Continuous Authentication Server"
    version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 10500
    http_enabled: bool = True

    # gRPC endpoint (h2c by default, upgrade to TLS when cert/key provided)
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 10500
    grpc_max_message_size: int = 4 * 1024 * 1024  # 4MB default
    grpc_max_concurrent_rpcs: int = 256

    # Optional TLS configuration; when unset the server stays on h2c (cleartext HTTP/2)
    tls_certfile: Optional[Path] = None
    tls_keyfile: Optional[Path] = None
    tls_ca_certs: Optional[Path] = None
    tls_keyfile_password: Optional[str] = None

    data_storage_path: Path = Path("./data_storage/raw_data")
    processed_data_path: Path = Path("./data_storage/processed_data")
    inference_storage_path: Path = Path("./data_storage/inference")
    log_path: Path = Path("./logs")
    hmog_data_path: Path = Path("/data/code/ca/refer/ContinAuth/src/data/processed/raw_hmog_data")
    processing_sampling_rate: int = 100
    processing_min_total_mb: int = 100
    processing_target_mb: int = 100
    hmog_acc_unit: str = "m/s^2"
    hmog_gyr_unit: str = "rad/s"
    hmog_mag_unit: str = "uT"

    encryption_key: str = "Continuous_Authentication"

    training_max_concurrent: int = 1
    training_check_interval_sec: int = 30
    auth_max_concurrent: int = 16
    auth_max_cached_models: int = 16
    auth_session_ttl_sec: int = 600

    max_request_size: int = 10 * 1024 * 1024

    log_level: str = "INFO"
    log_format: str = "json"

    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
