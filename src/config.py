from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils.runtime import app_root


class Settings(BaseSettings):
    app_name: str = "Continuous Authentication Server"
    version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 10500
    http_enabled: bool = False

    # gRPC endpoint (h2c by default, upgrade to TLS when cert/key provided)
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 10500
    grpc_max_message_size: int = 4 * 1024 * 1024  # 4MB default
    # 并发上限：需 >= 同时在线的认证流(StreamSensorData)用户数 + 健康检查等其它 RPC。
    # 设为 512 以从容支撑 100 并发认证用户（可用 GRPC_MAX_CONCURRENT_RPCS 覆盖）。
    grpc_max_concurrent_rpcs: int = 512

    # Optional TLS configuration; when unset the server stays on h2c (cleartext HTTP/2)
    tls_certfile: Optional[Path] = None
    tls_keyfile: Optional[Path] = None
    tls_ca_certs: Optional[Path] = None
    tls_keyfile_password: Optional[str] = None

    data_storage_path: Path = Field(default_factory=lambda: app_root() / "data_storage" / "raw_data")
    processed_data_path: Path = Field(default_factory=lambda: app_root() / "data_storage" / "processed_data")
    inference_storage_path: Path = Field(default_factory=lambda: app_root() / "data_storage" / "inference")
    log_path: Path = Field(default_factory=lambda: app_root() / "logs")
    hmog_data_path: Path = Field(default_factory=lambda: app_root() / "data_storage" / "hmog_preprocessed")
    processing_sampling_rate: int = 100
    # Deprecated: processing thresholds are read from ca_config.toml.
    processing_min_total_mb: int = 100
    processing_target_mb: int = 100
    hmog_acc_unit: str = "m/s^2"
    hmog_gyr_unit: str = "rad/s"
    hmog_mag_unit: str = "uT"

    encryption_key: str = "Continuous_Authentication"

    # 训练“真正同时在 NPU 上执行”的并行上限（每卡约 1 个，受 8 卡显存约束）。
    # 系统对训练用户是“无限接纳 + 队列”：100 个用户都会被接收并排队，最多 training_max_concurrent
    # 个同时训练，其余在 asyncio.Semaphore 上等待，因此可安全“支持 100 并发用户”而不 OOM。
    # 若目标机显存充足，可在 server.env 用 TRAINING_MAX_CONCURRENT 调大。
    training_max_concurrent: int = 8
    training_check_interval_sec: int = 30
    # 推理并发：支撑 100 个用户的瞬时前向（最终受 asyncio 线程池与本信号量共同约束）。
    auth_max_concurrent: int = 100
    # 在 NPU 上常驻的 per-user 模型 LRU 缓存数量；设为 100 以避免 100 用户场景频繁换入换出。
    auth_max_cached_models: int = 100
    auth_session_ttl_sec: int = 600

    management_api_enabled: bool = False
    management_api_key: Optional[str] = None

    max_request_size: int = 10 * 1024 * 1024
    max_decompressed_size: int = 10 * 1024 * 1024

    log_level: str = "INFO"
    log_format: str = "json"

    cors_enabled: bool = True
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
