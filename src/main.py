import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import health, sensor_data
from .config import settings
from .utils.logging_config import setup_logging
from .utils.tls import TLS12_13_CIPHERS, probe_tls_configuration

setup_logging(
    log_level=settings.log_level,
    log_path=settings.log_path,
    log_format=settings.log_format
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Data storage path: {settings.data_storage_path}")
    logger.info(f"Log path: {settings.log_path}")
    logger.info(f"Server listening on {settings.host}:{settings.port}")

    yield

    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan
)

if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router, tags=["health"])
app.include_router(sensor_data.router, tags=["sensor_data"])


def build_hypercorn_config() -> "Config":
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"{settings.host}:{settings.port}"]
    config.loglevel = settings.log_level.lower()
    config.accesslog = "-"
    config.errorlog = "-"
    config.alpn_protocols = ["h2", "http/1.1"]

    probe = probe_tls_configuration(
        settings.tls_certfile,
        settings.tls_keyfile,
        settings.tls_ca_certs,
        key_password=settings.tls_keyfile_password,
        allow_encrypted_key=True,
    )

    if probe.is_tls:
        config.certfile = str(probe.certfile)
        config.keyfile = str(probe.keyfile)
        config.ciphers = TLS12_13_CIPHERS
        if settings.tls_keyfile_password:
            config.keyfile_password = settings.tls_keyfile_password
        if probe.ca_certs:
            config.ca_certs = str(probe.ca_certs)
        if probe.ssl_context and hasattr(config, "ssl"):
            config.ssl = probe.ssl_context

        logger.info(
            "TLS enabled for HTTP server (TLS 1.2/1.3, ALPN=%s, cert=%s, key=%s)",
            config.alpn_protocols,
            probe.certfile,
            probe.keyfile,
        )
    else:
        if probe.reason and probe.reason != "cert_or_key_not_configured":
            logger.warning(
                "TLS disabled for HTTP server due to %s; serving via h2c (cleartext HTTP/2).",
                probe.reason,
            )
        else:
            logger.info("No TLS certificate configured; serving via h2c (HTTP/2 over cleartext).")

    return config


if __name__ == "__main__":
    import asyncio
    from hypercorn.asyncio import serve
    from .grpc_server import create_grpc_server

    async def _run_servers() -> None:
        http_enabled = settings.http_enabled
        if settings.grpc_port == settings.port and http_enabled:
            logger.warning(
                "HTTP and gRPC are both configured for port %s; disabling HTTP server to keep single-port gRPC.",
                settings.port,
            )
            http_enabled = False
        if not http_enabled:
            logger.info("HTTP server disabled; running gRPC only on %s:%s", settings.grpc_host, settings.grpc_port)

        grpc_server = await create_grpc_server()
        await grpc_server.start()

        http_task = (
            asyncio.create_task(serve(app, build_hypercorn_config())) if http_enabled else None
        )
        grpc_task = asyncio.create_task(grpc_server.wait_for_termination())

        try:
            tasks = {grpc_task} | ({http_task} if http_task else set())
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            await grpc_server.stop(grace=5)
            logger.info("gRPC server stopped")

    asyncio.run(_run_servers())
