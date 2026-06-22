from __future__ import annotations

import os
import time
from typing import Optional


DEFAULT_DB_NAME = "ngo_profiles"


def resolve_mongo_db_name(mongodb_uri: str, default: str = DEFAULT_DB_NAME) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(mongodb_uri)
        db_name = parsed.path.lstrip("/").split("?")[0] if parsed.path else default
        return db_name or default
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def create_mongo_client(
    mongodb_uri: str,
    *,
    server_selection_timeout_ms: Optional[int] = None,
    connect_timeout_ms: Optional[int] = None,
    socket_timeout_ms: Optional[int] = None,
    max_retries: Optional[int] = None,
    backoff_s: Optional[float] = None,
):
    """
    Create a MongoClient and verify connectivity with a ping.

    This is resilient to transient Atlas events (elections/maintenance) by:
    - using longer default timeouts than 5s
    - retrying with exponential backoff

    Env overrides:
      - MONGODB_SERVER_SELECTION_TIMEOUT_MS (default: 30000)
      - MONGODB_CONNECT_TIMEOUT_MS          (default: 20000)
      - MONGODB_SOCKET_TIMEOUT_MS           (default: 20000)
      - MONGODB_CONNECT_RETRIES             (default: 5)
      - MONGODB_CONNECT_BACKOFF_S           (default: 1.0)
    """
    from pymongo import MongoClient

    server_selection_timeout_ms = server_selection_timeout_ms or _env_int(
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS", 30_000
    )
    connect_timeout_ms = connect_timeout_ms or _env_int("MONGODB_CONNECT_TIMEOUT_MS", 20_000)
    socket_timeout_ms = socket_timeout_ms or _env_int("MONGODB_SOCKET_TIMEOUT_MS", 20_000)
    max_retries = max_retries or _env_int("MONGODB_CONNECT_RETRIES", 5)
    backoff_s = backoff_s if backoff_s is not None else _env_float("MONGODB_CONNECT_BACKOFF_S", 1.0)

    last_error: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
            socketTimeoutMS=socket_timeout_ms,
            retryWrites=True,
            retryReads=True,
        )
        try:
            client.admin.command("ping")
            return client
        except Exception as e:
            last_error = e
            topology = None
            try:
                topology = client.topology_description
            except Exception:
                topology = None
            try:
                client.close()
            except Exception:
                pass

            if attempt >= max_retries:
                details = f"{e}"
                if topology is not None:
                    details = f"{details}, Topology: {topology}"
                raise ConnectionError(
                    f"Failed to connect to MongoDB after {max_retries} attempt(s) "
                    f"(serverSelectionTimeoutMS={server_selection_timeout_ms}). {details}"
                ) from last_error

            time.sleep(backoff_s * (2 ** (attempt - 1)))

