from __future__ import annotations

import contextlib
import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

_SOCKS_SCHEMES = {"socks", "socks4", "socks4a", "socks5", "socks5h"}


def _clean_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_timeout_seconds(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0.")
    return max(1, int(math.ceil(out)))


def _positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return out


def _redact_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parts = urlsplit(value)
    except Exception:
        return value
    if not parts.username and not parts.password:
        return value
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    redacted_netloc = f"***:***@{host}"
    return urlunsplit((parts.scheme, redacted_netloc, parts.path, parts.query, parts.fragment))


def _is_socks_proxy(value: str | None) -> bool:
    if not value:
        return False
    try:
        scheme = urlsplit(value).scheme.lower()
    except Exception:
        return value.lower().startswith("socks")
    return scheme in _SOCKS_SCHEMES


@dataclass(slots=True)
class HFDownloadConfig:
    """Small, user-facing subset of Hugging Face download controls.

    This intentionally does not mirror every huggingface_hub option. It only exposes
    the options that materially affect reliability in common restricted-network
    setups: XET, a single proxy URL, metadata/file timeouts, and download workers.
    """

    xet: str = "auto"
    proxy: str | None = None
    etag_timeout: int | None = None
    download_timeout: int | None = None
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.xet, bool):
            self.xet = "on" if self.xet else "off"
        self.xet = str(self.xet or "auto").strip().lower()
        if self.xet not in {"auto", "on", "off"}:
            raise ValueError("hf_xet must be one of: auto, on, off.")
        self.proxy = _clean_optional_str(self.proxy)
        self.etag_timeout = _positive_timeout_seconds(self.etag_timeout, name="hf_etag_timeout")
        self.download_timeout = _positive_timeout_seconds(self.download_timeout, name="hf_download_timeout")
        self.max_workers = _positive_int(self.max_workers, name="hf_max_workers")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "HFDownloadConfig":
        data = dict(config or {})
        xet = data.get("hf_xet", "auto")
        if xet is None:
            xet = "auto"
        return cls(
            xet=xet,
            proxy=_clean_optional_str(data.get("hf_proxy")),
            etag_timeout=data.get("hf_etag_timeout"),
            download_timeout=data.get("hf_download_timeout"),
            max_workers=data.get("hf_max_workers"),
        )

    def snapshot_download_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.etag_timeout is not None:
            kwargs["etag_timeout"] = self.etag_timeout
        if self.max_workers is not None:
            kwargs["max_workers"] = self.max_workers
        return kwargs

    def env_overrides(self, *, local_files_only: bool) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if local_files_only:
            overrides["HF_HUB_OFFLINE"] = "1"
            overrides["TRANSFORMERS_OFFLINE"] = "1"
        if self.xet == "off":
            overrides["HF_HUB_DISABLE_XET"] = "1"
        elif self.xet == "on":
            overrides["HF_HUB_DISABLE_XET"] = "0"
        if self.proxy:
            overrides["HTTP_PROXY"] = self.proxy
            overrides["HTTPS_PROXY"] = self.proxy
            overrides["ALL_PROXY"] = self.proxy
            overrides["http_proxy"] = self.proxy
            overrides["https_proxy"] = self.proxy
            overrides["all_proxy"] = self.proxy
        if self.etag_timeout is not None:
            overrides["HF_HUB_ETAG_TIMEOUT"] = str(self.etag_timeout)
        if self.download_timeout is not None:
            overrides["HF_HUB_DOWNLOAD_TIMEOUT"] = str(self.download_timeout)
        return overrides

    def ensure_runtime_requirements(self) -> None:
        if _is_socks_proxy(self.proxy) and importlib.util.find_spec("socks") is None:
            raise RuntimeError(
                "A SOCKS proxy was configured with --transcriber-hf-proxy, but PySocks is not installed. "
                "Install or repair the audio environment with: py-roller install. "
                "Alternatively install the missing dependency with: pip install PySocks, "
                "or pass an HTTP proxy URL instead."
            )

    def summary(self) -> dict[str, object]:
        return {
            "xet": self.xet,
            "proxy": _redact_url(self.proxy) or "environment/default",
            "etag_timeout": self.etag_timeout or "default",
            "download_timeout": self.download_timeout or "default",
            "max_workers": self.max_workers or "default",
        }


@contextlib.contextmanager
def hf_download_environment(config: HFDownloadConfig, *, local_files_only: bool) -> Iterator[None]:
    config.ensure_runtime_requirements()
    previous: dict[str, str | None] = {}
    overrides = config.env_overrides(local_files_only=local_files_only)
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def huggingface_download_error_hints(exc: BaseException) -> list[str]:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    hints: list[str] = []
    if any(token in text for token in ("xet", "cas", "cas-bridge", "cas-server", "hf_xet")):
        hints.append("try --transcriber-hf-xet off to avoid the XET/CAS download path")
    if "socks" in text or "invalidschema" in text:
        hints.append("install SOCKS support with py-roller install or pip install PySocks")
    if any(token in text for token in ("timeout", "timed out", "readtimeout", "connecttimeout")):
        hints.append("increase --transcriber-hf-download-timeout and/or --transcriber-hf-etag-timeout")
    if any(token in text for token in ("proxy", "connection", "network is unreachable", "name resolution")):
        hints.append("set --transcriber-hf-proxy to an HTTP/SOCKS proxy reachable from this machine")
    return hints
