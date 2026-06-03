from __future__ import annotations

import importlib.util

import pytest

from pyroller.transcriber.hf_download_config import (
    HFDownloadConfig,
    hf_download_environment,
    huggingface_download_error_hints,
)


def test_hf_download_config_normalizes_values_and_redacts_proxy() -> None:
    config = HFDownloadConfig(
        xet=False,
        proxy="socks5://user:secret@127.0.0.1:1080",
        etag_timeout=1.2,
        download_timeout="3.1",
        max_workers="2",
    )

    assert config.xet == "off"
    assert config.etag_timeout == 2
    assert config.download_timeout == 4
    assert config.max_workers == 2
    assert config.summary()["proxy"] == "socks5://***:***@127.0.0.1:1080"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xet": "maybe"},
        {"etag_timeout": 0},
        {"download_timeout": float("inf")},
        {"max_workers": 0},
    ],
)
def test_hf_download_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HFDownloadConfig(**kwargs)


def test_hf_download_config_from_config_and_snapshot_kwargs() -> None:
    config = HFDownloadConfig.from_config(
        {
            "hf_xet": "on",
            "hf_proxy": " http://127.0.0.1:7890 ",
            "hf_etag_timeout": 5,
            "hf_download_timeout": 9,
            "hf_max_workers": 1,
        }
    )

    assert config.xet == "on"
    assert config.proxy == "http://127.0.0.1:7890"
    assert config.snapshot_download_kwargs() == {"etag_timeout": 5, "max_workers": 1}


def test_hf_download_environment_applies_and_restores_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://old-proxy")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    config = HFDownloadConfig(xet="off", proxy="http://new-proxy", etag_timeout=7, download_timeout=11)

    with hf_download_environment(config, local_files_only=True):
        assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
        assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
        assert __import__("os").environ["HF_HUB_DISABLE_XET"] == "1"
        assert __import__("os").environ["HTTP_PROXY"] == "http://new-proxy"
        assert __import__("os").environ["HF_HUB_ETAG_TIMEOUT"] == "7"
        assert __import__("os").environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "11"

    assert __import__("os").environ["HTTP_PROXY"] == "http://old-proxy"
    assert "HF_HUB_OFFLINE" not in __import__("os").environ


def test_hf_download_config_requires_pysocks_for_socks_proxy(monkeypatch) -> None:
    def fake_find_spec(name: str):
        if name == "socks":
            return None
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="SOCKS proxy"):
        HFDownloadConfig(proxy="socks5://127.0.0.1:1080").ensure_runtime_requirements()


def test_huggingface_download_error_hints_detect_common_failure_modes() -> None:
    config = HFDownloadConfig(proxy="http://127.0.0.1:7890")
    exc = RuntimeError("ChunkedEncodingError from XET CAS bridge: connection reset and timeout")

    hints = huggingface_download_error_hints(exc, config)

    assert any("XET" in hint or "xet" in hint for hint in hints)
    assert any("max-workers 1" in hint for hint in hints)
    assert any("download-timeout" in hint for hint in hints)
    assert any("proxy" in hint for hint in hints)
