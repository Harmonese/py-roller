from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pyroller.i18n import _
from pyroller.protocol import PROTOCOL_VERSION, engine_version
from pyroller.utils.json import json_default

MIN_TORCH = (2, 6, 0)
SOCKS_ENV_KEYS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy")


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    message: str
    version: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(slots=True)
class DoctorReport:
    ok: bool
    python_executable: str
    python_version: str
    platform_system: str
    platform_machine: str
    checks: list[CheckResult]
    suggested_next_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "command": "doctor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "platform": {
                "system": self.platform_system,
                "machine": self.platform_machine,
            },
            "checks": [item.to_dict() for item in self.checks],
            "suggested_next_step": self.suggested_next_step,
        }


def _format_exception(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _parse_version_tuple(version_text: str, *, width: int = 3) -> tuple[int, ...] | None:
    core = version_text.split("+", 1)[0]
    pieces: list[int] = []
    for token in core.split("."):
        digits = []
        for ch in token:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            break
        pieces.append(int("".join(digits)))
        if len(pieces) >= width:
            break
    if not pieces:
        return None
    while len(pieces) < width:
        pieces.append(0)
    return tuple(pieces)


def _dist_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _check_python() -> CheckResult:
    return CheckResult(
        name="python",
        status="ok",
        message=_("Python {} on {} {}").format(platform.python_version(), platform.system(), platform.machine()),
        version=platform.python_version(),
        details={"system": platform.system(), "machine": platform.machine()},
    )


def _check_torch() -> CheckResult:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return CheckResult("torch", "fail", _format_exception(exc))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    torch_version = str(getattr(torch, "__version__", "unknown"))
    flavor = _("cuda={}").format(cuda_version) if cuda_version else _("cpu")
    status = "ok"
    message = _("torch {} ({}, cuda_available={})").format(torch_version, flavor, cuda_available)
    parsed = _parse_version_tuple(torch_version)
    if parsed is None or parsed < MIN_TORCH:
        status = "fail"
        message += _(" | too old for the current transcriber stack; reinstall with: py-roller install")
    return CheckResult(
        "torch",
        status,
        message,
        version=torch_version,
        details={"cuda": cuda_version, "cuda_available": cuda_available},
    )


def _check_torchaudio() -> CheckResult:
    try:
        torchaudio = importlib.import_module("torchaudio")
        version = str(getattr(torchaudio, "__version__", _dist_version("torchaudio") or _("unknown")))
        return CheckResult("torchaudio", "ok", _("torchaudio {}").format(version), version=version)
    except Exception as exc:
        message = _format_exception(exc)
        if _contains_any(message, ("libcudart", "cudnn", "nvcuda", "libtorch_cuda", "fbgemm.dll", "libtorchaudio")):
            message += _(" | Detected a GPU-flavored or ABI-mismatched Torch/Torchaudio build. Reinstall with: py-roller install")
        return CheckResult("torchaudio", "fail", message)


def _check_module(name: str, install_hint: str) -> CheckResult:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None) or _dist_version(name)
        detail = _("{} {}").format(name, version) if version else _("{} import ok").format(name)
        return CheckResult(name, "ok", detail, version=str(version) if version else None)
    except Exception as exc:
        return CheckResult(name, "fail", _("{} | Install/repair with: {}").format(_format_exception(exc), install_hint))


def _check_proxy_support() -> CheckResult:
    proxy_values = {key: os.environ.get(key, "") for key in SOCKS_ENV_KEYS if os.environ.get(key)}
    uses_socks = any("socks" in value.lower() for value in proxy_values.values())
    if not uses_socks:
        return CheckResult("proxy-socks", "ok", _("no SOCKS proxy detected in environment"), details={"uses_socks": False})
    try:
        importlib.import_module("socks")
        return CheckResult(
            "proxy-socks",
            "ok",
            _("SOCKS proxy detected and PySocks is available"),
            details={"uses_socks": True, "proxy_env_keys": sorted(proxy_values)},
        )
    except Exception as exc:
        return CheckResult(
            "proxy-socks",
            "fail",
            _("SOCKS proxy detected but PySocks is unavailable: {} | Reinstall with: py-roller install").format(_format_exception(exc)),
            details={"uses_socks": True, "proxy_env_keys": sorted(proxy_values)},
        )


def collect_doctor_checks() -> list[CheckResult]:
    return [
        _check_python(),
        _check_torch(),
        _check_torchaudio(),
        _check_module("faster_whisper", "py-roller install"),
        _check_module("ctranslate2", "py-roller install"),
        _check_module("transformers", "py-roller install"),
        _check_proxy_support(),
        _check_module("demucs", "py-roller install"),
        _check_module("librosa", "py-roller install"),
    ]


def build_doctor_report() -> DoctorReport:
    checks = collect_doctor_checks()
    bad = any(item.status in {"fail", "warn"} for item in checks)
    return DoctorReport(
        ok=not bad,
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        checks=checks,
        suggested_next_step=_("py-roller install") if bad else None,
    )


def print_doctor_human(report: DoctorReport) -> None:
    print(_("py-roller doctor"))
    print(_("  python executable      : {}").format(report.python_executable))
    for item in report.checks:
        tag = item.status.upper()
        print(_("  [{:<4}] {:<18} {}").format(tag, item.name, item.message))

    if not report.ok:
        print()
        print(_("Suggested next step:"))
        print(_("  py-roller install"))
        return
    print()
    print(_("Environment looks healthy."))


def print_doctor_json(report: DoctorReport) -> None:
    payload = report.to_dict()
    payload.update(
        {
            "schema_version": PROTOCOL_VERSION,
            "engine": "py-roller",
            "engine_version": engine_version(),
            "protocol_version": PROTOCOL_VERSION,
            "type": "doctor_result",
            "status": "ok" if report.ok else "failed",
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


def run_doctor(output_format: str = "human") -> int:
    report = build_doctor_report()
    if output_format == "json":
        print_doctor_json(report)
    else:
        print_doctor_human(report)
    return 0 if report.ok else 1
