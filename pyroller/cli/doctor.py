from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import os
import platform
import sys
from dataclasses import dataclass

MIN_TORCH = (2, 6, 0)
SOCKS_ENV_KEYS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy")


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    message: str


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
        message=f"Python {platform.python_version()} on {platform.system()} {platform.machine()}",
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
    flavor = f"cuda={cuda_version}" if cuda_version else "cpu"
    status = "ok"
    message = f"torch {torch.__version__} ({flavor}, cuda_available={cuda_available})"
    parsed = _parse_version_tuple(torch.__version__)
    if parsed is None or parsed < MIN_TORCH:
        status = "fail"
        message += f" | too old for the current transcriber stack; reinstall with: py-roller install"
    return CheckResult("torch", status, message)


def _check_torchaudio() -> CheckResult:
    try:
        torchaudio = importlib.import_module("torchaudio")
        return CheckResult("torchaudio", "ok", f"torchaudio {torchaudio.__version__}")
    except Exception as exc:
        message = _format_exception(exc)
        if _contains_any(message, ("libcudart", "cudnn", "nvcuda", "libtorch_cuda", "fbgemm.dll", "libtorchaudio")):
            message += " | Detected a GPU-flavored or ABI-mismatched Torch/Torchaudio build. Reinstall with: py-roller install"
        return CheckResult("torchaudio", "fail", message)


def _check_module(name: str, install_hint: str) -> CheckResult:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None) or _dist_version(name)
        detail = f"{name} {version}" if version else f"{name} import ok"
        return CheckResult(name, "ok", detail)
    except Exception as exc:
        return CheckResult(name, "fail", f"{_format_exception(exc)} | Install/repair with: {install_hint}")


def _check_proxy_support() -> CheckResult:
    uses_socks = any("socks" in os.environ.get(key, "").lower() for key in SOCKS_ENV_KEYS)
    if not uses_socks:
        return CheckResult("proxy-socks", "ok", "no SOCKS proxy detected in environment")
    try:
        importlib.import_module("socksio")
        return CheckResult("proxy-socks", "ok", "SOCKS proxy detected and socksio is available")
    except Exception as exc:
        return CheckResult("proxy-socks", "fail", f"SOCKS proxy detected but socksio is unavailable: {_format_exception(exc)} | Reinstall with: py-roller install")


def run_doctor() -> int:
    checks = [
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

    print("py-roller doctor")
    print(f"  python executable      : {sys.executable}")
    bad = False
    for item in checks:
        tag = item.status.upper()
        print(f"  [{tag:<4}] {item.name:<18} {item.message}")
        if item.status in {"fail", "warn"}:
            bad = True

    if bad:
        print("\nSuggested next step:")
        print("  py-roller install")
        return 1
    print("\nEnvironment looks healthy.")
    return 0
