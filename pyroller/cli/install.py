from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import importlib.resources as resources
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PYTHON = sys.executable
DIST_NAME = "py-roller"
AUDIO_EXTRA = "audio-core"
MIN_TORCH = (2, 6, 0)
SOCKS_ENV_KEYS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy")


@dataclass(frozen=True)
class InstallProfile:
    name: str
    label: str
    index_url: str
    constraint_resource: str
    torch_packages: tuple[str, ...]


PROFILES: dict[str, InstallProfile] = {
    "cpu": InstallProfile(
        name="cpu",
        label="CPU-only latest validated audio environment",
        index_url="https://download.pytorch.org/whl/cpu",
        constraint_resource="audio-cpu.txt",
        torch_packages=("torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0"),
    ),
    "cu124": InstallProfile(
        name="cu124",
        label="CUDA 12.4 latest validated audio environment",
        index_url="https://download.pytorch.org/whl/cu124",
        constraint_resource="audio-cu124.txt",
        torch_packages=("torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0"),
    ),
}


@dataclass(frozen=True)
class ProfileDecision:
    candidates: tuple[InstallProfile, ...]
    reason: str


class InstallValidationError(RuntimeError):
    pass


def _command_text(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", _command_text(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _driver_version() -> tuple[int, int] | None:
    if not _has_nvidia_smi():
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        line = next((ln.strip() for ln in result.stdout.splitlines() if ln.strip()), "")
        pieces = line.split(".")
        if len(pieces) >= 2:
            return int(pieces[0]), int(pieces[1])
        if pieces:
            return int(pieces[0]), 0
    except Exception:
        return None
    return None


def _gpu_candidate_supported() -> bool:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    version = _driver_version()
    if version is None:
        return False
    return version >= (550, 0)


def detect_install_candidates(requested_profile: str) -> ProfileDecision:
    if requested_profile != "auto":
        return ProfileDecision((PROFILES[requested_profile],), f"using explicit profile '{requested_profile}'")
    if _gpu_candidate_supported():
        return ProfileDecision(
            (PROFILES["cu124"], PROFILES["cpu"]),
            "detected a validated NVIDIA driver on Linux x86_64; trying CUDA 12.4 first and automatically falling back to CPU if validation fails",
        )
    if _has_nvidia_smi():
        return ProfileDecision(
            (PROFILES["cpu"],),
            "NVIDIA tooling is present, but the platform/driver is outside the validated CUDA 12.4 profile; selecting CPU",
        )
    return ProfileDecision((PROFILES["cpu"],), "no validated NVIDIA runtime detected; selecting CPU profile")


def _materialize_resource(package: str, resource_name: str, *, subdir: str) -> Path:
    data = resources.files(package).joinpath(resource_name).read_text(encoding="utf-8")
    tmpdir = Path(tempfile.gettempdir()) / "py-roller-install" / subdir
    tmpdir.mkdir(parents=True, exist_ok=True)
    path = tmpdir / resource_name
    path.write_text(data, encoding="utf-8")
    return path


def _materialize_constraint_file(resource_name: str) -> Path:
    return _materialize_resource("pyroller.resources.constraints", resource_name, subdir="constraints")


def _materialize_audio_core_requirements() -> Path:
    return _materialize_resource("pyroller.resources.requirements", "audio-core.txt", subdir="requirements")


def _installed_audio_core_requirements() -> list[str]:
    try:
        requires = importlib_metadata.requires(DIST_NAME) or []
    except Exception:
        requires = []

    selected: list[str] = []
    for req in requires:
        if "extra == 'audio-core'" in req or 'extra == "audio-core"' in req:
            selected.append(req.split(";", 1)[0].strip())
    if selected:
        return selected

    fallback = _materialize_audio_core_requirements()
    return [line.strip() for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


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


def _env_uses_socks_proxy() -> bool:
    return any("socks" in os.environ.get(key, "").lower() for key in SOCKS_ENV_KEYS)


def _validate_profile(profile: InstallProfile) -> tuple[bool, str]:
    script = f"""
import importlib
import importlib.metadata as importlib_metadata
import json
import os

MIN_TORCH = {MIN_TORCH!r}
SOCKS_ENV_KEYS = {SOCKS_ENV_KEYS!r}
profile = {profile.name!r}
problems = []
notes = []


def parse_version_tuple(version_text, width=3):
    core = version_text.split('+', 1)[0]
    pieces = []
    for token in core.split('.'):
        digits = []
        for ch in token:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            break
        pieces.append(int(''.join(digits)))
        if len(pieces) >= width:
            break
    if not pieces:
        return None
    while len(pieces) < width:
        pieces.append(0)
    return tuple(pieces)


def dist_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None

try:
    torch = importlib.import_module('torch')
except Exception as exc:
    print(json.dumps({{'ok': False, 'message': f'torch import failed: {{exc.__class__.__name__}}: {{exc}}'}}))
    raise SystemExit(0)

try:
    importlib.import_module('torchaudio')
except Exception as exc:
    problems.append(f'torchaudio import failed: {{exc.__class__.__name__}}: {{exc}}')

loaded = {{'torch': getattr(torch, '__version__', '?')}}
for name in ('faster_whisper', 'ctranslate2', 'demucs', 'librosa', 'transformers', 'huggingface_hub'):
    try:
        module = importlib.import_module(name)
        loaded[name] = getattr(module, '__version__', dist_version(name) or 'unknown')
    except Exception as exc:
        problems.append(f'{{name}} import failed: {{exc.__class__.__name__}}: {{exc}}')

cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
try:
    cuda_available = bool(torch.cuda.is_available())
except Exception:
    cuda_available = False

parsed_torch = parse_version_tuple(loaded['torch'])
if parsed_torch is None or parsed_torch < MIN_TORCH:
    problems.append(f'torch {{loaded["torch"]}} is too old for the current transcriber stack; need >= {{".".join(str(x) for x in MIN_TORCH)}}')

transformers_version = loaded.get('transformers')
if transformers_version and parsed_torch is not None and parsed_torch < MIN_TORCH:
    problems.append(f'transformers {{transformers_version}} with torch {{loaded["torch"]}} may break local transcriber model loading')

uses_socks = any('socks' in os.environ.get(key, '').lower() for key in SOCKS_ENV_KEYS)
if uses_socks:
    try:
        importlib.import_module('socksio')
    except Exception as exc:
        problems.append(f'SOCKS proxy detected but socksio is unavailable: {{exc.__class__.__name__}}: {{exc}}')
    else:
        notes.append('socksio available for SOCKS proxy support')

if profile == 'cpu':
    if cuda_version is not None:
        problems.append(f'expected CPU torch build but found cuda={{cuda_version}}')
else:
    if cuda_version is None:
        problems.append('expected CUDA-enabled torch build but torch.version.cuda is None')
    elif not cuda_available:
        problems.append(f'expected CUDA to be available for profile {{profile}}, but torch.cuda.is_available() is False (cuda={{cuda_version}})')

summary = [f'torch={{loaded["torch"]}}', f'cuda={{cuda_version}}', f'cuda_available={{cuda_available}}']
if transformers_version:
    summary.append(f'transformers={{transformers_version}}')
if notes:
    summary.extend(notes)
message = '; '.join(problems) if problems else ', '.join(summary)
print(json.dumps({{'ok': not problems, 'message': message}}))
"""
    result = subprocess.run([PYTHON, "-c", script], capture_output=True, text=True)
    output = [line for line in (result.stdout or "").splitlines() if line.strip()]
    payload = output[-1] if output else ""
    if result.returncode != 0:
        return False, (result.stderr or payload or "validation subprocess failed").strip()
    try:
        import json
        data = json.loads(payload)
        return bool(data.get("ok")), str(data.get("message", ""))
    except Exception:
        return False, payload or (result.stderr or "unable to parse validation output").strip()


def _install_profile_packages(profile: InstallProfile, *, constraints_path: Path, requirements: Sequence[str], reset_torch: bool, dry_run: bool) -> None:
    _run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run)
    if reset_torch:
        _run([PYTHON, "-m", "pip", "uninstall", "-y", "torch", "torchaudio", "torchvision"], dry_run=dry_run)
    _run(
        [PYTHON, "-m", "pip", "install", "--force-reinstall", "--index-url", profile.index_url, *profile.torch_packages],
        dry_run=dry_run,
    )
    _run(
        [PYTHON, "-m", "pip", "install", "--upgrade", "-c", str(constraints_path), *requirements],
        dry_run=dry_run,
    )


def build_install_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the official py-roller audio environment into the current Python environment. "
            "Defaults to profile=auto, which chooses the best validated profile for this machine and automatically falls back to CPU if validation fails."
        )
    )
    parser.add_argument("--profile", choices=("auto", "cpu", "cu124"), default="auto", help="Installation profile. Default: auto")
    parser.add_argument(
        "--reset-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Uninstall existing torch/torchaudio/torchvision before reinstalling the selected profile. Default: true",
    )
    parser.add_argument("--skip-doctor", action="store_true", help="Do not run 'py-roller doctor' after installation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def run_install_command(args: argparse.Namespace) -> int:
    decision = detect_install_candidates(args.profile)
    requirements = _installed_audio_core_requirements()
    print(f"Python                  : {PYTHON}")
    print(f"Install reason          : {decision.reason}")
    print(f"Audio core requirements : {', '.join(requirements)}")
    if _env_uses_socks_proxy():
        print("Proxy note              : SOCKS proxy detected in environment; install validation will require socksio/httpx[socks]")

    if args.dry_run:
        for profile in decision.candidates:
            constraints_path = _materialize_constraint_file(profile.constraint_resource)
            print(f"\nPlanned profile         : {profile.name} ({profile.label})")
            print(f"Constraint file         : {constraints_path}")
            _install_profile_packages(profile, constraints_path=constraints_path, requirements=requirements, reset_torch=args.reset_torch, dry_run=True)
            if not args.skip_doctor:
                _run([PYTHON, "-m", "pyroller.cli.main", "doctor"], dry_run=True)
        return 0

    errors: list[str] = []
    for idx, profile in enumerate(decision.candidates, start=1):
        constraints_path = _materialize_constraint_file(profile.constraint_resource)
        print(f"\nInstalling profile {idx}/{len(decision.candidates)}: {profile.name} ({profile.label})")
        print(f"Constraint file         : {constraints_path}")
        try:
            _install_profile_packages(profile, constraints_path=constraints_path, requirements=requirements, reset_torch=args.reset_torch, dry_run=False)
        except subprocess.CalledProcessError as exc:
            message = f"profile {profile.name} installation command failed with exit code {exc.returncode}"
            errors.append(message)
            print(f"Profile {profile.name} install failed: {message}")
            if args.profile != "auto" or idx == len(decision.candidates):
                raise
            print("Falling back to the next profile candidate...")
            continue

        ok, validation_message = _validate_profile(profile)
        if ok:
            print(f"Validation succeeded for profile {profile.name}: {validation_message}")
            if not args.skip_doctor:
                _run([PYTHON, "-m", "pyroller.cli.main", "doctor"], dry_run=False)
            return 0

        errors.append(f"profile {profile.name} validation failed: {validation_message}")
        print(f"Profile {profile.name} validation failed: {validation_message}")
        if args.profile != "auto" or idx == len(decision.candidates):
            raise InstallValidationError(validation_message)
        print("Falling back to the next profile candidate...")

    raise InstallValidationError("; ".join(errors) if errors else "no validated install profile succeeded")
