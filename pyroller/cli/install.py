from __future__ import annotations

import argparse
import importlib.resources as resources
import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from pyroller.i18n import _
from pyroller.protocol import PROTOCOL_VERSION, protocol_envelope
from pyroller.utils.json import json_default

PYTHON = sys.executable
DIST_NAME = "py-roller"
MIN_TORCH = (2, 6, 0)
SOCKS_ENV_KEYS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy")
EVENT_PREFIX = "PYROLLER_EVENT "


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
        label=_("CPU-only latest validated audio environment"),
        index_url="https://download.pytorch.org/whl/cpu",
        constraint_resource="audio-cpu.txt",
        torch_packages=("torch==2.6.0", "torchaudio==2.6.0"),
    ),
    "cu124": InstallProfile(
        name="cu124",
        label=_("CUDA 12.4 latest validated audio environment"),
        index_url="https://download.pytorch.org/whl/cu124",
        constraint_resource="audio-cu124.txt",
        torch_packages=("torch==2.6.0", "torchaudio==2.6.0"),
    ),
}


@dataclass(frozen=True)
class ProfileDecision:
    candidates: tuple[InstallProfile, ...]
    reason: str


@dataclass(slots=True)
class StepResult:
    name: str
    command: list[str]
    status: str
    return_code: int | None = None
    duration_seconds: float = 0.0
    dry_run: bool = False
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "command": self.command,
            "duration_seconds": round(self.duration_seconds, 3),
            "dry_run": self.dry_run,
        }
        if self.return_code is not None:
            payload["return_code"] = self.return_code
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class ValidationResult:
    profile: str
    ok: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "ok": self.ok, "message": self.message}


@dataclass(slots=True)
class InstallReport:
    requested_profile: str
    decision_reason: str
    python_executable: str
    dry_run: bool
    reset_torch: bool
    skip_doctor: bool
    requirements: list[str]
    candidate_profiles: list[str]
    steps: list[StepResult] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    doctor: dict[str, Any] | None = None
    selected_profile: str | None = None
    ok: bool = False
    failed_step: str | None = None
    message: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return protocol_envelope(
            "install_result",
            status="ok" if self.ok else "failed",
            ok=self.ok,
            command="install",
            requested_profile=self.requested_profile,
            selected_profile=self.selected_profile,
            candidate_profiles=self.candidate_profiles,
            decision_reason=self.decision_reason,
            python_executable=self.python_executable,
            dry_run=self.dry_run,
            reset_torch=self.reset_torch,
            skip_doctor=self.skip_doctor,
            requirements=self.requirements,
            steps=[step.to_dict() for step in self.steps],
            validations=[validation.to_dict() for validation in self.validations],
            doctor=self.doctor,
            failed_step=self.failed_step,
            message=self.message,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


class InstallValidationError(RuntimeError):
    pass


class InstallReporter:
    def __init__(self, *, progress_format: str = "human") -> None:
        self.progress_format = progress_format

    @property
    def emit_human(self) -> bool:
        return self.progress_format in {"human", "both"}

    @property
    def emit_jsonl(self) -> bool:
        return self.progress_format in {"jsonl", "both"}

    def human(self, text: str = "") -> None:
        if self.emit_human:
            print(text, flush=True)

    def event(self, event_type: str, **payload: Any) -> None:
        if not self.emit_jsonl:
            return
        payload.setdefault("schema_version", PROTOCOL_VERSION)
        payload.setdefault("type", event_type)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload.setdefault("stage", "install")
        payload.setdefault("message", "")
        payload.setdefault("progress", None)
        print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_default), flush=True)

    def subprocess_output(self, *, step: str, line: str) -> None:
        if self.emit_human:
            print(line, flush=True)
        if self.emit_jsonl:
            self.event("install_subprocess_output", stage="install", step=step, line=line)


def _command_text(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: list[str], *, step: str, dry_run: bool, reporter: InstallReporter) -> StepResult:
    return _run_step(cmd, step=step, dry_run=dry_run, reporter=reporter)


def _run_step(cmd: list[str], *, step: str, dry_run: bool, reporter: InstallReporter) -> StepResult:
    started = time.monotonic()
    command_text = _command_text(cmd)
    reporter.human(_("+ {}").format(command_text))
    reporter.event("install_step_started", stage="install", step=step, command=cmd, message=command_text, dry_run=dry_run)
    reporter.event("install_subprocess_started", stage="install", step=step, command=cmd, dry_run=dry_run)
    if dry_run:
        duration = time.monotonic() - started
        result = StepResult(name=step, command=cmd, status="skipped", return_code=0, duration_seconds=duration, dry_run=True, message=_("dry run"))
        reporter.event("install_subprocess_completed", stage="install", step=step, command=cmd, return_code=0, duration_seconds=duration, dry_run=True)
        reporter.event("install_step_completed", stage="install", step=step, return_code=0, duration_seconds=duration, dry_run=True)
        return result

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_PROGRESS_BAR", "off")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line_queue.put(line.rstrip("\n"))
        finally:
            line_queue.put(None)

    thread = threading.Thread(target=_reader, name=f"pyroller-install-{step}-reader", daemon=True)
    thread.start()

    last_output = time.monotonic()
    last_heartbeat = last_output
    reader_done = False
    while not reader_done:
        try:
            line = line_queue.get(timeout=1.0)
        except queue.Empty:
            now = time.monotonic()
            if now - last_output >= 15.0 and now - last_heartbeat >= 15.0 and process.poll() is None:
                reporter.event(
                    "heartbeat",
                    stage="install",
                    step=step,
                    message=_("subprocess is still running; no output recently"),
                    seconds_since_last_output=round(now - last_output, 1),
                )
                last_heartbeat = now
            continue
        if line is None:
            reader_done = True
            continue
        last_output = time.monotonic()
        last_heartbeat = last_output
        if line:
            reporter.subprocess_output(step=step, line=line)

    return_code = process.wait()
    thread.join(timeout=1.0)
    duration = time.monotonic() - started
    reporter.event(
        "install_subprocess_completed",
        stage="install",
        step=step,
        command=cmd,
        return_code=return_code,
        duration_seconds=duration,
    )
    if return_code != 0:
        message = _("command failed with exit code {}").format(return_code)
        reporter.event("install_step_failed", stage="install", step=step, command=cmd, return_code=return_code, duration_seconds=duration, message=message)
        raise subprocess.CalledProcessError(return_code, cmd)

    result = StepResult(name=step, command=cmd, status="ok", return_code=return_code, duration_seconds=duration)
    reporter.event("install_step_completed", stage="install", step=step, command=cmd, return_code=return_code, duration_seconds=duration)
    return result


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
        return ProfileDecision((PROFILES[requested_profile],), _("using explicit profile '{}'").format(requested_profile))
    if _gpu_candidate_supported():
        return ProfileDecision(
            (PROFILES["cu124"], PROFILES["cpu"]),
            _("detected a validated NVIDIA driver on Linux x86_64; trying CUDA 12.4 first and automatically falling back to CPU if validation fails"),
        )
    if _has_nvidia_smi():
        return ProfileDecision(
            (PROFILES["cpu"],),
            _("NVIDIA tooling is present, but the platform/driver is outside the validated CUDA 12.4 profile; selecting CPU"),
        )
    return ProfileDecision((PROFILES["cpu"],), _("no validated NVIDIA runtime detected; selecting CPU profile"))


def _materialize_resource(package: str, resource_name: str, *, subdir: str) -> Path:
    data = resources.files(package).joinpath(resource_name).read_text(encoding="utf-8")
    tmpdir = Path(tempfile.gettempdir()) / "py-roller-install" / subdir
    tmpdir.mkdir(parents=True, exist_ok=True)
    path = tmpdir / resource_name
    path.write_text(data, encoding="utf-8")
    return path


def _materialize_constraint_file(resource_name: str) -> Path:
    return _materialize_resource("pyroller.resources.constraints", resource_name, subdir="constraints")


def _materialize_audio_runtime_requirements() -> Path:
    return _materialize_resource("pyroller.resources.requirements", "audio-runtime.txt", subdir="requirements")


def _bundled_audio_runtime_requirements() -> list[str]:
    path = _materialize_audio_runtime_requirements()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


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


def _validate_profile(profile: InstallProfile, reporter: InstallReporter | None = None) -> ValidationResult:
    if reporter is not None:
        reporter.event("install_validation_started", stage="install", profile=profile.name, message=_("Validating profile {}").format(profile.name))
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

try:
    from pyroller.i18n import _ as _i18n
except Exception:
    def _i18n(text): return text


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
    print(json.dumps({{'ok': False, 'message': _i18n('torch import failed: {{}}: {{}}').format(exc.__class__.__name__, exc)}}))
    raise SystemExit(0)

try:
    importlib.import_module('torchaudio')
except Exception as exc:
    problems.append(_i18n('torchaudio import failed: {{}}: {{}}').format(exc.__class__.__name__, exc))

loaded = {{'torch': getattr(torch, '__version__', '?')}}
for name in ('faster_whisper', 'ctranslate2', 'demucs', 'librosa', 'transformers', 'huggingface_hub'):
    try:
        module = importlib.import_module(name)
        loaded[name] = getattr(module, '__version__', dist_version(name) or _i18n('unknown'))
    except Exception as exc:
        problems.append(_i18n('{{}} import failed: {{}}: {{}}').format(name, exc.__class__.__name__, exc))

cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
try:
    cuda_available = bool(torch.cuda.is_available())
except Exception:
    cuda_available = False

parsed_torch = parse_version_tuple(loaded['torch'])
if parsed_torch is None or parsed_torch < MIN_TORCH:
    problems.append(_i18n('torch {{}} is too old for the current transcriber stack; need >= {{}}').format(loaded['torch'], '.'.join(str(x) for x in MIN_TORCH)))

transformers_version = loaded.get('transformers')
if transformers_version and parsed_torch is not None and parsed_torch < MIN_TORCH:
    problems.append(_i18n('transformers {{}} with torch {{}} may break local transcriber model loading').format(transformers_version, loaded['torch']))

uses_socks = any('socks' in os.environ.get(key, '').lower() for key in SOCKS_ENV_KEYS)
if uses_socks:
    try:
        importlib.import_module('socks')
    except Exception as exc:
        problems.append(_i18n('SOCKS proxy detected but PySocks is unavailable: {{}}: {{}}').format(exc.__class__.__name__, exc))
    else:
        notes.append(_i18n('PySocks available for SOCKS proxy support'))

if profile == 'cpu':
    if cuda_version is not None:
        problems.append(_i18n('expected CPU torch build but found cuda={{}}').format(cuda_version))
else:
    if cuda_version is None:
        problems.append(_i18n('expected CUDA-enabled torch build but torch.version.cuda is None'))
    elif not cuda_available:
        problems.append(_i18n('expected CUDA to be available for profile {{}}, but torch.cuda.is_available() is False (cuda={{}})').format(profile, cuda_version))

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
        validation = ValidationResult(profile.name, False, (result.stderr or payload or _("validation subprocess failed")).strip())
    else:
        try:
            data = json.loads(payload)
            validation = ValidationResult(profile.name, bool(data.get("ok")), str(data.get("message", "")))
        except Exception:
            validation = ValidationResult(profile.name, False, payload or (result.stderr or _("unable to parse validation output")).strip())
    if reporter is not None:
        reporter.event(
            "install_validation_completed",
            stage="install",
            profile=profile.name,
            ok=validation.ok,
            message=validation.message,
        )
    return validation


def _install_profile_packages(
    profile: InstallProfile,
    *,
    constraints_path: Path,
    requirements: Sequence[str],
    reset_torch: bool,
    dry_run: bool,
    reporter: InstallReporter,
) -> list[StepResult]:
    results: list[StepResult] = []
    results.append(_run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], step="upgrade_packaging_tools", dry_run=dry_run, reporter=reporter))
    if reset_torch:
        results.append(_run([PYTHON, "-m", "pip", "uninstall", "-y", "torch", "torchaudio"], step="uninstall_existing_torch", dry_run=dry_run, reporter=reporter))
    results.append(
        _run(
            [PYTHON, "-m", "pip", "install", "--force-reinstall", "--index-url", profile.index_url, *profile.torch_packages],
            step="install_torch_stack",
            dry_run=dry_run,
            reporter=reporter,
        )
    )
    results.append(
        _run(
            [PYTHON, "-m", "pip", "install", "--upgrade", "-c", str(constraints_path), *requirements],
            step="install_audio_requirements",
            dry_run=dry_run,
            reporter=reporter,
        )
    )
    return results


def build_install_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=_(
            "Install or repair the official py-roller audio/transcriber runtime in the current Python environment. "
            "The auto profile tries the validated CUDA 12.4 stack on supported Linux/NVIDIA machines and falls back to CPU if validation fails."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "cpu", "cu124"),
        default="auto",
        help=_("Runtime profile: auto selects the best validated option, cpu forces CPU wheels, cu124 forces CUDA 12.4 wheels. Default: auto"),
    )
    parser.add_argument(
        "--reset-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_("Uninstall existing torch/torchaudio first to avoid ABI/flavor mismatches. Default: true"),
    )
    parser.add_argument("--skip-doctor", action="store_true", help=_("Skip the post-install 'py-roller doctor' validation step."))
    parser.add_argument("--dry-run", action="store_true", help=_("Print the pip commands that would run, but do not install anything."))
    parser.add_argument(
        "--progress-format",
        choices=("human", "jsonl", "both"),
        default="human",
        help=_(
            "Install progress output format. human keeps terminal output; "
            "jsonl emits machine-readable PYROLLER_EVENT lines; both emits both. Default: human"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("human", "json"),
        default="human",
        help=_("Final install summary format. Use json for machine-readable install reports. Default: human"),
    )
    return parser


def _finish_report(report: InstallReport, *, ok: bool, message: str | None = None, failed_step: str | None = None) -> None:
    report.ok = ok
    report.message = message
    report.failed_step = failed_step
    report.completed_at = datetime.now(timezone.utc).isoformat()


def _print_final_report(report: InstallReport, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=json_default), flush=True)


def run_install_command(args: argparse.Namespace) -> int:
    reporter = InstallReporter(progress_format=args.progress_format)
    decision = detect_install_candidates(args.profile)
    requirements = _bundled_audio_runtime_requirements()
    report = InstallReport(
        requested_profile=args.profile,
        decision_reason=decision.reason,
        python_executable=PYTHON,
        dry_run=bool(args.dry_run),
        reset_torch=bool(args.reset_torch),
        skip_doctor=bool(args.skip_doctor),
        requirements=requirements,
        candidate_profiles=[profile.name for profile in decision.candidates],
    )

    reporter.event(
        "install_started",
        stage="install",
        requested_profile=args.profile,
        candidate_profiles=report.candidate_profiles,
        python_executable=PYTHON,
        dry_run=args.dry_run,
        reset_torch=args.reset_torch,
    )
    reporter.human(_("Python                  : {}").format(PYTHON))
    reporter.human(_("Install reason          : {}").format(decision.reason))
    reporter.human(_("Audio runtime requirements : {}").format(', '.join(requirements)))
    if _env_uses_socks_proxy():
        reporter.human(_("Proxy note              : SOCKS proxy detected in environment; install validation will require PySocks/requests[socks]"))
        reporter.event("install_proxy_detected", stage="install", message=_("SOCKS proxy detected in environment"))

    errors: list[str] = []
    try:
        if args.dry_run:
            for profile in decision.candidates:
                constraints_path = _materialize_constraint_file(profile.constraint_resource)
                reporter.human(_("\nPlanned profile         : {} ({})").format(profile.name, profile.label))
                reporter.human(_("Constraint file         : {}").format(constraints_path))
                reporter.event(
                    "install_profile_selected",
                    stage="install",
                    profile=profile.name,
                    label=profile.label,
                    constraint_file=constraints_path,
                    dry_run=True,
                )
                report.steps.extend(
                    _install_profile_packages(
                        profile,
                        constraints_path=constraints_path,
                        requirements=requirements,
                        reset_torch=args.reset_torch,
                        dry_run=True,
                        reporter=reporter,
                    )
                )
                if not args.skip_doctor:
                    report.steps.append(_run([PYTHON, "-m", "pyroller.cli.main", "doctor", "--output-format", "json"], step="doctor", dry_run=True, reporter=reporter))
            _finish_report(report, ok=True, message=_("dry run complete"))
            reporter.event("install_completed", stage="install", ok=True, dry_run=True, message=_("dry run complete"))
            _print_final_report(report, args.output_format)
            return 0

        for idx, profile in enumerate(decision.candidates, start=1):
            constraints_path = _materialize_constraint_file(profile.constraint_resource)
            reporter.human(_("\nInstalling profile {idx}/{total}: {name} ({label})").format(idx=idx, total=len(decision.candidates), name=profile.name, label=profile.label))
            reporter.human(_("Constraint file         : {}").format(constraints_path))
            reporter.event(
                "install_profile_selected",
                stage="install",
                profile=profile.name,
                label=profile.label,
                profile_index=idx,
                profile_count=len(decision.candidates),
                constraint_file=constraints_path,
            )
            try:
                report.steps.extend(
                    _install_profile_packages(
                        profile,
                        constraints_path=constraints_path,
                        requirements=requirements,
                        reset_torch=args.reset_torch,
                        dry_run=False,
                        reporter=reporter,
                    )
                )
            except subprocess.CalledProcessError as exc:
                failed_step = "subprocess"
                if report.steps:
                    failed_step = report.steps[-1].name
                message = _("profile {name} installation command failed with exit code {code}").format(name=profile.name, code=exc.returncode)
                errors.append(message)
                reporter.human(_("Profile {name} install failed: {msg}").format(name=profile.name, msg=message))
                if args.profile != "auto" or idx == len(decision.candidates):
                    _finish_report(report, ok=False, message=message, failed_step=failed_step)
                    reporter.event("install_failed", stage="install", profile=profile.name, ok=False, message=message, failed_step=failed_step)
                    _print_final_report(report, args.output_format)
                    return 1
                reporter.human(_("Falling back to the next profile candidate..."))
                reporter.event("install_profile_fallback", stage="install", profile=profile.name, message=message)
                continue

            validation = _validate_profile(profile, reporter=reporter)
            report.validations.append(validation)
            if validation.ok:
                report.selected_profile = profile.name
                reporter.human(_("Validation succeeded for profile {name}: {msg}").format(name=profile.name, msg=validation.message))
                if not args.skip_doctor:
                    reporter.event("install_doctor_started", stage="install", profile=profile.name)
                    from pyroller.cli.doctor import print_doctor_human
                    from pyroller.doctor import build_doctor_report

                    doctor_report = build_doctor_report()
                    report.doctor = doctor_report.to_dict()
                    if reporter.emit_human:
                        print_doctor_human(doctor_report)
                    reporter.event(
                        "install_doctor_completed",
                        stage="install",
                        profile=profile.name,
                        ok=doctor_report.ok,
                        checks=[item.to_dict() for item in doctor_report.checks],
                    )
                    if not doctor_report.ok:
                        message = _("post-install doctor reported problems")
                        _finish_report(report, ok=False, message=message, failed_step="doctor")
                        reporter.event("install_failed", stage="install", profile=profile.name, ok=False, message=message, failed_step="doctor")
                        _print_final_report(report, args.output_format)
                        return 1
                _finish_report(report, ok=True, message=_("profile {name} installed and validated").format(name=profile.name))
                reporter.event("install_completed", stage="install", profile=profile.name, ok=True, message=report.message)
                _print_final_report(report, args.output_format)
                return 0

            errors.append(_("profile {name} validation failed: {msg}").format(name=profile.name, msg=validation.message))
            reporter.human(_("Profile {name} validation failed: {msg}").format(name=profile.name, msg=validation.message))
            if args.profile != "auto" or idx == len(decision.candidates):
                message = validation.message or _("validation failed")
                _finish_report(report, ok=False, message=message, failed_step="validation")
                reporter.event("install_failed", stage="install", profile=profile.name, ok=False, message=message, failed_step="validation")
                _print_final_report(report, args.output_format)
                return 1
            reporter.human(_("Falling back to the next profile candidate..."))
            reporter.event("install_profile_fallback", stage="install", profile=profile.name, message=validation.message)

        message = "; ".join(errors) if errors else _("no validated install profile succeeded")
        _finish_report(report, ok=False, message=message, failed_step="profile_selection")
        reporter.event("install_failed", stage="install", ok=False, message=message, failed_step="profile_selection")
        _print_final_report(report, args.output_format)
        return 1
    except Exception as exc:
        message = _("{}: {}").format(exc.__class__.__name__, exc)
        _finish_report(report, ok=False, message=message, failed_step=report.failed_step or _("unexpected"))
        reporter.event("install_failed", stage="install", ok=False, message=message, failed_step=report.failed_step)
        _print_final_report(report, args.output_format)
        raise
