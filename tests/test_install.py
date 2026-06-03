from __future__ import annotations

import argparse
import json

import pytest

import pyroller.cli.install as install
from pyroller.cli.install import (
    InstallProfile,
    InstallReport,
    InstallReporter,
    StepResult,
    ValidationResult,
    build_install_parser,
)


def _jsonl_events(output: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(install.EVENT_PREFIX))
        for line in output.splitlines()
        if line.startswith(install.EVENT_PREFIX)
    ]


def test_step_result_serialization_rounds_duration_and_omits_empty_fields() -> None:
    result = StepResult(
        name="install",
        command=["python", "-m", "pip"],
        status="ok",
        return_code=0,
        duration_seconds=1.23456,
    )

    assert result.to_dict() == {
        "name": "install",
        "status": "ok",
        "command": ["python", "-m", "pip"],
        "duration_seconds": 1.235,
        "dry_run": False,
        "return_code": 0,
    }


def test_install_report_serialization_includes_steps_validations_and_doctor() -> None:
    report = InstallReport(
        requested_profile="auto",
        decision_reason="test",
        python_executable="/python",
        dry_run=True,
        reset_torch=True,
        skip_doctor=False,
        requirements=["demucs"],
        candidate_profiles=["cpu"],
        steps=[StepResult("pip", ["pip"], "skipped", dry_run=True, message="dry run")],
        validations=[ValidationResult("cpu", True, "ok")],
        doctor={"ok": True},
        selected_profile="cpu",
        ok=True,
        message="done",
        completed_at="now",
    )

    payload = report.to_dict()

    assert payload["command"] == "install"
    assert payload["steps"][0]["message"] == "dry run"
    assert payload["validations"] == [{"profile": "cpu", "ok": True, "message": "ok"}]
    assert payload["doctor"] == {"ok": True}


def test_install_report_finish_sets_status_fields() -> None:
    report = InstallReport(
        requested_profile="cpu",
        decision_reason="explicit",
        python_executable="/python",
        dry_run=False,
        reset_torch=True,
        skip_doctor=True,
        requirements=[],
        candidate_profiles=["cpu"],
    )

    install._finish_report(report, ok=False, message="failed", failed_step="validation")

    assert report.ok is False
    assert report.message == "failed"
    assert report.failed_step == "validation"
    assert report.completed_at is not None


def test_install_reporter_emits_human_and_jsonl(capsys) -> None:
    reporter = InstallReporter(progress_format="both")

    reporter.human("hello")
    reporter.event("install_started", stage="install", dry_run=True)
    reporter.subprocess_output(step="pip", line="line")

    output = capsys.readouterr().out
    events = _jsonl_events(output)

    assert "hello" in output
    assert "line" in output
    assert [event["type"] for event in events] == ["install_started", "install_subprocess_output"]
    assert all(event["schema_version"] == 1 for event in events)
    assert all("timestamp" in event for event in events)


def test_run_step_dry_run_emits_events_and_returns_skipped(capsys) -> None:
    reporter = InstallReporter(progress_format="jsonl")

    result = install._run_step(["python", "-m", "pip"], step="pip", dry_run=True, reporter=reporter)

    assert result.status == "skipped"
    assert result.return_code == 0
    assert result.dry_run is True
    events = _jsonl_events(capsys.readouterr().out)
    assert [event["type"] for event in events] == [
        "install_step_started",
        "install_subprocess_started",
        "install_subprocess_completed",
        "install_step_completed",
    ]


def test_build_install_parser_defaults_and_overrides() -> None:
    parser = build_install_parser()

    defaults = parser.parse_args([])
    custom = parser.parse_args(
        [
            "--profile",
            "cpu",
            "--no-reset-torch",
            "--skip-doctor",
            "--dry-run",
            "--progress-format",
            "jsonl",
            "--output-format",
            "json",
        ]
    )

    assert defaults.profile == "auto"
    assert defaults.reset_torch is True
    assert custom.profile == "cpu"
    assert custom.reset_torch is False
    assert custom.skip_doctor is True
    assert custom.dry_run is True
    assert custom.progress_format == "jsonl"
    assert custom.output_format == "json"


def test_detect_install_candidates_respects_explicit_profile() -> None:
    decision = install.detect_install_candidates("cpu")

    assert [profile.name for profile in decision.candidates] == ["cpu"]
    assert "explicit profile" in decision.reason


def test_detect_install_candidates_auto_selects_cuda_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(install, "_gpu_candidate_supported", lambda: True)
    monkeypatch.setattr(install, "_has_nvidia_smi", lambda: True)

    decision = install.detect_install_candidates("auto")

    assert [profile.name for profile in decision.candidates] == ["cu124", "cpu"]


def test_detect_install_candidates_auto_selects_cpu_when_gpu_is_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(install, "_gpu_candidate_supported", lambda: False)
    monkeypatch.setattr(install, "_has_nvidia_smi", lambda: True)

    decision = install.detect_install_candidates("auto")

    assert [profile.name for profile in decision.candidates] == ["cpu"]
    assert "outside the validated CUDA" in decision.reason


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.6.0+cpu", (2, 6, 0)),
        ("2.10", (2, 10, 0)),
        ("not-version", None),
    ],
)
def test_install_parse_version_tuple(version: str, expected: tuple[int, ...] | None) -> None:
    assert install._parse_version_tuple(version) == expected


def test_env_uses_socks_proxy_detects_case_insensitively(monkeypatch) -> None:
    for key in install.SOCKS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert install._env_uses_socks_proxy() is False

    monkeypatch.setenv("all_proxy", "SOCKS5://127.0.0.1:1080")

    assert install._env_uses_socks_proxy() is True


def test_install_profile_packages_builds_expected_dry_run_steps(tmp_path) -> None:
    profile = InstallProfile(
        name="test",
        label="Test profile",
        index_url="https://example.invalid/simple",
        constraint_resource="constraints.txt",
        torch_packages=("torch==1", "torchaudio==1"),
    )
    reporter = InstallReporter(progress_format="human")

    steps = install._install_profile_packages(
        profile,
        constraints_path=tmp_path / "constraints.txt",
        requirements=["demucs==1"],
        reset_torch=True,
        dry_run=True,
        reporter=reporter,
    )

    assert [step.name for step in steps] == [
        "upgrade_packaging_tools",
        "uninstall_existing_torch",
        "install_torch_stack",
        "install_audio_requirements",
    ]
    assert all(step.dry_run for step in steps)
    assert steps[2].command[-2:] == ["torch==1", "torchaudio==1"]


def test_run_install_command_dry_run_json_without_doctor(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        install,
        "detect_install_candidates",
        lambda _profile: install.ProfileDecision((install.PROFILES["cpu"],), "test decision"),
    )
    monkeypatch.setattr(install, "_bundled_audio_runtime_requirements", lambda: ["demucs==1"])
    monkeypatch.setattr(install, "_materialize_constraint_file", lambda _name: __import__("pathlib").Path("/tmp/constraints.txt"))

    exit_code = install.run_install_command(
        argparse.Namespace(
            profile="auto",
            progress_format="jsonl",
            output_format="json",
            dry_run=True,
            reset_torch=False,
            skip_doctor=True,
        )
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    events = _jsonl_events(output)
    report = json.loads("\n".join(line for line in output.splitlines() if not line.startswith("PYROLLER_EVENT ")))
    assert events[0]["type"] == "install_started"
    assert report["schema_version"] == 1
    assert report["type"] == "install_result"
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert [step["name"] for step in report["steps"]] == [
        "upgrade_packaging_tools",
        "install_torch_stack",
        "install_audio_requirements",
    ]
