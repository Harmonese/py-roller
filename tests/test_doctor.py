from __future__ import annotations

import json
import types

import pytest

import pyroller.doctor as doctor
import pyroller.cli.doctor as doctor_cli
from pyroller.doctor import CheckResult, DoctorReport


def test_check_result_dict_omits_empty_optional_fields() -> None:
    assert CheckResult("python", "ok", "ready").to_dict() == {
        "name": "python",
        "status": "ok",
        "message": "ready",
    }


def test_doctor_report_dict_contains_command_platform_and_checks() -> None:
    report = DoctorReport(
        ok=False,
        python_executable="/venv/bin/python",
        python_version="3.12.0",
        platform_system="Darwin",
        platform_machine="arm64",
        checks=[CheckResult("torch", "fail", "missing", version="none", details={"hint": "install"})],
        suggested_next_step="py-roller install",
    )

    payload = report.to_dict()

    assert payload["command"] == "doctor"
    assert payload["ok"] is False
    assert payload["platform"] == {"system": "Darwin", "machine": "arm64"}
    assert payload["checks"][0]["details"] == {"hint": "install"}
    assert payload["suggested_next_step"] == "py-roller install"
    assert "timestamp" in payload


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.6.0", (2, 6, 0)),
        ("2.7.1+cu124", (2, 7, 1)),
        ("3.12", (3, 12, 0)),
        ("1.2.3.4", (1, 2, 3)),
        ("not-a-version", None),
    ],
)
def test_parse_version_tuple(version: str, expected: tuple[int, ...] | None) -> None:
    assert doctor._parse_version_tuple(version) == expected


def test_contains_any_is_case_insensitive() -> None:
    assert doctor._contains_any("Missing LIBTORCH_CUDA dylib", ("libtorch_cuda",))
    assert not doctor._contains_any("all good", ("libtorch_cuda", "cudnn"))


def test_proxy_check_passes_without_socks_environment(monkeypatch) -> None:
    for key in doctor.SOCKS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    result = doctor._check_proxy_support()

    assert result.status == "ok"
    assert result.details == {"uses_socks": False}


def test_proxy_check_reports_missing_pysocks(monkeypatch) -> None:
    for key in doctor.SOCKS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")

    def fake_import_module(name: str):
        if name == "socks":
            raise ImportError("no socks")
        return types.SimpleNamespace()

    monkeypatch.setattr(doctor.importlib, "import_module", fake_import_module)

    result = doctor._check_proxy_support()

    assert result.status == "fail"
    assert result.details["uses_socks"] is True
    assert result.details["proxy_env_keys"] == ["HTTPS_PROXY"]


def test_build_doctor_report_uses_suggested_next_step_for_failed_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_doctor_checks",
        lambda: [
            CheckResult("python", "ok", "ready"),
            CheckResult("torch", "fail", "missing"),
        ],
    )

    report = doctor.build_doctor_report()

    assert report.ok is False
    assert report.suggested_next_step == "py-roller install"


def test_print_doctor_json_outputs_machine_readable_payload(capsys) -> None:
    report = DoctorReport(
        ok=True,
        python_executable="/python",
        python_version="3.12",
        platform_system="Linux",
        platform_machine="x86_64",
        checks=[CheckResult("python", "ok", "ready")],
    )

    doctor_cli.print_doctor_json(report)

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["checks"] == [{"name": "python", "status": "ok", "message": "ready"}]


def test_run_doctor_returns_nonzero_for_unhealthy_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor_cli,
        "build_doctor_report",
        lambda: DoctorReport(
            ok=False,
            python_executable="/python",
            python_version="3.12",
            platform_system="Linux",
            platform_machine="x86_64",
            checks=[CheckResult("torch", "fail", "missing")],
            suggested_next_step="py-roller install",
        ),
    )

    assert doctor_cli.run_doctor(output_format="human") == 1
    assert "py-roller install" in capsys.readouterr().out
