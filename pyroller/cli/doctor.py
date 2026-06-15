from __future__ import annotations

import json

from pyroller.doctor import (
    CheckResult,
    DoctorReport,
    build_doctor_report,
    collect_doctor_checks,
)
from pyroller.i18n import _
from pyroller.utils.json import json_default


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
    from pyroller.engine import doctor_protocol_request

    payload = doctor_protocol_request(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


def run_doctor(output_format: str = "human") -> int:
    report = build_doctor_report()
    if output_format == "json":
        print_doctor_json(report)
    else:
        print_doctor_human(report)
    return 0 if report.ok else 1


__all__ = [
    "CheckResult",
    "DoctorReport",
    "build_doctor_report",
    "collect_doctor_checks",
    "print_doctor_human",
    "print_doctor_json",
    "run_doctor",
]
