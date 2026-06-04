from pyroller.cli.doctor import DoctorReport
from pyroller.engine import doctor_protocol_request


def test_doctor_protocol_request_wraps_existing_report() -> None:
    report = DoctorReport(
        ok=True,
        python_executable="/python",
        python_version="3.12.0",
        platform_system="TestOS",
        platform_machine="arm64",
        checks=[],
    )

    payload = doctor_protocol_request(report)

    assert payload["schema_version"] == 1
    assert payload["type"] == "doctor_result"
    assert payload["status"] == "ok"
    assert payload["artifact_paths"] == {}
    assert payload["python_executable"] == "/python"
