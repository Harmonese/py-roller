from __future__ import annotations

import io
import signal
import subprocess

import pytest

import pyroller.process_control as process_control


def test_register_and_unregister_process_group() -> None:
    process_control.unregister_process_group(123456)

    process_control.register_process_group(123456)
    assert 123456 in process_control._ACTIVE_PGIDS

    process_control.unregister_process_group(123456)
    assert 123456 not in process_control._ACTIVE_PGIDS


def test_iter_subprocess_records_splits_newline_and_carriage_return() -> None:
    stream = io.StringIO("first\nsecond\rthird")

    assert list(process_control._iter_subprocess_records(stream)) == [
        ("first", "\n"),
        ("second", "\r"),
        ("third", "\n"),
    ]


def test_terminate_registered_process_groups_posix(monkeypatch) -> None:
    killed: list[tuple[int, int]] = []
    process_control.register_process_group(111)
    process_control.register_process_group(222)
    monkeypatch.setattr(process_control, "_IS_WINDOWS", False)
    monkeypatch.setattr(process_control.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(process_control, "_terminate_posix_process_group", lambda pgid, sig: killed.append((pgid, sig)))

    process_control.terminate_registered_process_groups(grace_seconds=0.01)

    assert sorted(killed) == [
        (111, signal.SIGKILL),
        (111, signal.SIGTERM),
        (222, signal.SIGKILL),
        (222, signal.SIGTERM),
    ]
    assert 111 not in process_control._ACTIVE_PGIDS
    assert 222 not in process_control._ACTIVE_PGIDS


def test_terminate_registered_process_groups_unregisters_missing_process(monkeypatch) -> None:
    calls = 0
    process_control.register_process_group(333)
    monkeypatch.setattr(process_control, "_IS_WINDOWS", False)
    monkeypatch.setattr(process_control.time, "sleep", lambda _seconds: None)

    def missing_process(_pgid: int, _sig: int) -> None:
        nonlocal calls
        calls += 1
        raise ProcessLookupError

    monkeypatch.setattr(process_control, "_terminate_posix_process_group", missing_process)

    process_control.terminate_registered_process_groups(grace_seconds=0.01)

    assert calls == 2
    assert 333 not in process_control._ACTIVE_PGIDS


def test_windows_process_tree_falls_back_to_os_kill_when_taskkill_missing(monkeypatch) -> None:
    killed: list[tuple[int, int]] = []

    def missing_taskkill(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(process_control.subprocess, "run", missing_taskkill)
    monkeypatch.setattr(process_control.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    process_control._terminate_windows_process_tree(44, force=False)

    assert killed == [(44, signal.SIGTERM)]


def test_run_subprocess_streams_records_and_unregisters(monkeypatch, capsys) -> None:
    callbacks: list[tuple[str, str]] = []
    popen_kwargs: dict[str, object] = {}

    class FakeProcess:
        pid = 555
        stdout = io.StringIO("one\ntwo\r")

        def wait(self) -> int:
            return 0

    def fake_popen(cmd, **kwargs):
        popen_kwargs.update(kwargs)
        assert cmd == ["fake", "cmd"]
        return FakeProcess()

    monkeypatch.setattr(process_control, "_IS_WINDOWS", False)
    monkeypatch.setattr(process_control.subprocess, "Popen", fake_popen)

    process_control.run_subprocess(["fake", "cmd"], output_callback=lambda record, sep: callbacks.append((record, sep)))

    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["stdout"] is subprocess.PIPE
    assert callbacks == [("one", "\n"), ("two", "\r")]
    assert "one\ntwo\r" in capsys.readouterr().err
    assert 555 not in process_control._ACTIVE_PGIDS


def test_run_subprocess_raises_called_process_error(monkeypatch) -> None:
    class FakeProcess:
        pid = 556
        stdout = None

        def wait(self) -> int:
            return 7

    monkeypatch.setattr(process_control.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        process_control.run_subprocess(["bad"])

    assert exc_info.value.returncode == 7
    assert 556 not in process_control._ACTIVE_PGIDS


def test_run_subprocess_terminates_registered_processes_on_exception(monkeypatch) -> None:
    terminated: list[float] = []

    class BrokenStdout:
        def read(self, _size: int) -> str:
            raise KeyboardInterrupt

    class FakeProcess:
        pid = 557
        stdout = BrokenStdout()

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(process_control.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(process_control, "terminate_registered_process_groups", lambda grace_seconds: terminated.append(grace_seconds))

    with pytest.raises(KeyboardInterrupt):
        process_control.run_subprocess(["interrupt"], output_callback=lambda _record, _sep: None)

    assert terminated == [0.2]
    assert 557 not in process_control._ACTIVE_PGIDS


def test_signal_handler_exits_with_signal_code(monkeypatch) -> None:
    called: list[float] = []
    monkeypatch.setattr(process_control, "terminate_registered_process_groups", lambda grace_seconds: called.append(grace_seconds))

    with pytest.raises(SystemExit) as exc_info:
        process_control._signal_handler(signal.SIGTERM, None)

    assert called == [0.2]
    assert exc_info.value.code == 128 + signal.SIGTERM
