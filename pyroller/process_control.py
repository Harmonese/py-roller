from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Sequence

_LOCK = threading.Lock()
_ACTIVE_PGIDS: set[int] = set()
_IS_WINDOWS = os.name == "nt"


def register_process_group(pgid: int) -> None:
    with _LOCK:
        _ACTIVE_PGIDS.add(pgid)


def unregister_process_group(pgid: int) -> None:
    with _LOCK:
        _ACTIVE_PGIDS.discard(pgid)


def _terminate_windows_process_tree(pid: int, *, force: bool) -> None:
    cmd = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        cmd.append("/F")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return
    if proc.returncode in {0, 128, 255}:
        return
    try:
        os.kill(pid, signal.SIGTERM if not force else signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_posix_process_group(pgid: int, sig: int) -> None:
    os.killpg(pgid, sig)


def terminate_registered_process_groups(*, grace_seconds: float = 1.0) -> None:
    with _LOCK:
        pgids = list(_ACTIVE_PGIDS)
    for pgid in pgids:
        try:
            if _IS_WINDOWS:
                _terminate_windows_process_tree(pgid, force=False)
            else:
                _terminate_posix_process_group(pgid, signal.SIGTERM)
        except ProcessLookupError:
            unregister_process_group(pgid)
    if grace_seconds > 0:
        time.sleep(grace_seconds)
    for pgid in pgids:
        try:
            if _IS_WINDOWS:
                _terminate_windows_process_tree(pgid, force=True)
            else:
                _terminate_posix_process_group(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            unregister_process_group(pgid)


def _signal_handler(signum, _frame) -> None:
    terminate_registered_process_groups(grace_seconds=0.2)
    raise SystemExit(128 + int(signum))


def install_worker_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


def run_subprocess(cmd: Sequence[str]) -> None:
    popen_kwargs: dict[str, object] = {}
    if _IS_WINDOWS:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(list(cmd), **popen_kwargs)
    register_process_group(proc.pid)
    try:
        rc = proc.wait()
    except BaseException:
        terminate_registered_process_groups(grace_seconds=0.2)
        raise
    finally:
        unregister_process_group(proc.pid)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, list(cmd))
