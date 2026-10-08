"""Per-execution scratch directories; markers never grant deletion authority."""
from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from pyroller.domain import PipelineRequest


class RunWorkspace:
    def __init__(self, request: PipelineRequest) -> None:
        root = request.intermediate_dir.expanduser().resolve()
        try:
            root.mkdir(parents=True, exist_ok=False)
            self.created_root = True
        except FileExistsError:
            self.created_root = False
            if not root.is_dir():
                raise ValueError(f"Intermediate root is not a directory: {root}")
        self.root = root
        self.path = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        stat = self.path.stat()
        self.identity = (stat.st_dev, stat.st_ino)
        self.token = uuid.uuid4().hex
        self.marker = self.path / ".py-roller-owner.json"
        self.marker.write_text(json.dumps({"owned_by": "py-roller", "token": self.token}), encoding="utf-8")
        self.request = replace(request, intermediate_dir=self.path)
        try:
            for field in fields(request):
                value = getattr(request, field.name)
                if field.name.endswith("_path") and isinstance(value, Path):
                    resolved = value.expanduser().resolve()
                    if resolved == self.path or self.path in resolved.parents:
                        raise ValueError(f"Input/output path is inside the cleanup directory: {value}")
        except BaseException:
            self.cleanup()
            raise

    def cleanup(self) -> bool:
        if not self.path.exists():
            return True
        # Check both the in-memory identity and token. A pre-existing marker,
        # replaced directory or symlink can never authorize recursive deletion.
        if self.path.is_symlink():
            return False
        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino) != self.identity or self.marker.is_symlink():
            return False
        try:
            marker = json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(marker, dict) or marker.get("token") != self.token:
            return False
        shutil.rmtree(self.path)
        if self.created_root:
            try:
                self.root.rmdir()  # Only remove an empty root created by us.
            except OSError:
                pass
        return True
