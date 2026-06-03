from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("PYROLLER_LANG", "en")

from pyroller.domain import PipelineRequest


@pytest.fixture
def base_request(tmp_path: Path) -> PipelineRequest:
    return PipelineRequest(
        stages=["w"],
        intermediate_dir=tmp_path / "intermediate",
        output_roller_path=tmp_path / "out.lrc",
        backend_config={},
    )
