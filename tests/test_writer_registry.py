from __future__ import annotations

import pytest

from pyroller.writer.ass_karaoke import ASSKaraokeWriter
from pyroller.writer.lrc import LRCWriter
from pyroller.writer.registry import build_writer, list_available_writer_backends


def test_writer_registry_lists_stable_public_backends() -> None:
    assert list_available_writer_backends() == ("lrc_ms", "lrc_cs", "lrc_compressed", "ass_karaoke")


def test_writer_registry_builds_default_lrc_ms_writer() -> None:
    writer = build_writer(None, {"by_tag": "tests", "spacing": "drop"})

    assert isinstance(writer, LRCWriter)
    assert writer.backend_name == "lrc_ms"
    assert writer.by_tag == "tests"
    assert writer.keep_spacing is False


def test_writer_registry_builds_centisecond_lrc_writer() -> None:
    writer = build_writer("lrc_cs", {"spacing": "keep"})

    assert isinstance(writer, LRCWriter)
    assert writer.backend_name == "lrc_cs"
    assert writer.decimals == 2
    assert writer.keep_spacing is True


def test_writer_registry_requires_drop_spacing_for_compressed_lrc() -> None:
    with pytest.raises(ValueError, match="lrc_compressed"):
        build_writer("lrc_compressed", {"spacing": "keep"})

    writer = build_writer("lrc_compressed", {"spacing": "drop"})
    assert isinstance(writer, LRCWriter)
    assert writer.backend_name == "lrc_compressed"
    assert writer.compressed is True


def test_writer_registry_builds_ass_writer_with_config() -> None:
    writer = build_writer(
        "ass_karaoke",
        {
            "by_tag": "tests",
            "tag_type": "K",
            "spacing": "keep",
            "unmatched_line_duration": 1.25,
        },
    )

    assert isinstance(writer, ASSKaraokeWriter)
    assert writer.backend_name == "ass_karaoke"
    assert writer.by_tag == "tests"
    assert writer.tag_type == "K"
    assert writer.skip_structural_lines is False
    assert writer.unmatched_line_duration == 1.25


def test_writer_registry_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported writer backend"):
        build_writer("unknown", {})
