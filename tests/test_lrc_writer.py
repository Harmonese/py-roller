from __future__ import annotations

from pyroller.writer.lrc import LRCWriter

from .factories import make_alignment_result, make_repeated_alignment_result


def test_lrc_writer_keeps_structural_spacing_when_requested(tmp_path) -> None:
    output = tmp_path / "song.lrc"
    result = LRCWriter(decimals=3, keep_spacing=True, by_tag="tests").write(make_alignment_result(), output)

    text = output.read_text(encoding="utf-8")
    assert result.writer_backend == "lrc_ms"
    assert "[by:tests]" in text
    assert "[00:01.000] 你好" in text
    assert "[00:02.500]" in text
    assert "[00:03.000] 世界" in text
    assert result.metadata["line_count"] == 3


def test_lrc_writer_drops_structural_spacing_by_default(tmp_path) -> None:
    output = tmp_path / "song.lrc"
    LRCWriter(decimals=2, keep_spacing=False).write(make_alignment_result(), output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert "[00:01.00] 你好" in lines
    assert "[00:03.00] 世界" in lines
    assert "[00:02.50]" not in lines


def test_compressed_lrc_groups_repeated_lines(tmp_path) -> None:
    output = tmp_path / "song.lrc"
    result = LRCWriter(decimals=2, compressed=True).write(make_repeated_alignment_result(), output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert "[00:01.00][00:05.00] 你好" in lines
    assert "[00:03.00] 世界" in lines
    assert result.writer_backend == "lrc_compressed"
    assert result.metadata["line_count"] == 2
