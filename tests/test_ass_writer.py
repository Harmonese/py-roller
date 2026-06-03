from __future__ import annotations

from pyroller.writer.ass_karaoke import ASSKaraokeWriter

from .factories import make_alignment_result


def test_ass_writer_emits_karaoke_dialogue_and_skips_spacing(tmp_path) -> None:
    output = tmp_path / "song.ass"
    result = ASSKaraokeWriter(by_tag="tests", tag_type="kf", skip_structural_lines=True).write(
        make_alignment_result(),
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert "ScriptType: v4.00+" in text
    assert "Dialogue: 0,0:00:01.00,0:00:02.00" in text
    assert "{\\kf45}你{\\kf55}好" in text
    assert "{\\kf50}世{\\kf50}界" in text
    assert result.writer_backend == "ass_karaoke"
    assert result.metadata["line_count"] == 2


def test_ass_writer_escapes_override_characters(tmp_path) -> None:
    alignment = make_alignment_result()
    alignment.lines[0].aligned_units = []
    alignment.lines[0].raw_text = r"a{b}\c"

    output = tmp_path / "escaped.ass"
    ASSKaraokeWriter(skip_structural_lines=True).write(alignment, output)

    assert r"a\{b\}\\c" in output.read_text(encoding="utf-8")
