from __future__ import annotations

import pyroller.pipeline.execution_context as execution_context
from pyroller.pipeline.execution_context import PipelineExecutionContext


class DummyTranscriber:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class FailingCloseTranscriber(DummyTranscriber):
    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        raise RuntimeError("close failed")


def test_execution_context_reuses_transcriber_for_same_sanitized_config(monkeypatch) -> None:
    built: list[DummyTranscriber] = []

    def fake_build_transcriber(language: str, backend_name: str | None, config: dict[str, object]):
        transcriber = DummyTranscriber(f"{language}:{backend_name}:{config}")
        built.append(transcriber)
        return transcriber

    monkeypatch.setattr(execution_context, "build_transcriber", fake_build_transcriber)
    context = PipelineExecutionContext()
    try:
        first = context.get_transcriber(
            language="zh",
            backend_name="faster_whisper",
            config={"model_name": "turbo", "unsupported": "ignored"},
        )
        second = context.get_transcriber(
            language="zh",
            backend_name="faster_whisper",
            config={"model_name": "turbo"},
        )
    finally:
        context.close()

    assert first is second
    assert len(built) == 1
    assert first.close_calls == 1


def test_execution_context_switches_and_closes_old_transcriber(monkeypatch) -> None:
    built: list[DummyTranscriber] = []

    def fake_build_transcriber(language: str, backend_name: str | None, config: dict[str, object]):
        transcriber = DummyTranscriber(f"{language}:{backend_name}:{config}")
        built.append(transcriber)
        return transcriber

    monkeypatch.setattr(execution_context, "build_transcriber", fake_build_transcriber)
    context = PipelineExecutionContext()
    try:
        first = context.get_transcriber(language="zh", backend_name="faster_whisper", config={"model_name": "large-v2"})
        second = context.get_transcriber(language="zh", backend_name="faster_whisper", config={"model_name": "turbo"})
    finally:
        context.close()

    assert first is not second
    assert len(built) == 2
    assert first.close_calls == 1
    assert second.close_calls == 1


def test_execution_context_falls_back_unknown_language_to_multilingual(monkeypatch) -> None:
    calls: list[tuple[str, str | None, dict[str, object]]] = []

    def fake_build_transcriber(language: str, backend_name: str | None, config: dict[str, object]):
        calls.append((language, backend_name, dict(config)))
        return DummyTranscriber("fallback")

    monkeypatch.setattr(execution_context, "build_transcriber", fake_build_transcriber)
    context = PipelineExecutionContext()
    try:
        context.get_transcriber(language="xx", backend_name=None, config={})
    finally:
        context.close()

    assert calls == [("mul", "faster_whisper", {})]


def test_execution_context_close_swallows_transcriber_close_errors() -> None:
    context = PipelineExecutionContext()
    transcriber = FailingCloseTranscriber("bad")
    context._transcriber = transcriber
    context._transcriber_key = ("zh", "faster_whisper", "{}")

    context.close()

    assert transcriber.closed is True
    assert context._transcriber is None
    assert context._transcriber_key is None
