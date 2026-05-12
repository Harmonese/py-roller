from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import LyricsDocument, ParsedLyrics


class LyricsParser(ABC):
    name = "parser"
    accepts = ("lyrics_text",)
    produces = "parsed_lyrics"

    @abstractmethod
    def parse(self, lyrics_document: LyricsDocument, language: str, tone_mode: str) -> ParsedLyrics:
        raise NotImplementedError
