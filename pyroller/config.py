"""Stable file-engine options. Algorithm experiments are not runtime options."""
from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class EngineConfig:
    alignment_mode: str = field(default_factory=lambda: os.environ.get('PYROLLER_ALIGNMENT_MODE', 'ordered'))
    cache_dir: Path = Path.home() / '.cache/py-roller/v1'
    whisper_model: Path | None = None
    ctc_model: Path | None = None
    audio_mode: str = 'mix'
    ctc_device: str = 'auto'
    offline: bool = False
    cpu_threads: int = 4
    previews: bool = True

    def __post_init__(self):
        if self.alignment_mode not in ('unordered', 'ordered'):
            raise ValueError('alignment_mode must be unordered or ordered')
        if self.audio_mode not in ('mix', 'vocals'):
            raise ValueError('audio_mode must be mix or vocals')
        if self.ctc_device not in ('auto', 'cpu', 'mps'):
            raise ValueError('ctc_device must be auto, cpu or mps')
        if type(self.cpu_threads) is not int or self.cpu_threads < 1:
            raise ValueError('cpu_threads must be a positive integer')
        for key in ('offline', 'previews'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        for key in ('cache_dir', 'whisper_model', 'ctc_model'):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, Path(value).expanduser().resolve())
