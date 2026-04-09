from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.filter.base import AudioFilter
from pyroller.utils.ids import make_id

logger = logging.getLogger("pyroller.filter")


class AdaptiveNoiseGateFilter(AudioFilter):
    name = "noise_gate"

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        threshold_percentile: float = 20.0,
        threshold_bias_db: float = 4.0,
        min_threshold_db: float = -55.0,
        max_threshold_db: float = -22.0,
        hangover_frames: int = 4,
        **_: Any,
    ) -> None:
        self.frame_length = max(128, int(frame_length))
        self.hop_length = max(64, int(hop_length))
        self.threshold_percentile = float(threshold_percentile)
        self.threshold_bias_db = float(threshold_bias_db)
        self.min_threshold_db = float(min_threshold_db)
        self.max_threshold_db = float(max_threshold_db)
        self.hangover_frames = max(0, int(hangover_frames))

    def process(self, audio_artifact: AudioArtifact, output_dir: Path) -> AudioArtifact:
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "noise_gate dependencies are not installed. Install with: pip install numpy soundfile"
            ) from exc

        if audio_artifact.path is None:
            raise ValueError("noise_gate requires an audio artifact with a concrete path")

        source_path = Path(audio_artifact.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found for noise_gate filter: {source_path}")

        audio, sample_rate = sf.read(str(source_path), always_2d=True)
        if audio.size == 0:
            logger.warning("noise_gate received empty audio at %s; forwarding unchanged", source_path)
            return audio_artifact

        mono = audio.mean(axis=1)
        frame_db = self._frame_rms_db(np, mono)
        threshold_db = self._estimate_threshold_db(np, frame_db)
        keep_mask = self._build_keep_mask(np, frame_db, threshold_db)
        sample_mask = self._expand_keep_mask(np, keep_mask, len(mono))
        gated = audio.copy()
        gated[~sample_mask, :] = 0.0

        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{source_path.stem}.noise_gate{source_path.suffix}"
        sf.write(str(destination), gated, sample_rate)

        suppressed_ratio = float(1.0 - keep_mask.mean()) if keep_mask.size else 0.0
        logger.info(
            "Applied adaptive noise gate to %s -> %s | threshold_db=%.2f suppressed_frames=%.2f%%",
            source_path,
            destination,
            threshold_db,
            suppressed_ratio * 100.0,
        )
        metadata = dict(audio_artifact.metadata)
        metadata.update(
            {
                "backend": self.name,
                "source_artifact_id": audio_artifact.artifact_id,
                "source_audio": str(source_path),
                "sample_rate": sample_rate,
                "frame_length": self.frame_length,
                "hop_length": self.hop_length,
                "threshold_percentile": self.threshold_percentile,
                "threshold_bias_db": self.threshold_bias_db,
                "threshold_db": threshold_db,
                "suppressed_frame_ratio": suppressed_ratio,
            }
        )
        return AudioArtifact(
            artifact_id=make_id("artifact"),
            stage="filter",
            kind="audio",
            path=destination,
            sample_rate=sample_rate,
            channels=int(audio.shape[1]),
            duration=float(len(audio) / sample_rate) if sample_rate else None,
            role="filtered_vocal_audio",
            metadata=metadata,
        )

    def _frame_rms_db(self, np: Any, mono: Any) -> Any:
        if len(mono) < self.frame_length:
            rms = np.sqrt(np.mean(np.square(mono), dtype=np.float64))
            return np.array([20.0 * np.log10(max(float(rms), 1e-8))], dtype=np.float64)

        frame_count = 1 + max(0, (len(mono) - self.frame_length) // self.hop_length)
        rms_values = np.empty(frame_count, dtype=np.float64)
        for index in range(frame_count):
            start = index * self.hop_length
            end = start + self.frame_length
            frame = mono[start:end]
            rms = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
            rms_values[index] = 20.0 * np.log10(max(float(rms), 1e-8))
        return rms_values

    def _estimate_threshold_db(self, np: Any, frame_db: Any) -> float:
        percentile_floor = float(np.percentile(frame_db, self.threshold_percentile))
        median_db = float(np.percentile(frame_db, 50.0))
        dynamic_threshold = min(percentile_floor + self.threshold_bias_db, median_db - 6.0)
        threshold_db = max(self.min_threshold_db, min(dynamic_threshold, self.max_threshold_db))
        return float(threshold_db)

    def _build_keep_mask(self, np: Any, frame_db: Any, threshold_db: float) -> Any:
        keep = frame_db >= threshold_db
        if not keep.any() and keep.size:
            keep[int(frame_db.argmax())] = True
        if self.hangover_frames <= 0 or keep.size == 0:
            return keep
        smoothed = keep.copy()
        for index in range(len(keep)):
            if keep[index]:
                start = max(0, index - self.hangover_frames)
                end = min(len(keep), index + self.hangover_frames + 1)
                smoothed[start:end] = True
        return smoothed

    def _expand_keep_mask(self, np: Any, keep_mask: Any, sample_count: int) -> Any:
        if keep_mask.size == 0:
            return np.ones(sample_count, dtype=bool)
        sample_mask = np.zeros(sample_count, dtype=bool)
        for index, keep in enumerate(keep_mask):
            start = index * self.hop_length
            end = min(sample_count, start + self.frame_length)
            if keep:
                sample_mask[start:end] = True
        if not sample_mask.any():
            sample_mask[:] = True
        return sample_mask
