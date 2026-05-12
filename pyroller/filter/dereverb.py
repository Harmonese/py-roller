from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.filter.base import AudioFilter
from pyroller.utils.ids import make_id

logger = logging.getLogger("pyroller.filter")


class DereverbFilter(AudioFilter):
    name = "dereverb"

    def __init__(
        self,
        stft_size: int = 1024,
        stft_shift: int = 256,
        taps: int = 10,
        delay: int = 3,
        iterations: int = 3,
        psd_context: int = 0,
        statistics_mode: str = "full",
        clip_output: bool = True,
        max_abs_amplitude: float = 0.999,
        **_: Any,
    ) -> None:
        self.stft_size = max(128, int(stft_size))
        self.stft_shift = max(32, int(stft_shift))
        self.taps = max(1, int(taps))
        self.delay = max(0, int(delay))
        self.iterations = max(1, int(iterations))
        self.psd_context = max(0, int(psd_context))
        self.statistics_mode = str(statistics_mode)
        self.clip_output = bool(clip_output)
        self.max_abs_amplitude = max(0.1, float(max_abs_amplitude))

    def process(self, audio_artifact: AudioArtifact, output_dir: Path) -> AudioArtifact:
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
            from nara_wpe.utils import istft, stft  # type: ignore
            from nara_wpe.wpe import wpe_v8  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "dereverb dependencies are not installed. Install with: pip install nara_wpe numpy soundfile scipy bottleneck"
            ) from exc

        if audio_artifact.path is None:
            raise ValueError("dereverb requires an audio artifact with a concrete path")

        source_path = Path(audio_artifact.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found for dereverb filter: {source_path}")

        audio, sample_rate = sf.read(str(source_path), always_2d=True)
        if audio.size == 0:
            logger.warning("dereverb received empty audio at %s; forwarding unchanged", source_path)
            return audio_artifact

        time_major = np.asarray(audio, dtype=np.float64)
        channel_major = time_major.T
        original_samples = int(channel_major.shape[1])
        input_peak = float(np.max(np.abs(channel_major))) if channel_major.size else 0.0

        if original_samples < self.stft_size:
            logger.warning(
                "dereverb input shorter than stft_size (%d < %d); forwarding unchanged",
                original_samples,
                self.stft_size,
            )
            return audio_artifact

        try:
            Y = stft(channel_major, size=self.stft_size, shift=self.stft_shift)
            if Y.ndim != 3:
                raise RuntimeError(f"Unexpected STFT rank from nara_wpe: {Y.shape}")
            Y_fdt = np.transpose(Y, (2, 0, 1))
            X_fdt = wpe_v8(
                Y_fdt,
                taps=self.taps,
                delay=self.delay,
                iterations=self.iterations,
                psd_context=self.psd_context,
                statistics_mode=self.statistics_mode,
                inplace=False,
            )
            X = np.transpose(X_fdt, (1, 2, 0))
            dereverb = istft(X, size=self.stft_size, shift=self.stft_shift)
        except Exception as exc:
            raise RuntimeError(f"dereverb failed while processing {source_path}: {exc}") from exc

        dereverb = np.asarray(dereverb, dtype=np.float64)
        if dereverb.ndim == 1:
            dereverb = dereverb[None, :]
        if dereverb.shape[1] < original_samples:
            pad = original_samples - dereverb.shape[1]
            dereverb = np.pad(dereverb, ((0, 0), (0, pad)), mode="constant")
        dereverb = dereverb[:, :original_samples]

        output_peak_preclip = float(np.max(np.abs(dereverb))) if dereverb.size else 0.0
        if self.clip_output and output_peak_preclip > self.max_abs_amplitude:
            dereverb = np.clip(dereverb, -self.max_abs_amplitude, self.max_abs_amplitude)

        processed = dereverb.T.astype(audio.dtype, copy=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{source_path.stem}.dereverb{source_path.suffix}"
        sf.write(str(destination), processed, sample_rate)

        output_peak = float(np.max(np.abs(dereverb))) if dereverb.size else 0.0
        logger.info(
            "Applied nara_wpe dereverberation to %s -> %s | channels=%d sr=%d stft=%d/%d taps=%d delay=%d iterations=%d",
            source_path,
            destination,
            int(processed.shape[1]) if processed.ndim == 2 else 1,
            sample_rate,
            self.stft_size,
            self.stft_shift,
            self.taps,
            self.delay,
            self.iterations,
        )
        metadata = dict(audio_artifact.metadata)
        metadata.update(
            {
                "backend": self.name,
                "implementation": "nara_wpe",
                "source_artifact_id": audio_artifact.artifact_id,
                "source_audio": str(source_path),
                "sample_rate": int(sample_rate),
                "channels": int(processed.shape[1]) if processed.ndim == 2 else 1,
                "stft_size": self.stft_size,
                "stft_shift": self.stft_shift,
                "taps": self.taps,
                "delay": self.delay,
                "iterations": self.iterations,
                "psd_context": self.psd_context,
                "statistics_mode": self.statistics_mode,
                "input_peak": input_peak,
                "output_peak": output_peak,
                "clip_output": self.clip_output,
            }
        )
        return AudioArtifact(
            artifact_id=make_id("artifact"),
            stage="filter",
            kind="audio",
            path=destination,
            sample_rate=int(sample_rate),
            channels=int(processed.shape[1]) if processed.ndim == 2 else 1,
            duration=float(processed.shape[0] / sample_rate) if sample_rate else None,
            role="filtered_vocal_audio",
            metadata=metadata,
        )
