from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyroller.progress import StageProgress

logger = logging.getLogger("pyroller.transcriber")

_DEFAULT_MODEL_NAME_BY_BACKEND = {
    "mms_phonetic": "Chuatury/wav2vec2-mms-1b-cmn-phonetic",
    "wav2vec2_phoneme": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "whisperx": "large-v2",
}


def default_transcriber_model_store_root() -> Path:
    return Path.home() / ".cache" / "py-roller" / "models" / "transcriber"


@dataclass(slots=True)
class AuxiliaryAssetPlan:
    role: str
    cache_root: Path | None = None
    required: bool = False
    local_only_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "cache_root": str(self.cache_root) if self.cache_root is not None else None,
            "required": self.required,
            "local_only_checked": self.local_only_checked,
        }


@dataclass(slots=True)
class TranscriberResolutionPlan:
    backend: str
    language: str
    requested_model_name: str | None
    effective_model_name: str
    model_store_root: Path
    provider: str
    provider_kind: str
    local_files_only: bool
    network_required: bool
    network_allowed: bool
    network_reason: str | None
    provider_cache_root: Path | None = None
    resolved_model_dir: Path | None = None
    download_root: Path | None = None
    auxiliary_assets: list[AuxiliaryAssetPlan] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)

    def runtime_record(self) -> dict[str, Any]:
        return {
            "execution_location": "local",
            "backend": self.backend,
            "language": self.language,
            "requested_model_name": self.requested_model_name,
            "effective_model_name": self.effective_model_name,
            "model_store_root": str(self.model_store_root),
            "provider": self.provider,
            "provider_kind": self.provider_kind,
            "provider_cache_root": str(self.provider_cache_root) if self.provider_cache_root is not None else None,
            "resolved_model_dir": str(self.resolved_model_dir) if self.resolved_model_dir is not None else None,
            "download_root": str(self.download_root) if self.download_root is not None else None,
            "local_files_only": self.local_files_only,
            "network_required": self.network_required,
            "network_allowed": self.network_allowed,
            "network_reason": self.network_reason,
            "auxiliary_assets": [item.to_dict() for item in self.auxiliary_assets],
            "validations": list(self.validations),
        }


class TranscriberModelResolver:
    def __init__(
        self,
        *,
        backend: str,
        language: str,
        model_name: str | None = None,
        model_path: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.backend = str(backend).strip().lower()
        self.language = str(language).strip().lower()
        self.requested_model_name = str(model_name).strip() if model_name is not None and str(model_name).strip() else None
        self.model_store_root = Path(model_path).expanduser().resolve() if model_path is not None else default_transcriber_model_store_root().expanduser().resolve()
        self.local_files_only = bool(local_files_only)

    def resolve(self, *, materialize: bool, stage: StageProgress | None = None) -> TranscriberResolutionPlan:
        self._ensure_store_layout()
        effective_model_name = self.requested_model_name or _DEFAULT_MODEL_NAME_BY_BACKEND.get(self.backend)
        if not effective_model_name:
            raise ValueError(f"No default model is defined for transcriber backend {self.backend!r}. Please pass --transcriber-model-name.")

        local_ref = self._resolve_explicit_local_path(effective_model_name)
        if local_ref is not None:
            if not local_ref.exists():
                raise FileNotFoundError(
                    f"Transcriber model reference points to a local path that does not exist: {local_ref}"
                )
            plan = TranscriberResolutionPlan(
                backend=self.backend,
                language=self.language,
                requested_model_name=self.requested_model_name,
                effective_model_name=effective_model_name,
                model_store_root=self.model_store_root,
                provider="local_path",
                provider_kind="local",
                local_files_only=self.local_files_only,
                network_required=False,
                network_allowed=not self.local_files_only,
                network_reason=None,
                resolved_model_dir=local_ref,
            )
            plan.validations.append("resolved explicit local model path")
            self._write_manifest(plan)
            return plan

        if self.backend in {"mms_phonetic", "wav2vec2_phoneme"}:
            return self._resolve_hf_snapshot(effective_model_name, materialize=materialize, stage=stage)
        if self.backend == "whisperx":
            return self._resolve_whisperx_model(effective_model_name, materialize=materialize, stage=stage)
        raise ValueError(f"Unsupported transcriber backend for model resolution: {self.backend!r}")

    def _resolve_hf_snapshot(self, model_name: str, *, materialize: bool, stage: StageProgress | None = None) -> TranscriberResolutionPlan:
        provider_cache_root = self.model_store_root / "providers" / "huggingface" / "hub"
        provider_cache_root.mkdir(parents=True, exist_ok=True)
        plan = TranscriberResolutionPlan(
            backend=self.backend,
            language=self.language,
            requested_model_name=self.requested_model_name,
            effective_model_name=model_name,
            model_store_root=self.model_store_root,
            provider="hf_repo",
            provider_kind="transformers",
            local_files_only=self.local_files_only,
            network_required=not self.local_files_only,
            network_allowed=not self.local_files_only,
            network_reason=(None if self.local_files_only else "resolve or download Hugging Face snapshot"),
            provider_cache_root=provider_cache_root,
        )
        existing = self._read_manifest_entry(self.backend, model_name)
        existing_dir = Path(existing.get("resolved_model_dir")).resolve() if existing and existing.get("resolved_model_dir") else None
        if existing_dir is not None and existing_dir.exists():
            plan.resolved_model_dir = existing_dir
            plan.network_required = False
            plan.validations.append("reused resolved model snapshot already present in py-roller model store")
            self._write_manifest(plan)
            return plan

        if materialize:
            from pyroller.transcriber.download_logging import snapshot_download_with_logging

            try:
                snapshot_dir = Path(
                    snapshot_download_with_logging(
                        repo_id=model_name,
                        cache_dir=provider_cache_root,
                        local_files_only=self.local_files_only,
                        log_label=f"transcriber model {model_name}",
                        stage=stage,
                    )
                ).resolve()
            except Exception as exc:
                guidance = (
                    f"Unable to resolve transcriber model {model_name!r} into the local py-roller model store at {provider_cache_root}. "
                    f"If you are on a restricted network, pre-download the model into {self.model_store_root} or pass --transcriber-local-files-only once the cache is ready."
                )
                raise RuntimeError(guidance) from exc
            plan.resolved_model_dir = snapshot_dir
            plan.network_required = False
            plan.validations.append("resolved Hugging Face snapshot into local py-roller model store")
        else:
            plan.validations.append("would resolve Hugging Face snapshot on demand")
        self._write_manifest(plan)
        return plan

    def _resolve_whisperx_model(self, model_name: str, *, materialize: bool, stage: StageProgress | None = None) -> TranscriberResolutionPlan:
        provider_cache_root = self.model_store_root / "providers" / "whisperx"
        main_download_root = provider_cache_root / "asr"
        aux_hf_home = provider_cache_root / "hf_home"
        (aux_hf_home / "hub").mkdir(parents=True, exist_ok=True)
        main_download_root.mkdir(parents=True, exist_ok=True)
        plan = TranscriberResolutionPlan(
            backend=self.backend,
            language=self.language,
            requested_model_name=self.requested_model_name,
            effective_model_name=model_name,
            model_store_root=self.model_store_root,
            provider="whisperx_builtin",
            provider_kind="whisperx",
            local_files_only=self.local_files_only,
            network_required=not self.local_files_only,
            network_allowed=not self.local_files_only,
            network_reason=(None if self.local_files_only else "allow WhisperX to materialize ASR and auxiliary models"),
            provider_cache_root=aux_hf_home / "hub",
            download_root=main_download_root,
            auxiliary_assets=[
                AuxiliaryAssetPlan(role="align", cache_root=aux_hf_home / "hub", required=False, local_only_checked=self.local_files_only),
                AuxiliaryAssetPlan(role="vad", cache_root=aux_hf_home / "hub", required=False, local_only_checked=self.local_files_only),
            ],
        )
        repo_id = self._resolve_whisperx_repo_id(model_name)
        existing = self._read_manifest_entry(self.backend, model_name)
        existing_dir = Path(existing.get("resolved_model_dir")).resolve() if existing and existing.get("resolved_model_dir") else None
        if existing_dir is not None and existing_dir.exists():
            plan.resolved_model_dir = existing_dir
            plan.network_required = False
            plan.validations.append("reused resolved WhisperX ASR snapshot already present in py-roller model store")
            self._write_manifest(plan)
            return plan

        if materialize:
            from pyroller.transcriber.download_logging import snapshot_download_with_logging

            try:
                snapshot_dir = Path(
                    snapshot_download_with_logging(
                        repo_id=repo_id,
                        cache_dir=main_download_root,
                        local_files_only=self.local_files_only,
                        log_label=f"WhisperX ASR model {repo_id}",
                        stage=stage,
                    )
                ).resolve()
            except Exception as exc:
                guidance = (
                    f"Unable to resolve WhisperX ASR model {repo_id!r} into the local py-roller model store at {main_download_root}. "
                    f"If you are on a restricted network, pre-download the model into {self.model_store_root} or pass --transcriber-local-files-only once the cache is ready."
                )
                raise RuntimeError(guidance) from exc
            plan.resolved_model_dir = snapshot_dir
            plan.network_required = False
            plan.validations.append("resolved WhisperX ASR snapshot into local py-roller model store")
        else:
            plan.validations.append("would resolve WhisperX ASR snapshot on demand")
        self._write_manifest(plan)
        return plan

    def _resolve_whisperx_repo_id(self, model_name: str) -> str:
        if "/" in model_name or "\\" in model_name:
            return model_name
        return f"Systran/faster-whisper-{model_name}"

    def _resolve_explicit_local_path(self, model_name: str) -> Path | None:
        candidate = Path(model_name).expanduser()
        if candidate.is_absolute() or model_name.startswith("."):
            return candidate.resolve()
        if "/" not in model_name and "\\" not in model_name:
            return None
        if candidate.exists():
            return candidate.resolve()
        return None


    def _read_manifest_entry(self, backend: str, model_name: str) -> dict[str, Any] | None:
        manifest_path = self.model_store_root / "manifests" / "transcriber-index.json"
        if not manifest_path.exists():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        models = data.get("models", {})
        entry = models.get(f"{backend}:{model_name}")
        return entry if isinstance(entry, dict) else None

    def _ensure_store_layout(self) -> None:
        self.model_store_root.mkdir(parents=True, exist_ok=True)
        (self.model_store_root / "providers").mkdir(parents=True, exist_ok=True)
        (self.model_store_root / "manifests").mkdir(parents=True, exist_ok=True)

    def _write_manifest(self, plan: TranscriberResolutionPlan) -> None:
        manifest_path = self.model_store_root / "manifests" / "transcriber-index.json"
        data: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        models = data.setdefault("models", {})
        key = f"{plan.backend}:{plan.effective_model_name}"
        models[key] = plan.runtime_record()
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@contextlib.contextmanager
def transcriber_provider_environment(plan: TranscriberResolutionPlan):
    previous: dict[str, str | None] = {}
    overrides: dict[str, str] = {}
    if plan.provider_kind == "whisperx":
        hf_home = plan.download_root.parent / "hf_home" if plan.download_root is not None else plan.model_store_root / "providers" / "whisperx" / "hf_home"
        hub_root = hf_home / "hub"
        hub_root.mkdir(parents=True, exist_ok=True)
        (hf_home / "transformers").mkdir(parents=True, exist_ok=True)
        overrides["HF_HOME"] = str(hf_home)
        overrides["HF_HUB_CACHE"] = str(hub_root)
        overrides["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    if plan.local_files_only:
        overrides["HF_HUB_OFFLINE"] = "1"
        overrides["TRANSFORMERS_OFFLINE"] = "1"
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
