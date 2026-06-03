from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyroller.transcriber.hf_download_config import HFDownloadConfig
from pyroller.transcriber.model_resolver import (
    AuxiliaryAssetPlan,
    TranscriberModelResolver,
    TranscriberResolutionPlan,
)


def test_auxiliary_asset_plan_serializes_paths() -> None:
    plan = AuxiliaryAssetPlan(role="lexicon", cache_root=Path("/tmp/cache"), required=True, local_only_checked=True)

    assert plan.to_dict() == {
        "role": "lexicon",
        "cache_root": "/tmp/cache",
        "required": True,
        "local_only_checked": True,
    }


def test_transcriber_resolution_plan_runtime_record_serializes_public_contract(tmp_path) -> None:
    aux = AuxiliaryAssetPlan(role="phonemizer", cache_root=tmp_path / "aux", required=True)
    plan = TranscriberResolutionPlan(
        backend="faster_whisper",
        language="zh",
        requested_model_name="turbo",
        effective_model_name="turbo",
        model_store_root=tmp_path,
        provider="faster_whisper",
        provider_kind="faster_whisper",
        local_files_only=True,
        network_required=False,
        network_allowed=False,
        network_reason=None,
        provider_cache_root=tmp_path / "providers",
        resolved_model_dir=tmp_path / "snapshot",
        download_root=tmp_path / "downloads",
        auxiliary_assets=[aux],
        validations=["ok"],
        hf_download={"xet": "off"},
    )

    record = plan.runtime_record()

    assert record["execution_location"] == "local"
    assert record["model_store_root"] == str(tmp_path)
    assert record["resolved_model_dir"] == str(tmp_path / "snapshot")
    assert record["auxiliary_assets"] == [aux.to_dict()]
    assert record["validations"] == ["ok"]
    assert record["hf_download"] == {"xet": "off"}


def test_model_resolver_resolves_explicit_local_path_without_network(tmp_path) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    store = tmp_path / "store"

    resolver = TranscriberModelResolver(
        backend="faster_whisper",
        language="zh",
        model_name=str(local_model),
        model_path=store,
        local_files_only=True,
        hf_download_config=HFDownloadConfig(xet="off"),
    )
    plan = resolver.resolve(materialize=False)

    assert plan.provider == "local_path"
    assert plan.provider_kind == "local"
    assert plan.resolved_model_dir == local_model.resolve()
    assert plan.network_required is False
    manifest = json.loads((store / "manifests" / "transcriber-index.json").read_text(encoding="utf-8"))
    assert manifest["models"][f"faster_whisper:{local_model}"]["provider"] == "local_path"


def test_model_resolver_rejects_missing_explicit_local_path(tmp_path) -> None:
    resolver = TranscriberModelResolver(
        backend="faster_whisper",
        language="zh",
        model_name=str(tmp_path / "missing-model"),
        model_path=tmp_path / "store",
    )

    with pytest.raises(FileNotFoundError, match="local path"):
        resolver.resolve(materialize=False)


def test_model_resolver_plans_faster_whisper_alias_without_materializing(tmp_path) -> None:
    resolver = TranscriberModelResolver(
        backend="faster_whisper",
        language="zh",
        model_name="turbo",
        model_path=tmp_path / "store",
        local_files_only=False,
        hf_download_config=HFDownloadConfig(xet="off", etag_timeout=5),
    )

    plan = resolver.resolve(materialize=False)

    assert plan.provider == "faster_whisper"
    assert plan.provider_kind == "faster_whisper"
    assert plan.effective_model_name == "turbo"
    assert plan.network_required is True
    assert plan.network_allowed is True
    assert plan.download_root == tmp_path / "store" / "providers" / "faster_whisper" / "hub"
    assert plan.hf_download["xet"] == "off"
    assert "would resolve faster-whisper snapshot on demand" in plan.validations


def test_model_resolver_reuses_manifest_entry_when_snapshot_exists(tmp_path) -> None:
    store = tmp_path / "store"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest_dir = store / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "transcriber-index.json").write_text(
        json.dumps(
            {
                "models": {
                    "faster_whisper:turbo": {
                        "resolved_model_dir": str(snapshot),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    plan = TranscriberModelResolver(
        backend="faster_whisper",
        language="zh",
        model_name="turbo",
        model_path=store,
    ).resolve(materialize=False)

    assert plan.resolved_model_dir == snapshot.resolve()
    assert plan.network_required is False
    assert any("reused resolved faster-whisper snapshot" in item for item in plan.validations)


def test_model_resolver_plans_transformers_snapshot_without_materializing(tmp_path) -> None:
    plan = TranscriberModelResolver(
        backend="mms_phonetic",
        language="zh",
        model_path=tmp_path / "store",
        local_files_only=True,
    ).resolve(materialize=False)

    assert plan.effective_model_name == "Chuatury/wav2vec2-mms-1b-cmn-phonetic"
    assert plan.provider == "hf_repo"
    assert plan.provider_kind == "transformers"
    assert plan.network_required is False
    assert plan.network_allowed is False
    assert "would resolve Hugging Face snapshot on demand" in plan.validations


def test_model_resolver_rejects_unknown_backend_without_default_model(tmp_path) -> None:
    resolver = TranscriberModelResolver(
        backend="custom",
        language="mul",
        model_path=tmp_path / "store",
    )

    with pytest.raises(ValueError, match="No default model"):
        resolver.resolve(materialize=False)
