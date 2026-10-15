from pathlib import Path
from dataclasses import replace

import pytest

from pyroller.backends import catalog
from pyroller.config import EngineConfig


def checkpoint(path):
    path.mkdir(parents=True)
    for name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json'):
        (path/name).write_bytes(name.encode())
    return path


def test_explicit_migration_and_cache_isolation(tmp_path, monkeypatch):
    home = tmp_path/'home'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    source = checkpoint(home/'.cache/py-roller/v1/models/large-v3-edaa852')
    config = EngineConfig(cache_dir=tmp_path/'custom', offline=True)
    assert catalog.local_model('whisper', config) is None
    with pytest.raises(FileNotFoundError): catalog.resolve_model('whisper', config)
    imported = catalog.migrate_model('whisper', config)
    assert imported == catalog.model_directory('whisper', config)
    assert catalog.fingerprint(imported, 'whisper') == catalog.fingerprint(source, 'whisper')
    assert catalog.resolve_model('whisper', config) == imported
    assert catalog.migrate_model('whisper', config) == imported
    assert source.exists()


def test_remove_preview_and_external_model_preserved(tmp_path):
    config = EngineConfig(cache_dir=tmp_path/'cache')
    managed = checkpoint(catalog.model_directory('whisper', config))
    external = checkpoint(tmp_path/'external')
    config = replace(config, whisper_model=external)
    catalog.remove_model('whisper', config)
    assert managed.exists()
    catalog.remove_model('whisper', config, apply=True)
    assert not managed.exists() and external.exists()


def test_atomic_import_failure_and_lock(tmp_path, monkeypatch):
    config = EngineConfig(cache_dir=tmp_path/'cache')
    source = checkpoint(tmp_path/'source')
    def fail(*args): raise OSError('disk full')
    monkeypatch.setattr(catalog, '_copy', fail)
    with pytest.raises(OSError): catalog.install_model('whisper', config, source)
    assert not catalog.model_directory('whisper', config).exists()
    assert not list((config.cache_dir/'models/whisper').iterdir())
    with catalog.model_lock(config):
        with pytest.raises(RuntimeError): catalog.install_model('whisper', config, source)


def test_demucs_bad_checksum_not_installed(tmp_path):
    source = tmp_path/catalog.DEMUCS_FILE
    source.write_bytes(b'bad weights')
    config = EngineConfig(cache_dir=tmp_path/'cache')
    with pytest.raises(ValueError, match='checksum'): catalog.install_model('demucs', config, source)
    assert catalog.local_model('demucs', config) is None


def test_download_uses_pinned_revision_and_managed_directory(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    seen = []
    def download(repo, revision, local_dir):
        seen.append((repo, revision))
        for name in ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json'):
            (local_dir/name).write_bytes(b'weights')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    config = EngineConfig(cache_dir=tmp_path)
    assert catalog.resolve_model('whisper', config) == catalog.model_directory('whisper', config)
    assert seen == [catalog.MODELS['whisper'][:2]]


def test_offline_demucs_passes_local_repo(tmp_path, monkeypatch):
    from pyroller.backends import separation
    config = EngineConfig(cache_dir=tmp_path, offline=True)
    repo = tmp_path/'repo'; repo.mkdir()
    target = tmp_path/'vocals.wav'; target.write_bytes(b'wave')
    calls = []
    def resolve(kind, settings):
        assert kind == 'demucs' and settings.offline
        return repo
    monkeypatch.setattr(separation, 'resolve_model', resolve)
    monkeypatch.setattr(separation.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(separation, 'decode_audio', lambda path: None)
    separation._run_demucs(tmp_path/'input.flac', tmp_path, target, config)
    assert calls[-1][calls[-1].index('--repo')+1] == str(repo)


def test_remove_rejects_symlink(tmp_path):
    config = EngineConfig(cache_dir=tmp_path/'cache')
    target = catalog.model_directory('whisper', config)
    target.parent.mkdir(parents=True)
    external = checkpoint(tmp_path/'external')
    target.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError): catalog.remove_model('whisper', config, apply=True)
    assert external.exists()
