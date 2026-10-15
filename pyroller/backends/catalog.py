"""Fixed, locally resolved neural backends and content-addressed evidence."""
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from ..hybrid.io import save_json, sha256

MODELS = {
    'whisper': ('Systran/faster-whisper-large-v3', 'edaa852ec7e145841d8ffdb056a99866b5f0a478', 'faster_whisper'),
    'ctc': ('facebook/wav2vec2-large-960h-lv60-self', '54074b1c16f4de6a5ad59affb4caa8f2ea03a119', 'huggingface'),
}
DEMUCS_FILE = '955717e8-8726e21a.th'
MODELS['demucs'] = ('htdemucs', '955717e8-8726e21a', 'demucs')
WHISPER_OPTIONS = dict(language='en', task='transcribe', beam_size=5, temperature=0.,
    condition_on_previous_text=False, initial_prompt=None, word_timestamps=True,
    vad_filter=False, no_speech_threshold=None, max_new_tokens=224)


def checkpoint_available(path, kind):
    path = Path(path)
    if kind == 'demucs':
        return (path/DEMUCS_FILE).is_file() and (path/'htdemucs.yaml').is_file()
    if kind == 'whisper':
        return (all((path/name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json'))
                and any((path/name).is_file() for name in ('vocabulary.json', 'vocabulary.txt')))
    return (all((path/name).is_file() for name in ('config.json', 'vocab.json', 'preprocessor_config.json'))
            and bool(list(path.glob('*.safetensors')) or list(path.glob('pytorch_model*.bin'))))


def model_directory(kind, config):
    return Path(config.cache_dir)/'models'/kind/MODELS[kind][1]


def local_model(kind, config):
    explicit = getattr(config, kind+'_model', None)
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir(): raise ValueError(f'Model directory does not exist: {path}')
        return path
    path = model_directory(kind, config)
    if checkpoint_available(path, kind): return path
    return None


@contextmanager
def model_lock(config):
    root = Path(config.cache_dir)/'models'
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'.management.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Model management is busy: {lock}') from exc
    try:
        yield
    finally:
        lock.rmdir()


def _copy(source, target):
    # Hard links avoid duplicating gigabytes; cross-volume imports use copies.
    try:
        os.link(Path(source).resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def legacy_model(kind, config):
    repo, revision, provider = MODELS[kind]
    if kind == 'demucs':
        torch_root = Path(os.environ.get('TORCH_HOME', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'torch')))
        path = torch_root/'hub/checkpoints'/DEMUCS_FILE
        return path if path.is_file() else None
    roots = [Path(config.cache_dir), Path.home()/'.cache/py-roller/v1']
    candidates = [root/'models/large-v3-edaa852' for root in roots] if kind == 'whisper' else []
    hubs = [root/'models' for root in roots]
    hubs.append(Path.home()/'.cache/py-roller/models/transcriber/providers'/provider/'hub')
    candidates += [root/('models--'+repo.replace('/', '--'))/'snapshots'/revision for root in hubs]
    return next((path for path in candidates if checkpoint_available(path, kind)), None)


def install_model(kind, config, source=None):
    target = model_directory(kind, config)
    with model_lock(config):
        if checkpoint_available(target, kind): return target
        if target.exists(): raise ValueError(f'Incomplete managed model; remove it before retrying: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.install-', dir=target.parent) as temporary:
            stage = Path(temporary)/'checkpoint'
            stage.mkdir()
            if kind == 'demucs':
                if source:
                    _copy(source, stage/DEMUCS_FILE)
                else:
                    from torch.hub import download_url_to_file
                    download_url_to_file('https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/'+DEMUCS_FILE,
                                         str(stage/DEMUCS_FILE), hash_prefix='8726e21a')
                if not sha256(stage/DEMUCS_FILE).startswith('8726e21a'):
                    raise ValueError('Demucs checkpoint checksum mismatch')
                (stage/'htdemucs.yaml').write_text("models: ['955717e8']\n")
            else:
                if source is None:
                    from huggingface_hub import snapshot_download
                    repo, revision, _ = MODELS[kind]
                    snapshot_download(repo, revision=revision, local_dir=stage)
                else:
                    shutil.copytree(source, stage, dirs_exist_ok=True, copy_function=_copy)
            if not checkpoint_available(stage, kind): raise ValueError(f'Incomplete {kind} checkpoint')
            stage.rename(target)
    return target


def migrate_model(kind, config):
    if local_model(kind, config): return local_model(kind, config)
    source = legacy_model(kind, config)
    if source is None: return None
    return install_model(kind, config, source)


def remove_model(kind, config, apply=False):
    target = model_directory(kind, config)
    if apply:
        with model_lock(config):
            if target.is_symlink() or target.resolve() != target.absolute():
                raise ValueError('Refusing to remove a redirected model directory')
            if target.exists(): shutil.rmtree(target)
    return target


def resolve_model(kind, config):
    local = local_model(kind, config)
    if local is not None:
        if not checkpoint_available(local, kind):
            raise ValueError(f'Incomplete {kind} checkpoint: {local}')
        return local
    if config.offline:
        raise FileNotFoundError(f'Missing {kind} model; run models migrate or models download: {model_directory(kind, config)}')
    return install_model(kind, config)


def fingerprint(path, kind):
    if kind == 'whisper':
        files = [path/'model.bin', path/'config.json', path/'tokenizer.json',
                 path/'vocabulary.json', path/'vocabulary.txt', path/'preprocessor_config.json']
        if not (path/'model.bin').is_file(): raise ValueError(f'Whisper checkpoint has no model.bin: {path}')
    else:
        weights = sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
        if not weights: raise ValueError(f'CTC checkpoint has no weights: {path}')
        files = [*weights, path/'config.json', path/'preprocessor_config.json', path/'vocab.json']
    return {p.name: sha256(p) for p in files if p.exists()}


def evidence_directory(root, metadata):
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    directory = Path(root)/key
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory/'manifest.json', metadata)
    return directory
