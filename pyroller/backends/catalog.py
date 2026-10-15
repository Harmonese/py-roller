"""Fixed, locally resolved neural backends and content-addressed evidence."""
import hashlib
import json
from pathlib import Path
from ..hybrid.io import save_json, sha256

MODELS = {
    'whisper': ('Systran/faster-whisper-large-v3', 'edaa852ec7e145841d8ffdb056a99866b5f0a478', 'faster_whisper'),
    'ctc': ('facebook/wav2vec2-large-960h-lv60-self', '54074b1c16f4de6a5ad59affb4caa8f2ea03a119', 'huggingface'),
}
WHISPER_OPTIONS = dict(language='en', task='transcribe', beam_size=5, temperature=0.,
    condition_on_previous_text=False, initial_prompt=None, word_timestamps=True,
    vad_filter=False, no_speech_threshold=None, max_new_tokens=224)


def checkpoint_available(path, kind):
    path = Path(path)
    if kind == 'whisper':
        return (all((path/name).is_file() for name in ('model.bin', 'config.json', 'tokenizer.json'))
                and any((path/name).is_file() for name in ('vocabulary.json', 'vocabulary.txt')))
    return (all((path/name).is_file() for name in ('config.json', 'vocab.json', 'preprocessor_config.json'))
            and bool(list(path.glob('*.safetensors')) or list(path.glob('pytorch_model*.bin'))))


def local_model(kind, config):
    explicit = config.whisper_model if kind == 'whisper' else config.ctc_model
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir(): raise ValueError(f'Model directory does not exist: {path}')
        return path
    repo, revision, provider = MODELS[kind]
    # Adopt the verified local v3 trial checkpoint without another 3 GB download.
    if kind == 'whisper':
        trial = Path(config.cache_dir)/'models/large-v3-edaa852'
        if checkpoint_available(trial, kind):
            return trial
    old = Path.home()/'.cache/py-roller/models/transcriber/providers'/provider/'hub'
    for root in (old, Path(config.cache_dir)/'models'):
        path = root/('models--'+repo.replace('/', '--'))/'snapshots'/revision
        if checkpoint_available(path, kind): return path
    return None


def resolve_model(kind, config):
    local = local_model(kind, config)
    if local is not None:
        if not checkpoint_available(local, kind):
            raise ValueError(f'Incomplete {kind} checkpoint: {local}')
        return local
    repo, revision, _ = MODELS[kind]
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision,
        cache_dir=Path(config.cache_dir)/'models', local_files_only=config.offline))


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
