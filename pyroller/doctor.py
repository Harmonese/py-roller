"""Read-only installation checks; never downloads or loads model weights."""
import importlib
import shutil
import sys

from . import __version__
from .backends.catalog import MODELS, local_model, checkpoint_available, model_directory, legacy_model
from .config import EngineConfig


def model_inventory(config):
    models = []
    for kind, (repo, revision, _) in MODELS.items():
        path = local_model(kind, config)
        complete = bool(path and checkpoint_available(path, kind))
        models.append(dict(kind=kind, repo=repo, revision=revision,
            path=str(path) if path else None, available=complete,
            managed_path=str(model_directory(kind, config)),
            legacy_path=str(legacy) if (legacy := legacy_model(kind, config)) else None))
    return models


def inspect(config=None):
    config = config or EngineConfig()
    checks = [dict(name='python', ok=(3, 10) <= sys.version_info[:2] < (3, 13), detail=sys.version.split()[0]),
              dict(name='ffmpeg', ok=shutil.which('ffmpeg') is not None, detail=shutil.which('ffmpeg'))]
    modules = [('numpy', 'numpy'), ('torch', 'torch'), ('transformers', 'transformers'),
               ('faster-whisper', 'faster_whisper'), ('cmudict', 'cmudict'),
               ('native_ctc', 'pyroller.hybrid._alignment')]
    if config.audio_mode == 'vocals': modules += [('demucs', 'demucs'), ('torchaudio', 'torchaudio')]
    for name, module in modules:
        try:
            loaded = importlib.import_module(module)
            checks.append(dict(name=name, ok=True, detail=getattr(loaded, '__version__', 'importable')))
        except Exception as exc:
            checks.append(dict(name=name, ok=False, detail=str(exc)))
    models = model_inventory(config)
    runtime_ok = all(check['ok'] for check in checks)
    try:
        import ssl
        network = dict(https_available=True, detail=ssl.OPENSSL_VERSION)
    except ImportError as exc:
        network = dict(https_available=False, detail=str(exc))
    return dict(protocol_version=2, engine='py-roller', engine_version=__version__,
        type='doctor', ready=runtime_ok and all(m['available'] for m in models
            if m['kind'] != 'demucs' or config.audio_mode == 'vocals'),
        runtime_ok=runtime_ok, checks=checks, models=models, network=network,
        capabilities=dict(languages=['en'], formats=['lrc'], audio_modes=['mix', 'vocals'],
                          revisable_previews=True, protocol_versions=[2]))
