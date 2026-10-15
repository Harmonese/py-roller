from pathlib import Path
from contextlib import contextmanager
import subprocess
import sys

from .catalog import evidence_directory
from ..hybrid.io import sha256, read_json, save_json, decode_audio


@contextmanager
def cache_lock(folder):
    lock = Path(folder)/'working.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f'Separation cache is busy: {lock}. Retry after its worker exits.') from exc
    try:
        yield
    finally:
        lock.rmdir()


def separate(audio, config, progress):
    from importlib.metadata import version, PackageNotFoundError
    try: demucs_version = version('demucs')
    except PackageNotFoundError as exc:
        raise RuntimeError('Vocals mode requires pip install "py-roller[vocals]"') from exc
    metadata = dict(audio_sha256=sha256(audio), demucs=demucs_version, model='htdemucs',
        torch=version('torch'), torchaudio=version('torchaudio'),
        shifts=1, seed=0, cpu_threads=config.cpu_threads, device='cpu', float32=True,
        input_decode='ffmpeg stereo 44100 float32', clip_mode='none')
    folder = evidence_directory(Path(config.cache_dir)/'separation', metadata)
    target = folder/'htdemucs/input/vocals.wav'
    completed = folder/'complete.json'
    if target.exists() and completed.exists():
        if sha256(target) != read_json(completed)['audio_sha256']:
            raise ValueError(f'Corrupt separation cache: {target}')
        return target
    if config.offline:
        raise ValueError('Offline vocals mode requires a separation cache; use mix or pre-separated audio')
    with cache_lock(folder):
        if target.exists() and completed.exists():
            if sha256(target) != read_json(completed)['audio_sha256']:
                raise ValueError(f'Corrupt separation cache: {target}')
            return target
        progress(dict(type='progress', stage='separation', audio=str(audio)))
        _run_demucs(audio, folder, target, config)
        save_json(completed, dict(audio_sha256=sha256(target)))
    return target


def _run_demucs(audio, folder, target, config):
    decoded = folder/'input.wav'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(audio),'-ar','44100','-ac','2',
        '-c:a','pcm_f32le',str(decoded)], check=True, capture_output=True)
    with (folder/'demucs.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pyroller.backends.demucs_worker', str(config.cpu_threads),
            '--two-stems', 'vocals', '-n', 'htdemucs', '--shifts', '1', '-d', 'cpu',
            '--float32', '--clip-mode', 'none', '-o', str(folder), str(decoded)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    if not target.is_file(): raise RuntimeError(f'Demucs produced no vocals: {folder}/demucs.log')
    decode_audio(target)
