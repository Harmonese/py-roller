"""Import historical raw evidence without importing the historical engine."""
from pathlib import Path
import numpy as np

from .core import align_evidence
from .io import atomic_write, lrc_text, read_json, save_json, sha256


def load_evidence(whisper, lyrics, cache, model):
    whisper, lyrics, cache, model = map(Path, (whisper, lyrics, cache, model))
    source, manifest = read_json(whisper/'result.json'), read_json(cache/'manifest.json')
    if source.get('reference_used') is not False or source.get('lyrics_used') is not False:
        raise ValueError('Replay requires unprompted, reference-free ASR')
    if manifest['audio_sha256'] != source['audio_sha256'] or sha256(source['audio']) != source['audio_sha256']:
        raise ValueError('Whisper and CTC audio identity mismatch')
    for filename, digest in manifest['model_files'].items():
        if sha256(model/filename) != digest: raise ValueError(f'CTC model mismatch: {filename}')
    config = read_json(model/'config.json')
    stride = receptive = 1
    for kernel, step in zip(config['conv_kernel'], config['conv_stride']):
        receptive += (kernel-1)*stride
        stride *= step
    rate = manifest['sample_rate']
    blocks, frame = [], 0
    for path in sorted(cache.glob('*.npy')):
        begin, end = map(int, path.stem.split('-'))
        data = np.load(path, allow_pickle=False)
        if begin != frame or data.shape != (end-begin, config['vocab_size']):
            raise ValueError(f'Noncontiguous or invalid CTC block: {path}')
        blocks.append(data)
        frame = end
    if frame != (round(source['duration']*rate)-receptive)//stride+1:
        raise ValueError('Incomplete CTC evidence')
    windows = [read_json(p) for p in sorted(whisper.glob('window-*.json'))]
    if [w['index'] for w in windows] != list(range(len(source['windows']))):
        raise ValueError('Incomplete Whisper evidence')
    inputs = (lyrics.read_text(encoding='utf-8-sig'), windows, np.concatenate(blocks),
        read_json(model/'vocab.json'), receptive/(2*rate), stride/rate, source['duration'])
    provenance = dict(whisper=str(whisper), ctc_cache=str(cache), model=str(model),
        lyrics=str(lyrics), lyrics_sha256=sha256(lyrics), audio=source['audio'],
        audio_sha256=source['audio_sha256'], duration=source['duration'], replay=True,
        reference_used=False, reference_order_used=False)
    return inputs,provenance


def write_replay(result,provenance,output):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'result.json', result)
    atomic_write(output/'result.lrc', lrc_text(result['occurrences']))
    atomic_write(output/'coarse.lrc', lrc_text(result['coarse_occurrences']))
    save_json(output/'run.json', provenance)


def replay(whisper, lyrics, cache, model, output, settings=None):
    inputs,provenance=load_evidence(whisper,lyrics,cache,model)
    result=align_evidence(*inputs,settings=settings)
    write_replay(result,provenance,output)
    return result
