"""File-job orchestration, independent of matching and neural implementations."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .config import EngineConfig
from .hybrid.config import AlignmentSettings
from .hybrid.core import align_evidence
from .hybrid.io import atomic_write, decode_audio, lrc_text, save_json, sha256
from .hybrid.matching import lyric_inventory
from .backends.catalog import resolve_model
from .backends.ctc import CTCStream
from .backends.whisper import WhisperStream
from .backends.separation import separate
from .protocol import EventStream, alignment_document, error_code
from .language import resolve_language, TEXT_POLICY
from .backends.codec import text_diagnostics


def visible_frames(blocks, available_until):
    """Do not expose a block before its right-context audio is available."""
    end = 0
    for block in blocks:
        if block['audio_available_until'] > available_until+1e-6: break
        end = block['end_frame']
    return end


def align(audio, lyrics, output, config=None, on_event=None, *, request_id=None):
    """Create a new output directory and return protocol-v2 alignment.json data.

    Events use protocol v2. Internal diagnostics live separately in result.json.
    Final means computation ended, not that a human verified the lyrics.
    """
    stream = EventStream(request_id, on_event)
    ctc = whisper = None
    run = None
    started = time.perf_counter()
    try:
        stream.emit(dict(type='started'))
        config = config or EngineConfig()
        audio, lyrics, output = (Path(p).expanduser().resolve() for p in (audio, lyrics, output))
        text = lyrics.read_text(encoding='utf-8-sig')
        lyric_inventory(text)
        requested_language=config.language
        config=replace(config,language=resolve_language(text,config.language))
        if not audio.is_file(): raise FileNotFoundError(f'Audio does not exist: {audio}')
        output.mkdir(parents=True, exist_ok=False)
        stream.attach(output/'events.jsonl')
        run = dict(schema_version=2, protocol_version=2, request_id=stream.request_id,
            status='running', audio=str(audio), lyrics=str(lyrics),
            config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            reference_used=False, reference_order_used=config.alignment_mode == 'ordered')
        save_json(output/'run.json', run)
        run.update(audio_sha256=sha256(audio), lyrics_sha256=sha256(lyrics))
        run.update(requested_language=requested_language,language=config.language,text_policy=TEXT_POLICY)
        tick = time.perf_counter()
        selected_audio = separate(audio, config, stream.emit) if config.audio_mode == 'vocals' else audio
        separation_seconds = time.perf_counter()-tick
        stream.emit(dict(type='progress', stage='decode'))
        samples = decode_audio(selected_audio)
        duration = len(samples)/16000
        audio_hash = sha256(selected_audio)
        stream.emit(dict(type='progress', stage='models'))
        ctc_path, whisper_path = resolve_model('ctc', config), resolve_model('whisper', config)
        ctc = CTCStream(samples, audio_hash, ctc_path, config, text)
        blocks, arrays = [], []
        for block in ctc:
            arrays.append(block['logp'])
            blocks.append({k: v for k, v in block.items() if k != 'logp'})
            stream.emit(dict(type='progress', stage='ctc', audio_available_until=block['audio_available_until'],
                duration=duration, cached=block['cached']))
        if not arrays: raise ValueError('Audio too short for CTC model')
        logp = np.concatenate(arrays)
        del arrays
        ctc.close()
        whisper = WhisperStream(samples, audio_hash, whisper_path, config)
        windows, revision, processing_seconds = [], 0, 0.
        for window in whisper:
            windows.append(window)
            stream.emit(dict(type='progress', stage='whisper', audio_available_until=window['end'], duration=duration))
            final = window['core_end'] >= duration
            if not config.previews and not final: continue
            count = len(logp) if final else visible_frames(blocks, window['end'])
            if count == 0: continue
            result = align_evidence(text, windows, logp[:count], ctc.vocabulary,
                ctc.origin, ctc.step, duration if final else window['end'], final=final,
                settings=AlignmentSettings(alignment_mode=config.alignment_mode))
            processing_seconds += result['metrics']['processing_seconds']
            timing_backend='wav2vec2_phoneme_ctc' if getattr(ctc,'encoder',None) is not None else 'wav2vec2_ctc'
            result.update(language=config.language,timing_backend=timing_backend)
            if config.language!='en':result['text_diagnostics']=text_diagnostics(text,ctc.vocabulary)
            revision += 1
            event = dict(type='snapshot', revision=revision,
                audio_available_until=window['end'], final=final, occurrences=result['occurrences'],
                language=config.language,timing_backend=timing_backend)
            if config.alignment_mode == 'ordered':
                event.update(alignment_mode='ordered', unaligned_lines=result['unaligned_lines'])
            if not final:
                atomic_write(output/'preview.lrc', lrc_text(result['occurrences']))
                stream.emit(event)
        if not windows or not final: raise RuntimeError('Whisper did not complete the audio')
        whisper.close()
        result['metrics'].update(processing_all_revisions_seconds=processing_seconds,
            ctc_inference_seconds=ctc.inference_seconds, whisper_inference_seconds=whisper.inference_seconds,
            ctc_load_seconds=ctc.load_seconds, whisper_load_seconds=whisper.load_seconds,
            separation_seconds=separation_seconds, total_seconds=time.perf_counter()-started,
            ctc_cache_hits=ctc.cache_hits, whisper_cache_hits=whisper.cache_hits)
        document = alignment_document(result, stream.request_id, duration)
        save_json(output/'result.json', result)
        save_json(output/'alignment.json', document)
        atomic_write(output/'result.lrc', lrc_text(result['occurrences']))
        atomic_write(output/'coarse.lrc', lrc_text(result['coarse_occurrences']))
        run.update(status='complete', selected_audio=str(selected_audio), selected_audio_sha256=audio_hash,
            duration=duration, ctc_model=str(ctc_path), whisper_model=str(whisper_path),
            ctc_cache=str(ctc.cache), whisper_cache=str(whisper.cache),
            frame_origin=ctc.origin, frame_step=ctc.step, metrics=result['metrics'])
        save_json(output/'run.json', run)
        stream.emit(event)
        stream.emit(dict(type='completed', revision=revision,
            artifacts={key: str(output/name) for key, name in dict(
                lrc='result.lrc', alignment='alignment.json', diagnostics='result.json',
                run='run.json', events='events.jsonl').items()},
            summary=dict(lines=len(document['lines']), duration=duration), metrics=result['metrics']))
        return document
    except BaseException as exc:
        cancelled = isinstance(exc, KeyboardInterrupt)
        if run is not None:
            run.update(status='cancelled' if cancelled else 'failed', error=str(exc))
            try: save_json(output/'run.json', run)
            except OSError: pass
        if not stream.terminal:
            stream.emit(dict(type='cancelled' if cancelled else 'failed',
                code='cancelled' if cancelled else error_code(exc),
                message=str(exc) or ('Cancelled' if cancelled else type(exc).__name__)))
        raise
    finally:
        if ctc is not None: ctc.close()
        if whisper is not None: whisper.close()
        stream.close()
