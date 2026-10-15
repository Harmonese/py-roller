"""Fixed, locally resolved neural backends and content-addressed evidence."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..hybrid.io import read_json, save_array, save_json, sha256

from .catalog import fingerprint, evidence_directory, WHISPER_OPTIONS

class WhisperStream:
    def __init__(self, samples, audio_hash, model, config, lyrics=None):
        import faster_whisper
        import ctranslate2
        self.samples, self.path, self.config = samples, Path(model), config
        self.model = None
        self.inference_seconds = self.load_seconds = 0.
        self.cache_hits = 0
        self.metadata = dict(format=1, audio_sha256=audio_hash, model_files=fingerprint(self.path, 'whisper'),
            faster_whisper=faster_whisper.__version__, ctranslate2=ctranslate2.__version__,
            device='cpu', compute_type='int8', cpu_threads=config.cpu_threads,
            options=WHISPER_OPTIONS, core_seconds=26, context_seconds=2, sample_rate=16000)
        self.cache = evidence_directory(Path(config.cache_dir)/'whisper', self.metadata)
        self.contextual = None
        if getattr(config, 'whisper_context', 'none') == 'lyrics':
            from ..hybrid.matching import lyric_inventory
            self.contextual = LyricContextPass(samples, model, config, lyric_inventory(lyrics or ''),
                dict(audio_sha256=audio_hash,base_metadata=self.metadata))

    def __iter__(self):
        duration = len(self.samples)/16000
        for index, core_sample in enumerate(range(0, len(self.samples), 26*16000)):
            core = core_sample/16000
            end = min(duration, core+26)
            a, b = max(0, core-2), min(duration, end+2)
            file = self.cache/f'window-{index:03d}.json'
            if file.exists():
                window = read_json(file)
                self.cache_hits += 1
            else:
                if self.model is None:
                    from faster_whisper import WhisperModel
                    tick = time.perf_counter()
                    self.model = WhisperModel(str(self.path), device='cpu', compute_type='int8',
                        cpu_threads=self.config.cpu_threads, num_workers=1, local_files_only=True)
                    self.load_seconds += time.perf_counter()-tick
                tick = time.perf_counter()
                segments, info = self.model.transcribe(self.samples[round(a*16000):round(b*16000)], **WHISPER_OPTIONS)
                segments = [asdict(segment) for segment in segments]
                seconds = time.perf_counter()-tick
                self.inference_seconds += seconds
                window = dict(index=index, start=a, end=b, core_start=core, core_end=end,
                    segments=segments, seconds=seconds, language=info.language)
                save_json(file, window)
            if any(window.get(key) != expected for key, expected in
                   dict(index=index, start=a, end=b, core_start=core, core_end=end).items()):
                raise ValueError(f'Invalid Whisper cache window: {file}')
            if self.contextual is not None:
                if self.model is not None and self.contextual.model is None:
                    self.contextual.model=self.model
                window = self.contextual.process(window)
            yield window

    def close(self):
        self.model = None
        if self.contextual is not None: self.contextual.close()
        gc.collect()


class LyricContextPass:
    """One bounded re-transcription, conditioned only on plain lyric retrieval."""
    def __init__(self,samples,model,config,inventory,source):
        from tokenizers import Tokenizer
        self.samples,self.path,self.config,self.inventory=samples,Path(model),config,inventory
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.model=None
        self.inference_seconds=self.load_seconds=0.
        self.cache_hits=0
        import faster_whisper
        import ctranslate2
        self.metadata=dict(format=1,policy='current_unprompted_window_ordered_coverage_v1',
            token_budget=192,line_budget=6,source=source,inventory=inventory,
            model_files=fingerprint(self.path,'whisper'),options=WHISPER_OPTIONS,
            faster_whisper=faster_whisper.__version__,ctranslate2=ctranslate2.__version__,
            cpu_threads=config.cpu_threads,compute_type='int8')
        self.cache=evidence_directory(Path(config.cache_dir)/'whisper-context',self.metadata)

    def process(self,window):
        from ..hybrid.context import retrieve_context
        digest=hashlib.sha256(json.dumps(window,sort_keys=True).encode()).hexdigest()
        file=self.cache/f"window-{window['index']:03d}-{digest[:16]}.json"
        if file.exists():
            saved=read_json(file)
            if saved['input_sha256']!=digest:raise ValueError('Context cache mismatch')
            self.cache_hits+=1
            return saved['window']
        context=retrieve_context(window,self.inventory,self.tokenizer)
        result=dict(window,lyric_context=context)
        if context['prompt']:
            if self.model is None:
                from faster_whisper import WhisperModel
                tick=time.perf_counter()
                self.model=WhisperModel(str(self.path),device='cpu',compute_type='int8',
                    cpu_threads=self.config.cpu_threads,num_workers=1,local_files_only=True)
                self.load_seconds+=time.perf_counter()-tick
            tick=time.perf_counter()
            a,b=window['start'],window['end']
            segments,info=self.model.transcribe(self.samples[round(a*16000):round(b*16000)],
                **dict(WHISPER_OPTIONS,initial_prompt=context['prompt']))
            result['segments']=[asdict(segment) for segment in segments]
            result['seconds']=time.perf_counter()-tick
            self.inference_seconds+=result['seconds']
            result['language']=info.language
            context['status']='retranscribed'
        else:
            context['status']='no_context_reuse_unprompted'
            result['seconds']=0.
        save_json(file,dict(input_sha256=digest,window=result))
        return result

    def close(self):
        self.model=None
        gc.collect()
