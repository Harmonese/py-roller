"""Fixed, locally resolved neural backends and content-addressed evidence."""
import gc
from pathlib import Path
import time

import numpy as np

from ..hybrid.io import read_json, save_array

from .catalog import fingerprint, evidence_directory
from .codec import Vocabulary, FrameGeometry, Projection

class CTCStream:
    """4-second cores with 2-second context on a continuous feature grid."""
    def __init__(self, samples, audio_hash, model, config, lyrics=None):
        import torch
        import transformers
        self.torch = torch
        self.config = config
        self.path = Path(model)
        self.samples = samples
        self.device = config.ctc_device
        if self.device == 'auto': self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; use --ctc-device cpu')
        self.model = self.extractor = None
        self.inference_seconds = self.load_seconds = 0.
        self.cache_hits = 0
        data = read_json(self.path/'config.json')
        self.vocabulary = Vocabulary(read_json(self.path/'vocab.json'),data['pad_token_id'])
        self.blank = data['pad_token_id']
        if self.vocabulary.get('<pad>') != self.blank:
            raise ValueError('CTC blank must match vocabulary <pad>')
        self.width = data['vocab_size']
        self.encoder=None
        tokenizer=self.path/'tokenizer_config.json'
        phonetic=tokenizer.exists() and 'Phoneme' in read_json(tokenizer).get('tokenizer_class','')
        if config.language=='mul' and not phonetic:
            raise ValueError('mul requires a Wav2Vec2Phoneme checkpoint with tokenizer_config.json')
        if phonetic:
            from .phonemes import PhonemeEncoder
            self.encoder=PhonemeEncoder(lyrics or '')
            self.vocabulary.encoder=self.encoder
        geometry=FrameGeometry.wav2vec2(data)
        self.stride,self.receptive=geometry.stride,geometry.receptive
        self.step,self.origin=geometry.step,geometry.origin
        self.total_frames = max(0, (len(samples)-self.receptive)//self.stride+1)
        self.hop, self.context = round(4/self.step), round(2/self.step)
        self.metadata = dict(format=3, audio_sha256=audio_hash,
            model_files=fingerprint(self.path, 'ctc'), torch=torch.__version__, transformers=transformers.__version__,
            device=self.device, precision='float16' if self.device == 'mps' else 'float32',
            attention='sdpa', sample_rate=16000, hop_frames=self.hop, context_frames=self.context,
            stride_samples=self.stride, receptive_samples=self.receptive,
            normalization='checkpoint per window', word_delimiter_to_blank=self.vocabulary.get('|'),
            cpu_threads=config.cpu_threads)
        if self.encoder is not None:self.metadata['phoneme_encoding']=self.encoder.metadata
        self.projection=None
        if lyrics is not None and config.language in ('zh','zh-en') and self.encoder is None:
            self.projection=Projection(self.vocabulary,lyrics)
            self.metadata.update(projection=self.projection.metadata)
            self.full_vocabulary=self.vocabulary
            self.vocabulary=self.projection.vocabulary
        self.cache = evidence_directory(Path(config.cache_dir)/'ctc', self.metadata)

    def prepare(self):
        if self.model is not None: return
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        tick = time.perf_counter()
        self.torch.set_num_threads(self.config.cpu_threads)
        self.dtype = self.torch.float16 if self.device == 'mps' else self.torch.float32
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.path, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.path, local_files_only=True,
            dtype=self.dtype, attn_implementation='sdpa', trust_remote_code=False).eval().to(self.device)
        self.load_seconds += time.perf_counter()-tick

    def __iter__(self):
        torch = self.torch
        for start in range(0, self.total_frames, self.hop):
            end = min(start+self.hop, self.total_frames)
            left, right = max(0, start-self.context), min(self.total_frames, end+self.context)
            sample_end = (right-1)*self.stride+self.receptive
            file = self.cache/f'{start:09d}-{end:09d}.npy'
            cached = file.exists()
            if cached:
                logp = np.load(file, allow_pickle=False)
                self.cache_hits += 1
            else:
                self.prepare()
                tick = time.perf_counter()
                inputs = self.extractor(self.samples[left*self.stride:sample_end],
                    sampling_rate=16000, return_tensors='pt')
                with torch.inference_mode():
                    args = dict(input_values=inputs.input_values.to(device=self.device, dtype=self.dtype))
                    if 'attention_mask' in inputs: args['attention_mask'] = inputs.attention_mask.to(self.device)
                    logits = self.model(**args).logits[0][start-left:end-left].float()
                    delimiter = (self.full_vocabulary if self.projection else self.vocabulary).get('|')
                    if delimiter is not None:
                        logits[:, self.blank] = torch.logaddexp(logits[:, self.blank], logits[:, delimiter])
                        logits[:, delimiter] = -1e9
                    logp = logits.log_softmax(-1).cpu().numpy()
                    if self.projection: logp=self.projection.apply(logp)
                self.inference_seconds += time.perf_counter()-tick
                save_array(file, logp)
            width=len(self.projection.columns)+1 if self.projection else self.width
            if logp.shape != (end-start, width) or not np.isfinite(logp).all():
                raise ValueError(f'Invalid CTC cache block; remove and retry: {file}')
            yield dict(start_frame=start, end_frame=end, logp=logp, cached=cached,
                audio_available_until=sample_end/16000)

    def close(self):
        self.model = self.extractor = None
        gc.collect()
        if self.device == 'mps': self.torch.mps.empty_cache()
