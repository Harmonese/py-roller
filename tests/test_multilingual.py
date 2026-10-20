import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyroller.language import normalize,text_units,resolve_language,whisper_options
from pyroller.config import EngineConfig
from pyroller.backends.catalog import model_directory,MODELS,WHISPER_OPTIONS
from pyroller.backends.codec import Vocabulary,encode_units,text_diagnostics
from pyroller.backends.phonemes import PhonemeEncoder
from pyroller.hybrid.matching import lyric_inventory,window_words
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.config import AlignmentSettings


def test_unicode_identity_and_offsets():
    text='Песня без слов — déjà vu 我 love 你'
    assert normalize(text)=='песня без слов déjà vu 我 love 你'
    assert text_units(text)[0]==('песня',0,5)
    assert lyric_inventory('Слово')[0]['text']=='Слово'
    assert lyric_inventory('всё')[0]['line_id']!=lyric_inventory('все')[0]['line_id']
    assert normalize('我love你')=='我 love 你'
    assert normalize('かなカナ ภาษา')=='か な カ ナ ภ า ษ า'


def test_profile_routing_and_protocol(tmp_path):
    assert resolve_language('Песня без слов')=='mul'
    assert resolve_language('你好','mul')=='mul'
    with pytest.raises(ValueError):resolve_language('Песня','zh')
    config=EngineConfig(cache_dir=tmp_path,language='mul')
    assert MODELS['ctc-mul'][1] in str(model_directory('ctc',config))
    assert model_directory('ctc',config)!=model_directory('ctc',replace(config,language='en'))
    options=whisper_options('mul',WHISPER_OPTIONS)
    assert options['language'] is None and options['multilingual']
    assert options['initial_prompt'] is None and not options['condition_on_previous_text']
    from pyroller.protocol import load_request
    from pyroller.cli import parser
    file=tmp_path/'r.json'
    file.write_text(json.dumps(dict(protocol_version=2,request_id='mul',audio='a',lyrics='b',output='c',options=dict(language='mul'))))
    assert load_request(file).config.language=='mul'
    assert parser().parse_args(['align','a','b','-o','c','--language','mul']).language=='mul'


def fake_encoder(units):
    e=object.__new__(PhonemeEncoder);e.units=units;e.errors={};return e


def test_multicodepoint_phone_is_one_label_and_oov_not_dropped():
    v=Vocabulary({'<pad>':0,'tʃʲ':1,'aɪ':2,'n':3},0)
    v.encoder=fake_encoder({'ночь':('n','tʃʲ'),'i':('aɪ',),'bad':('missing',)})
    assert encode_units('ночь i',v)==([3,1,2],[(0,2),(2,3)])
    with pytest.raises(ValueError,match='missing'):encode_units('bad',v)
    assert text_diagnostics('bad',v)[0]['reasons']==['outside_ctc_vocabulary']


def test_russian_bpe_shared_source_is_not_independent_times():
    words=window_words(dict(index=0,start=0,end=10,language='ru',segments=[dict(words=[
        dict(word=' Пес',start=1,end=1.5),dict(word='ня',start=1.5,end=2),
        dict(word=' без слов',start=2,end=3)])]))
    assert [w['token'] for w in words]==['песня','без','слов']
    assert words[0]['source_id']==[0,0,0,1]
    assert words[1]['source_id']==words[2]['source_id']
    assert words[1]['start']==words[2]['start']==2


@pytest.mark.parametrize('mode',['ordered','unordered'])
def test_russian_full_core_and_preview(mode):
    v=Vocabulary({'<pad>':0,'pʲ':1,'e':2,'s':3},0)
    v.encoder=fake_encoder({'песня':('pʲ','e'),'слов':('s',)})
    x=np.full((100,4),.001,dtype=np.float32);x[:,0]=.997
    for frame,label in [(20,1),(25,2),(40,3)]:x[frame]=.001;x[frame,label]=.997
    window=dict(index=0,start=0,end=2,core_start=0,core_end=2,language='ru',segments=[dict(words=[
        dict(word=' Песня',start=.4,end=.65),dict(word=' слов',start=.8,end=1.)])])
    settings=AlignmentSettings(alignment_mode=mode)
    for final in (False,True):
        r=align_evidence('Песня слов',[window],np.log(x),v,0,.02,2,settings,final)
        assert len(r['occurrences'])==1
        assert r['occurrences'][0]['text']=='Песня слов'
        assert r['occurrences'][0]['character_path']['tokens']==[1,2,3]


def test_voice_plan_and_encoding_identity(monkeypatch):
    from pyroller.backends import phonemes
    class Backend:
        @staticmethod
        def supported_languages():return {'en-us':'English','cmn':'Mandarin','ru':'Russian'}
        @staticmethod
        def version():return (1,52,0)
    class Identifier:
        def classify(self,text):return ('ru' if 'песня' in text else 'en',1.)
    monkeypatch.setattr(phonemes,'dependencies',lambda:(Backend,None,Identifier()))
    monkeypatch.setattr(phonemes,'version',lambda name:'test')
    seen=[]
    def phones(voice,units):seen.append((voice,units));return tuple((u,) for u in units)
    monkeypatch.setattr(phonemes,'phonemize_units',phones)
    encoder=PhonemeEncoder('我 love 你 песня')
    assert encoder.voices==dict(han='cmn',latin='en-us',cyrillic='ru')
    assert all('phoneme' not in k for k in encoder.units)
    assert encoder.metadata['units']['песня']==['песня']
    frozen=PhonemeEncoder.from_metadata(json.loads(json.dumps(encoder.metadata)))
    v=Vocabulary({'<pad>':0,'песня':1},0)
    assert frozen.encode('песня',v)==encoder.encode('песня',v)
    with pytest.raises(ValueError):PhonemeEncoder.from_metadata(dict(policy='future'))


def test_phonetic_model_cache_includes_pronunciation_resources(tmp_path,monkeypatch):
    from pyroller.backends.ctc import CTCStream
    from pyroller.backends import ctc
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps(dict(pad_token_id=0,vocab_size=2,conv_kernel=[2],conv_stride=[2])))
    (model/'vocab.json').write_text(json.dumps({'<pad>':0,'n':1}))
    (model/'tokenizer_config.json').write_text(json.dumps(dict(tokenizer_class='Wav2Vec2PhonemeCTCTokenizer')))
    monkeypatch.setattr(ctc,'fingerprint',lambda *a:{'weights':'test'})
    import sys
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(__version__='test'))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(__version__='test'))
    from pyroller.backends import phonemes
    def enc(text):
        e=fake_encoder({'ночь':('n',)});e.metadata=dict(policy=text);return e
    monkeypatch.setattr(phonemes,'PhonemeEncoder',enc)
    config=EngineConfig(language='mul',cache_dir=tmp_path/'cache',ctc_device='cpu')
    a=CTCStream(np.zeros(20),'audio',model,config,'ночь')
    b=CTCStream(np.zeros(20),'audio',model,config,'ночь ночь')
    assert a.cache!=b.cache and a.vocabulary.encoder is a.encoder
