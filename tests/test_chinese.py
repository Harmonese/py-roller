import json
from dataclasses import replace

import numpy as np
import pytest

from pyroller.config import EngineConfig
from pyroller.language import normalize, text_units, resolve_language, pronunciations, whisper_options, independent_support
from pyroller.hybrid.matching import lyric_inventory, window_words
from pyroller.hybrid.text_distance import phrase_comparator, substitution
from pyroller.hybrid.ordered import align_words
from pyroller.hybrid.alignment import encode
from pyroller.backends.codec import Vocabulary, Projection, FrameGeometry
from pyroller.backends.catalog import WHISPER_OPTIONS, model_directory, MODELS
from pyroller.hybrid._alignment import forced_path


def test_simplified_traditional_and_display_offsets():
    text='漁王還想 stay with me！'
    assert normalize(text)=='渔 王 还 想 stay with me'
    assert text_units(text)[0]==('渔',0,1)
    assert text_units(text)[4]==('stay',5,9)
    row=lyric_inventory(text)[0]
    assert row['text']==text
    assert row['line_id']==lyric_inventory('渔王还想 stay with me')[0]['line_id']
    assert lyric_inventory('胸口')[0]['line_id']!=lyric_inventory('凶口')[0]['line_id']
    assert normalize('乾坤')=='乾 坤'
    assert row['source_units'][0]==dict(token='渔',start=0,end=1)


@pytest.mark.parametrize('text,expected',[('Hello world','en'),('你好','zh'),('你好 hello','zh-en')])
def test_language_resolution(text,expected):
    assert resolve_language(text)==expected
    assert EngineConfig(language=expected).language==expected
    assert resolve_language('[Chorus]\n'+text)==expected


def test_scripts_and_options_validation():
    assert resolve_language('こんにちは')=='mul'
    with pytest.raises(ValueError):resolve_language('你好','en')
    with pytest.raises(ValueError):EngineConfig(language='yue')
    assert whisper_options('en',WHISPER_OPTIONS)==WHISPER_OPTIONS
    assert whisper_options('zh',WHISPER_OPTIONS)['language']=='zh'
    mixed=whisper_options('zh-en',WHISPER_OPTIONS)
    assert mixed['multilingual'] and mixed['language'] is None and mixed['task']=='transcribe'


def test_context_readings_and_soft_homophones():
    assert pronunciations(('银','行'))[-1]=='hang'
    assert pronunciations(('行','走'))[0]=='xing'
    compare=phrase_comparator(['银','行'],['银','航'])
    assert compare(2,2,1,1)==(.5,'mandarin_syllable')
    assert substitution('我','爱')[0]==1
    assert substitution('爱','i')[0]==1


def test_shared_time_is_one_source_not_four_anchors():
    window=dict(index=0,start=0,end=10,segments=[dict(words=[dict(word='我喜欢你',start=2,end=4)])])
    words=window_words(window)
    assert len(words)==4
    assert {(w['start'],w['end']) for w in words}=={(2.,4.)}
    assert independent_support([dict(asr_word=i) for i in range(4)],words)==1
    assert words[-1]['source_span']==[3,4]


def test_mixed_global_mapping_and_missing_chinese():
    target=normalize('我爱你 stay with me 明天见').split()
    observed=normalize('我爱你 stay with me 见').split()
    mapping,diag=align_words(target,observed)
    assert (8,6,0.) in mapping
    assert not any(i in (6,7) for i,j,c in mapping)


def test_label_encoder_unknowns_and_blank_alias():
    vocab=Vocabulary({'_':3,'我':0,'爱':1,'A':2},3)
    assert encode('我愛 a',vocab)==([0,1,2],[(0,1),(1,2),(2,3)])
    with pytest.raises(ValueError,match='digits'):encode('我爱你2026',vocab)
    from pyroller.backends.codec import text_diagnostics
    issues=text_diagnostics('我爱2026\n[Chorus]',vocab)
    assert issues==[dict(input_line_index=0,unit='2026',start=2,end=6,
                         reasons=['numeric_reading_not_expanded','outside_ctc_vocabulary'])]


def test_projection_preserves_forced_scores_and_support():
    vocab=Vocabulary({'<pad>':0,'我':1,'爱':2,'你':3,'天':4},0)
    projection=Projection(vocab,'我爱')
    rng=np.random.default_rng(7)
    x=rng.uniform(.05,1,(35,5));x=np.log(x/x.sum(axis=1,keepdims=True)).astype(np.float32)
    compact=projection.apply(x)
    full=forced_path(x,encode('我爱',vocab)[0],0,6.)
    small=forced_path(compact,encode('我爱',projection.vocabulary)[0],0,6.)
    assert full==small
    assert np.allclose(x.max(axis=1),compact.max(axis=1))
    other=Projection(vocab,'我你')
    assert projection.metadata!=other.metadata


def test_frame_geometry_and_model_routing(tmp_path):
    geo=FrameGeometry.wav2vec2(dict(conv_kernel=[10,3,3,3,3,2,2],conv_stride=[5,2,2,2,2,2,2]))
    assert geo.step==.02 and geo.origin==.0125
    config=EngineConfig(cache_dir=tmp_path,language='zh')
    assert MODELS['ctc-zh'][1] in str(model_directory('ctc',config))
    assert model_directory('ctc',config)!=model_directory('ctc',replace(config,language='en'))


def test_protocol_and_cli_language(tmp_path,monkeypatch):
    from pyroller.protocol import load_request
    from pyroller import cli
    file=tmp_path/'request.json'
    file.write_text(json.dumps(dict(protocol_version=2,request_id='zh',audio='a',lyrics='b',output='c',options=dict(language='zh-en'))))
    assert load_request(file).config.language=='zh-en'
    seen=[]
    monkeypatch.setattr(cli,'align',lambda *args,**kwargs:seen.append(args[3]))
    assert cli.main(['align','a','b','-o','c','--language','zh'])==0
    assert seen[0].language=='zh'


def test_repeated_chinese_labels_require_blank():
    vocab=Vocabulary({'<pad>':0,'人':1},0)
    x=np.log(np.array([[.01,.99],[.99,.01],[.01,.99]],dtype=np.float32))
    path=forced_path(x,encode('人人',vocab)[0],0,6.)
    assert path['starts']==[0,2]


@pytest.mark.parametrize('mode',['ordered','unordered'])
def test_chinese_mixed_engine_paths(mode):
    from pyroller.hybrid.core import align_evidence
    from pyroller.hybrid.config import AlignmentSettings
    vocab=Vocabulary({'<pad>':0,'我':1,'爱':2,'你':3,'H':4,'I':5},0)
    x=np.full((80,6),.001,dtype=np.float32);x[:,0]=.995
    for frame,label in [(10,1),(15,2),(20,3),(30,4),(35,5)]:
        x[frame:frame+3]=.001;x[frame:frame+3,label]=.995
    window=dict(index=0,start=0,end=8,core_start=0,core_end=8,segments=[dict(words=[
        dict(word='我',start=1,end=1.3),dict(word='爱',start=1.5,end=1.8),
        dict(word='你',start=2,end=2.3),dict(word='hi',start=3,end=3.8)])])
    result=align_evidence('我愛你 hi',[window],np.log(x),vocab,0,.1,8,
                          settings=AlignmentSettings(alignment_mode=mode))
    assert len(result['occurrences'])==1
    assert result['occurrences'][0]['text']=='我愛你 hi'
    assert result['occurrences'][0]['start']==pytest.approx(1)


def test_blank_id_not_named_pad_in_checkpoint():
    vocabulary=Vocabulary({'_':2,'人':0,'口':1},2)
    projected=Projection(vocabulary,'人口')
    x=np.log(np.array([[.98,.01,.01],[.01,.01,.98],[.01,.98,.01]],dtype=np.float32))
    path=forced_path(projected.apply(x),encode('人口',projected.vocabulary)[0],0,6.)
    assert path['starts']==[0,2]


def test_unencodable_row_does_not_disable_neighbour_timing():
    from pyroller.hybrid.alignment import refine
    from pyroller.hybrid.config import AlignmentSettings
    rows=[dict(text=t,start=a,end=a+.4,token_start=i,token_end=i+1,
        matched_words=[dict(lyric_word=0,asr_word=i,cost=0)]) for i,(t,a) in enumerate([('人',1.),('汐',2.),('口',3.)])]
    words=[dict(start=r['start'],end=r['end']) for r in rows]
    vocab=Vocabulary({'<pad>':0,'人':1,'口':2},0)
    x=np.full((60,3),.005,dtype=np.float32);x[:,0]=.99
    for a,label in [(11,1),(31,2)]:x[a:a+3]=.005;x[a:a+3,label]=.99
    result,reports=refine(rows,words,np.log(x),vocab,0,.1,6,AlignmentSettings())
    assert result[0]['timing_method']=='local_ordered_ctc'
    assert result[1]['timing_status']=='coarse_unencodable'
    assert result[2]['timing_method']=='local_ordered_ctc'


def test_exact_whole_phrase_retains_one_source_coarse_time():
    from pyroller.hybrid.core import align_evidence
    from pyroller.hybrid.config import AlignmentSettings
    vocabulary=Vocabulary({'<pad>':0,'我':1,'爱':2,'你':3},0)
    x=np.full((600,4),.001,dtype=np.float32);x[:,0]=.997
    for frame,label in [(20,1),(23,2),(26,3)]:x[frame:frame+2]=.001;x[frame:frame+2,label]=.997
    window=dict(index=0,start=0,end=60,core_start=0,core_end=60,segments=[dict(words=[dict(word='我爱你',start=2,end=3)])])
    result=align_evidence('我愛你',[window],np.log(x),vocabulary,0,.1,60,
        settings=AlignmentSettings(alignment_mode='ordered'))
    assert len(result['coarse_occurrences'])==1
    assert result['coarse_occurrences'][0]['start']==2
    assert independent_support(result['coarse_occurrences'][0]['matched_words'],result['stitched_words'])==1


def test_chinese_preview_keeps_future_line_pending():
    from pyroller.hybrid.core import align_evidence
    from pyroller.hybrid.config import AlignmentSettings
    vocabulary=Vocabulary({'<pad>':0,'你':1,'好':2,'再':3,'见':4},0)
    x=np.full((50,5),.001,dtype=np.float32);x[:,0]=.996
    for frame,label in [(10,1),(13,2),(30,3),(33,4)]:x[frame:frame+2]=.001;x[frame:frame+2,label]=.996
    window=dict(index=0,start=0,end=5,core_start=0,core_end=5,segments=[dict(words=[dict(word='你好',start=1,end=1.5)])])
    result=align_evidence('你好\n再见',[window],np.log(x),vocabulary,0,.1,5,final=False,
        settings=AlignmentSettings(alignment_mode='ordered'))
    assert [r['text'] for r in result['occurrences']]==['你好']
    assert result['unaligned_lines'][0]['reason']=='pending_audio'


def test_whisper_language_cache_isolation_without_loading_weights(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from pyroller.backends import whisper
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(__version__='fixed'))
    monkeypatch.setitem(sys.modules,'ctranslate2',SimpleNamespace(__version__='fixed'))
    monkeypatch.setattr(whisper,'fingerprint',lambda *args:{'model.bin':'fixed-hash'})
    def stream(language):
        return whisper.WhisperStream(np.zeros(16000,dtype=np.float32),'audio',tmp_path,
            EngineConfig(language=language,cache_dir=tmp_path))
    en,auto,zh,mixed=(stream(language) for language in ('en','auto','zh','zh-en'))
    assert en.cache==auto.cache
    assert len({en.cache,zh.cache,mixed.cache})==3
    assert all(s.model is None for s in (en,auto,zh,mixed))


def test_latin_subwords_in_chinese_whisper_are_reassembled():
    window=dict(index=0,start=0,end=10,language='zh',segments=[dict(words=[
        dict(word='我',start=1,end=1.2),dict(word=' lo',start=1.2,end=1.5),
        dict(word='ve',start=1.5,end=1.7),dict(word=' you',start=1.7,end=2),dict(word='你',start=2,end=2.2)])])
    words=window_words(window)
    assert [w['token'] for w in words]==['我','love','you','你']
    assert words[1]['start']==1.2 and words[1]['end']==1.7
    assert words[1]['source_id']==[0,0,1,2]
    assert words[1]['probability'] is None
