"""Three fixed songs: ordered, original mix; reference times are evaluation-only."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pyroller.hybrid.io import read_json,save_json,atomic_write,sha256
from pyroller.language import normalize
from evaluation import read_lrc

OUT=ROOT/'py-roller-artifacts/mul-validation'
LIB=Path('/Volumes/StudioVantablack/Main/14 Music Library/Albums')
SONGS={
    'M01':('The Veldt',LIB/'deadmau5 - _ album title goes here _ (Deluxe)/03 deadmau5 feat. Chris James - The Veldt (8 minute edit).mp3'),
    'M02':('從流水線上來，到流水線上去',LIB/'簡諧點源 - 血紅都會/07 簡諧點源 - 從流水線上來，到流水線上去.mp3'),
    'M03':('Песня без слов',LIB/'Кино - Кино в Севкабеле/01 Песня без слов.mp3'),
}


def prepare():
    songs=[]
    for sid,(name,audio) in SONGS.items():
        refs=read_lrc(audio.with_suffix('.lrc'))
        save_json(OUT/sid/'reference.json',refs)
        atomic_write(OUT/sid/'ordered.txt','\n'.join(r['text'] for r in refs)+'\n')
        songs.append(dict(id=sid,name=name,audio=str(audio),audio_sha256=sha256(audio),
            reference=str(audio.with_suffix('.lrc')),reference_sha256=sha256(audio.with_suffix('.lrc')),lines=len(refs)))
    save_json(OUT/'experiment.json',dict(songs=songs,language='mul',alignment_mode='ordered',audio_mode='mix',
        reference_times_used_in_inference=False,parameter_scan=False,previews=False))


def evaluate(sid):
    folder=OUT/sid/'mul';result=read_json(folder/'result.json');run=read_json(folder/'run.json')
    refs=read_json(OUT/sid/'reference.json');located={r['input_line_index']:r for r in result['occurrences']}
    pairs=[];missing=[];spans=[];offset=0
    for ref in refs:
        end=offset+len(normalize(ref['text']).split());spans.append((offset,end));offset=end
    for index,ref in enumerate(refs):
        if index in located:
            row=located[index]
            assert normalize(row['text'])==normalize(ref['text'])
            pairs.append(dict(index=index,reference=ref,output=row,error=row['start']-ref['start']))
        else:
            failure=next(r for r in result['unaligned_lines'] if r['input_line_index']==index)
            recovery=next((r for r in result.get('recovery_regions',[]) if r['input_line_index']==index),{})
            issue=[d for d in result.get('text_diagnostics',[]) if d['input_line_index']==index]
            a,b=spans[index]
            mapping=[(li,wi,c) for li,wi,c in result['word_mapping'] if a<=li<b]
            clues=[dict(lyric_unit=li-a,asr=result['stitched_words'][wi]['token'],
                start=result['stitched_words'][wi]['start'],cost=c) for li,wi,c in mapping]
            category=('unencodable' if issue else 'timing_budget' if failure['reason']=='region_budget_or_empty' else
                      'timing_conflict' if failure['reason']=='overlapping_assignment' else 'weak_local_ctc_support')
            missing.append(dict(index=index,reference=ref,reason=failure['reason'],category=category,
                matched_unit_coverage=len({li for li,wi,c in mapping})/max(1,b-a),asr_clues=clues,
                recovery=recovery,encoding_issues=issue))
    windows=[read_json(p) for p in sorted(Path(run['whisper_cache']).glob('window-*.json'))]
    atomic_write(folder/'raw-transcript.txt','\n\n'.join(f"[{w['start']:.2f}-{w['end']:.2f}] "+
        ' '.join(s['text'] for s in w['segments']) for w in windows)+'\n')
    resource=(OUT/sid/'mul-resources.log').read_text()
    rss=re.search(r'(\d+)\s+maximum resident set size',resource)
    manifest=read_json(Path(run['ctc_cache'])/'manifest.json')
    changes=[]
    for p in pairs:
        coarse=next((r for r in result['coarse_occurrences'] if r['input_line_index']==p['index']),None)
        if coarse:
            before=coarse['start']-p['reference']['start'];after=p['error']
            if (abs(before)<=.5)!=(abs(after)<=.5):changes.append(dict(index=p['index'],text=p['reference']['text'],
                reference_start=p['reference']['start'],coarse_error=before,final_error=after,
                change='gain' if abs(after)<=.5 else 'loss'))
    row=dict(song=sid,name=SONGS[sid][0],reference_lines=len(refs),located=len(pairs),missing=len(missing),
        over_half_second=sum(abs(p['error'])>.5 for p in pairs),over_two_seconds=sum(abs(p['error'])>2 for p in pairs),
        within_half_second=sum(abs(p['error'])<=.5 for p in pairs),pairs=pairs,missing_details=missing,
        encoding_issues=result.get('text_diagnostics',[]),voices=manifest['phoneme_encoding']['voices'],
        metrics=result['metrics'],timing_changes=changes,rss_bytes=int(rss.group(1)) if rss else None,
        evidence_bytes=sum(p.stat().st_size for key in ('ctc_cache','whisper_cache') for p in Path(run[key]).rglob('*') if p.is_file()))
    save_json(folder/'evaluation.json',row)
    return row


def summarize():
    rows=[evaluate(sid) for sid in SONGS if (OUT/sid/'mul/result.json').exists()]
    save_json(OUT/'summary.json',rows)
    md=['# mul 三首固定验证','',
        '全部 ordered、原声、Whisper large-v3 + XLSR-53-espeak；未使用参考时间推理，不逐曲调参。',
        '同名 LRC 仅作参考，未经逐句人工核听；按输入行身份配对，已定位不等于已确认识别正确。','',
        '|歌曲|参考行|未定位|已定位 >0.5s|其中 >2s|≤0.5s|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:md.append('|'+ '|'.join(str(r[k]) for k in ('name','reference_lines','missing','over_half_second','over_two_seconds','within_half_second'))+'|')
    for r in rows:
        sid=r['song'];md+=['',f"## {r['name']}",'',f"[LRC]({sid}/mul/result.lrc) · [逐句记录]({sid}/mul/evaluation.json) · [Whisper 原文]({sid}/mul/raw-transcript.txt)",'',
            f"G2P 语音配置：{r['voices']}；编码异常 {len(r['encoding_issues'])} 个单位。",'',
            '### 未定位','']
        for m in r['missing_details']:
            md.append(f"- {m['reference']['start']:.2f}s `{m['reference']['text']}`：{m['reason']}；ASR 对应覆盖 {m['matched_unit_coverage']:.0%}；{m['recovery'].get('error','')}。")
        if not r['missing_details']:md.append('无。')
        md+=['','### 已定位但起点误差 >0.5 秒','']
        for p in r['pairs']:
            if abs(p['error'])>.5:md.append(f"- {p['reference']['start']:.2f}s `{p['reference']['text']}` → {p['output']['start']:.2f}s（{p['error']:+.2f}s，{p['output']['timing_method']}）。")
        md+=['','### 音素 CTC 相对同一实例的 Whisper 粗时间','']
        if not r['timing_changes']:md.append('没有跨越 0.5 秒达标边界的变化。')
        for c in r['timing_changes']:md.append(f"- {c['change']} `{c['text']}`：{c['coarse_error']:+.2f}s → {c['final_error']:+.2f}s。")
        md+=['','### 开销','',f"主进程峰值 RSS {r['rss_bytes']} bytes；证据缓存 {r['evidence_bytes']} bytes。",'',
            '```json',json.dumps(r['metrics'],ensure_ascii=False,indent=2),'```']
    atomic_write(OUT/'summary.md','\n'.join(md)+'\n')
    for r in rows:print({k:r[k] for k in ('song','reference_lines','missing','over_half_second','over_two_seconds','within_half_second')},flush=True)


def run():
    save_json(OUT/'source-hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller').rglob('*.py')})
    for sid,(_,audio) in SONGS.items():
        folder=OUT/sid/'mul'
        if (folder/'result.json').exists():continue
        with (OUT/sid/'mul-resources.log').open('w') as err,(OUT/sid/'mul-stdout.log').open('w') as out:
            cmd=['/usr/bin/time','-l',sys.executable,'-m','pyroller','align',str(audio),str(OUT/sid/'ordered.txt'),
                '-o',str(folder),'--language','mul','--alignment-mode','ordered','--audio-mode','mix','--no-previews','--offline']
            print('RUN',sid,flush=True);subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err,check=True)
        summarize()


def verify():
    import numpy as np
    import jsonschema
    from pyroller.backends.codec import Vocabulary
    from pyroller.backends.phonemes import PhonemeEncoder
    from pyroller.hybrid.core import align_evidence
    from pyroller.hybrid.config import AlignmentSettings
    schemas=ROOT/'docs/protocol-v2'
    event_schema=read_json(schemas/'event.schema.json')
    resolver=jsonschema.RefResolver(base_uri=schemas.as_uri()+'/',referrer=event_schema)
    event_validator=jsonschema.Draft202012Validator(event_schema,resolver=resolver)
    checks=[]
    for song in read_json(OUT/'experiment.json')['songs']:
        sid=song['id'];folder=OUT/sid/'mul';run=read_json(folder/'run.json');old=read_json(folder/'result.json')
        assert sha256(song['audio'])==song['audio_sha256']
        assert sha256(song['reference'])==song['reference_sha256']
        assert run['config']['language']=='mul' and run['config']['alignment_mode']=='ordered' and run['config']['audio_mode']=='mix'
        manifest=read_json(Path(run['ctc_cache'])/'manifest.json')
        vocab=Vocabulary(read_json(Path(run['ctc_model'])/'vocab.json'),read_json(Path(run['ctc_model'])/'config.json')['pad_token_id'])
        vocab.encoder=PhonemeEncoder.from_metadata(manifest['phoneme_encoding'])
        x=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(Path(run['ctc_cache']).glob('*.npy'))])
        windows=[read_json(p) for p in sorted(Path(run['whisper_cache']).glob('window-*.json'))]
        new=align_evidence((OUT/sid/'ordered.txt').read_text(),windows,x,vocab,run['frame_origin'],run['frame_step'],
            run['duration'],AlignmentSettings(alignment_mode='ordered'))
        new=json.loads(json.dumps(new))
        for key in old.keys()-{'metrics','language','timing_backend','text_diagnostics'}:assert old[key]==new[key],(sid,key)
        doc=read_json(folder/'alignment.json')
        jsonschema.validate(doc,read_json(schemas/'alignment.schema.json'))
        events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        for event in events:event_validator.validate(event)
        assert events[-1]['type']=='completed' and [e['sequence'] for e in events]==list(range(1,len(events)+1))
        row=dict(song=sid,replay_identical=True,events_validated=len(events),sources_unchanged=True)
        checks.append(row);print('VERIFIED',row,flush=True)
    save_json(OUT/'verification.json',checks)
    save_json(OUT/'source-hashes.final.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller').rglob('*.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','summarize','verify']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'summarize':summarize,'verify':verify}[args.action]()
