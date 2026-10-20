"""Frozen two-song Mandarin acceptance. LRC times are evaluation-only."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pyroller.hybrid.io import read_json,save_json,atomic_write,sha256
from pyroller.language import normalize
from evaluation import read_lrc

OUT=ROOT/'py-roller-artifacts/chinese-validation'
LIBRARY=Path('/Volumes/StudioVantablack/Main/14 Music Library/Albums')
SONGS={
    'Z01':LIBRARY/'万能青年旅店 - 万能青年旅店/04 大石碎胸口.mp3',
    'Z02':LIBRARY/'簡諧點源 - 血紅都會/07 簡諧點源 - 從流水線上來，到流水線上去.mp3',
}


def prepare():
    songs=[]
    for sid,audio in SONGS.items():
        reference=audio.with_suffix('.lrc');rows=read_lrc(reference);disputed=[]
        if sid=='Z01':
            disputed=[r for r in rows if normalize(r['text'])==normalize('大石碎胸口')]
            rows=[r for r in rows if r not in disputed]
        folder=OUT/sid;save_json(folder/'reference.json',rows)
        save_json(folder/'disputed.json',dict(rows=disputed,reason='possible title; generated LRC; not human-verified'))
        text='\n'.join(r['text'] for r in rows)+'\n'
        atomic_write(folder/'ordered.txt',text)
        seen=set();unique=[]
        for row in rows:
            key=normalize(row['text'])
            if key not in seen:unique.append(row['text']);seen.add(key)
        atomic_write(folder/'unordered.txt','\n'.join(unique)+'\n')
        songs.append(dict(id=sid,audio=str(audio),audio_sha256=sha256(audio),reference=str(reference),
            reference_sha256=sha256(reference),reference_lines=len(rows),disputed_lines=len(disputed)))
    save_json(OUT/'experiment.json',dict(songs=songs,language='zh',parameter_scan=False,
        prompt=None,reference_times_used_in_inference=False,views=['mix','vocals'],modes=['ordered','unordered']))


def evaluate(sid,view,mode):
    folder=OUT/sid/view/mode
    original_folder=folder
    if (folder.parent/(mode+'-final')/'result.json').exists():folder=folder.parent/(mode+'-final')
    refs=read_json(OUT/sid/'reference.json')
    result=read_json(folder/('replay-result.json' if (folder/'replay-result.json').exists() else 'result.json'))
    if mode=='ordered':
        located={r['input_line_index']:r for r in result['occurrences']}
        pairs=[dict(index=i,reference=refs[i],output=r,error=r['start']-refs[i]['start']) for i,r in located.items()]
        missing=[dict(index=i,**r) for i,r in enumerate(refs) if i not in located];extra=[]
    else:
        from validate_scoring import evaluate as identity_evaluate
        ev=identity_evaluate(result['occurrences'],refs)
        pairs,missing,extra=ev['pairs'],ev['missing'],ev['extra']
    vocabulary_source=read_json(folder/'run.json')['ctc_model']
    vocab=read_json(Path(vocabulary_source)/'vocab.json')
    from pyroller.backends.codec import encode_units,Vocabulary
    from pyroller.hybrid.ordered import align_words
    from pyroller.language import independent_support
    from pyroller.hybrid.text_distance import phrase_comparator
    words=result['stitched_words'];target=[];spans=[]
    for r in refs:
        a=len(target);target.extend(normalize(r['text']).split());spans.append((a,len(target)))
    mappings,_=align_words(target,[w['token'] for w in words])
    classified=[]
    for row in missing:
        index=row['index'];a,b=spans[index]
        matches=[(li,wi,c) for li,wi,c in mappings if a<=li<b]
        source_count=independent_support([dict(asr_word=wi) for li,wi,c in matches],words)
        support=len({li for li,wi,c in matches})/max(1,b-a)
        # A post-hoc attribution diagnostic, never a search or acceptance rule.
        try:encode_units(row['text'],Vocabulary(vocab,read_json(Path(vocabulary_source)/'config.json')['pad_token_id']));unknown=False
        except ValueError:unknown=True
        candidates=[c for c in result.get('candidates',[]) if normalize(c['text'])==normalize(row['text'])
                    and abs(c['start']-row['start'])<=2]
        ordered_reason=next((r['reason'] for r in result.get('unaligned_lines',[]) if r['input_line_index']==index),None)
        recovery=next((r for r in result.get('recovery_regions',[]) if r['input_line_index']==index),{})
        coarse=any(r.get('input_line_index')==index for r in result.get('coarse_occurrences',[])) if mode=='ordered' else False
        if mode=='ordered' and unknown and 'encode' in (recovery.get('error') or ''):category='unencodable'
        elif ordered_reason in ('region_budget_or_empty','overlapping_assignment') or coarse:category='timing_or_budget'
        elif candidates:category='matching_competition'
        elif support>=.5 and source_count>=2:category='matching_with_asr_clues'
        else:category='weak_asr_clues_not_proof_of_no_ctc_evidence'
        classified.append(dict(row,category=category,asr_unit_coverage=support,independent_sources=source_count,
            ordered_reason=ordered_reason,near_candidates=len(candidates),
            has_unencodable_characters=unknown,
            recovery_status=recovery.get('status'),recovery_error=recovery.get('error'),
            ctc_supported_units=sum(w['character_support']>=.5 for w in recovery.get('word_evidence',[])),
            ctc_attempted_units=len(recovery.get('word_evidence',[]))))
    usage=(folder/'resource.log').read_text() if (folder/'resource.log').exists() else ''
    match=re.search(r'(\d+)\s+maximum resident set size',usage)
    run=read_json(folder/'run.json')
    cache_bytes=sum(p.stat().st_size for key in ('ctc_cache','whisper_cache') for p in Path(run[key]).rglob('*') if p.is_file())
    timing_gains=[];timing_losses=[]
    for pair in pairs:
        output=pair['output']
        coarse=next((r for r in result.get('coarse_occurrences',[]) if (
            r.get('input_line_index')==pair['index'] if mode=='ordered' else
            (r['line_id'],r['token_start'],r['token_end'])==(output['line_id'],output['token_start'],output['token_end']))),None)
        if coarse is None:continue
        before=coarse['start']-pair['reference']['start'];after=pair['error']
        detail=dict(index=pair['index'],text=pair['reference']['text'],reference_start=pair['reference']['start'],
                    coarse_error=before,final_error=after)
        if abs(before)>.5 and abs(after)<=.5:timing_gains.append(detail)
        elif abs(before)<=.5 and abs(after)>.5:timing_losses.append(detail)
    row=dict(song=sid,view=view,mode=mode,reference_lines=len(refs),located=len(result['occurrences']),missing=len(missing),
        over_half_second=sum(abs(p['error'])>.5 for p in pairs),over_two_seconds=sum(abs(p['error'])>2 for p in pairs),
        within_half_second=sum(abs(p['error'])<=.5 for p in pairs),extra=len(extra),metrics=result['metrics'],
        rss_bytes=int(match.group(1)) if match else None,evidence_bytes=cache_bytes,
        pairs=pairs,missing_details=classified,extra_details=extra,ctc_timing_gains=timing_gains,ctc_timing_losses=timing_losses)
    row['output_folder']=str(folder.relative_to(OUT))
    row['lrc_file']='replay.lrc' if (folder/'replay.lrc').exists() else 'result.lrc'
    if 'original_metrics' in result:row['cold_run_metrics']=result['original_metrics']
    if folder!=original_folder:
        row['initial_metrics']=read_json(original_folder/'run.json')['metrics']
        original_usage=(original_folder/'resource.log').read_text()
        original_rss=re.search(r'(\d+)\s+maximum resident set size',original_usage)
        row['initial_rss_bytes']=int(original_rss.group(1)) if original_rss else None
    save_json(folder/'evaluation.json',row)
    windows=[read_json(p) for p in sorted(Path(run['whisper_cache']).glob('window-*.json'))]
    atomic_write(folder/'raw-transcript.txt','\n\n'.join(
        f"[{w['start']:.2f}-{w['end']:.2f}] "+' '.join(s['text'] for s in w['segments']) for w in windows)+'\n')
    return row


def summarize():
    rows=[]
    for sid in SONGS:
        for view in ('mix','vocals'):
            for mode in ('ordered','unordered'):
                if (OUT/sid/view/mode/'result.json').exists():rows.append(evaluate(sid,view,mode))
    save_json(OUT/'summary.json',rows)
    md=['# 两首中文歌曲固定验收','',
        '参考 LRC 不视为已核实人工真值；Z01 题名行单列争议。参考时间不进入推理。',
        'ordered 按输入实例编号配对；unordered 按规范化身份与重复实例匹配，远处错误实例另列 >2 秒。','',
        '|歌曲|音轨|模式|参考行|未定位|已定位 >0.5s|其中 >2s|<=0.5s|额外|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append('|'+ '|'.join(str(r[k]) for k in ('song','view','mode','reference_lines','missing','over_half_second','over_two_seconds','within_half_second','extra'))+'|')
    for r in rows:
        md+=['',f"## {r['song']} / {r['view']} / {r['mode']}",'',
            f"[LRC]({r['output_folder']}/{r['lrc_file']}) · [逐句记录]({r['output_folder']}/evaluation.json)",'',
            '### 未定位','']
        if not r['missing_details']:md.append('无。')
        for m in r['missing_details']:
            md.append(f"- {m['start']:.2f}s `{m['text']}`：{m['category']}；ASR 单位覆盖 {m['asr_unit_coverage']:.0%}，独立来源 {m['independent_sources']}，引擎原因 {m['ordered_reason']}；CTC 支持 {m['ctc_supported_units']}/{m['ctc_attempted_units']}。")
        md+=['','### 已定位但时间差 >0.5 秒','']
        bad=[p for p in r['pairs'] if abs(p['error'])>.5]
        if not bad:md.append('无。')
        for p in sorted(bad,key=lambda p:p['index']):
            md.append(f"- {p['reference']['start']:.2f}s `{p['reference']['text']}` → {p['output']['start']:.2f}s（{p['error']:+.2f}s，{p['output']['timing_method']}）。")
        md+=['','### CTC 相对同一实例粗时间的变化','']
        for label,key in [('改善','ctc_timing_gains'),('退步','ctc_timing_losses')]:
            md.append(f"{label} {len(r[key])} 行。")
            for p in r[key]:md.append(f"- {p['reference_start']:.2f}s `{p['text']}`：{p['coarse_error']:+.2f}s → {p['final_error']:+.2f}s。")
        md+=['','### 开销','',f"峰值 RSS（主进程）：{r['rss_bytes']} bytes；CTC+Whisper 证据缓存：{r['evidence_bytes']} bytes。",'',
             '```json',json.dumps(r['metrics'],ensure_ascii=False,indent=2),'```']
        if 'initial_metrics' in r:md+=['','本组修复后仅复用缓存重放；初次冷运行开销另列：','```json',json.dumps(r['initial_metrics'],indent=2),'```']
        elif 'cold_run_metrics' in r:md+=['','初始任务开销（模型与分离只发生在此处，最终回放不重跑）：','```json',json.dumps(r['cold_run_metrics'],indent=2),'```']
    atomic_write(OUT/'summary.md','\n'.join(md)+'\n')
    for r in rows:print({k:r[k] for k in ('song','view','mode','reference_lines','located','missing','over_half_second','over_two_seconds','extra')},flush=True)


def run(songs):
    save_json(OUT/'source-hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller').rglob('*.py')})
    for sid in songs:
        for view in ('mix','vocals'):
            for mode in ('ordered','unordered'):
                folder=OUT/sid/view/mode
                if (folder/'result.json').exists():continue
                folder.parent.mkdir(parents=True,exist_ok=True)
                resource=folder.parent/(mode+'-resource.log')
                stdout=folder.parent/(mode+'-stdout.log')
                with resource.open('w') as err,stdout.open('w') as out:
                    command=['/usr/bin/time','-l',sys.executable,'-m','pyroller','align',str(SONGS[sid]),
                        str(OUT/sid/(mode+'.txt')),'-o',str(folder),'--language','zh',
                        '--alignment-mode',mode,'--audio-mode',view,'--no-previews','--offline']
                    print('RUN',sid,view,mode,flush=True)
                    subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=True)
                atomic_write(folder/'resource.log',resource.read_text())
                summarize()


def replay_final():
    """Freeze final algorithm outputs without repeating any neural inference."""
    import numpy as np
    from pyroller.backends.codec import Projection,Vocabulary,text_diagnostics
    from pyroller.hybrid.core import align_evidence
    from pyroller.hybrid.config import AlignmentSettings
    from pyroller.hybrid.io import lrc_text
    from pyroller.protocol import alignment_document
    save_json(OUT/'source-hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller').rglob('*.py')})
    for sid in SONGS:
        for view in ('mix','vocals'):
            for mode in ('ordered','unordered'):
                folder=OUT/sid/view/mode
                if (folder.parent/(mode+'-final')/'result.json').exists():folder=folder.parent/(mode+'-final')
                run=read_json(folder/'run.json');previous=read_json(folder/'result.json')
                text=(OUT/sid/(mode+'.txt')).read_text()
                model=Path(run['ctc_model']);vocabulary=Vocabulary(read_json(model/'vocab.json'),read_json(model/'config.json')['pad_token_id'])
                projection=Projection(vocabulary,text)
                manifest=read_json(Path(run['ctc_cache'])/'manifest.json')
                assert manifest['projection']==projection.metadata
                logp=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(Path(run['ctc_cache']).glob('*.npy'))])
                windows=[read_json(p) for p in sorted(Path(run['whisper_cache']).glob('window-*.json'))]
                result=align_evidence(text,windows,logp,projection.vocabulary,run['frame_origin'],run['frame_step'],run['duration'],
                    settings=AlignmentSettings(alignment_mode=mode))
                result.update(language='zh',timing_backend='wav2vec2_ctc',text_diagnostics=text_diagnostics(text,projection.vocabulary),
                    original_metrics=previous['metrics'])
                save_json(folder/'replay-result.json',result)
                save_json(folder/'replay-alignment.json',alignment_document(result,run['request_id'],run['duration']))
                atomic_write(folder/'replay.lrc',lrc_text(result['occurrences']))
                print('FINAL',sid,view,mode,len(result['occurrences']),flush=True)
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','summarize','replay'])
    parser.add_argument('--songs',nargs='+',choices=list(SONGS),default=list(SONGS))
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.songs)
    elif args.action=='replay':replay_final()
    else:summarize()
