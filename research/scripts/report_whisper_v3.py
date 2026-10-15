"""Build Chinese comparison tables and evidence excerpts without new inference."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pyroller.hybrid.io import atomic_write
from pyroller.hybrid.matching import normalize, window_words

NAMES = {'E01': 'Blossom', 'E02': 'Gloria', 'E05': 'I Was a Teenage Anarchist',
         'E06': 'Within', 'E08': 'Fly for Your Life', 'E18': 'Lily'}


def read(path):
    return json.loads(path.read_text())


def main():
    folder = Path(sys.argv[1])
    rows = read(folder/'summary.json')
    lines = ['# Whisper large-v3 扩展测试逐句数据', '',
        '以下均为最终歌词对齐结果。缺行按归一化整行文本及重复次数统计；同文实例先最大数量配对，再最小化总起点误差。',
        '这不是人工核听确认的召回率：远处同文重复可能掩盖漏句。另列 ±2 秒内配对数与逐实例变化。', '',
        '| 歌曲 | 音源 | 模型 | 参考行数 | 缺行 | 已配对但偏差 >0.5秒 | 其中 >2秒 | ±0.5秒内 | ±2秒内 | 多余行 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {NAMES[r['song']]} | {'原声' if r['view']=='mix' else '分离人声'} | {r['model']} | "
            f"{len(r['missing'])+len(r['pairs'])} | {len(r['missing'])} | "
            f"{sum(abs(p['error'])>.5 for p in r['pairs'])} | {r['over_two']} | {r['within_half']} | "
            f"{sum(abs(p['error'])<=2 for p in r['pairs'])} | {len(r['extra'])} |")
    for r in rows:
        if r['model'] != 'v3': continue
        sid, view = r['song'], r['view']
        base = next(b for b in rows if (b['song'],b['view'],b['model'])==(sid,view,'v2'))
        lines += ['', f"## {NAMES[sid]} / {'原声' if view=='mix' else '分离人声'}"]
        bm = {p['index']: p for p in base['missing']}
        nm = {p['index']: p for p in r['missing']}
        lines += ['', '### 缺行变化', '']
        for label, ids, source in [('新增缺行',nm.keys()-bm.keys(),nm), ('不再缺行',bm.keys()-nm.keys(),bm)]:
            for i in sorted(ids):
                p = source[i]
                lines.append(f"- {label}：{p['start']:.2f}s，{p['text']}")
        for label in ('near_gains','near_losses'):
            lines += ['', '### '+('新进入 ±2 秒范围' if label=='near_gains' else '退出 ±2 秒范围'), '']
            if not r[label]: lines.append('无。')
            for p in r[label]:
                lines.append(f"- {p['reference']['start']:.2f}s，{p['reference']['text']}")
        changed = {p['index']: p['reference'] for key in ('near_gains','near_losses') for p in r[key]}
        changed.update({i:nm[i] for i in nm.keys()-bm.keys()})
        lines += ['', '### 变化位置的原始证据', '',
            '片段按模型时间戳截取参考点前后 7 秒，仅用于诊断；模型时间本身可能偏移。候选检查范围为起点 ±4 秒。']
        data = {m:read(folder/sid/view/m/'result.json') for m in ('v2','v3')}
        windows = {m:read(folder/sid/view/m/'raw-windows.json') for m in ('v2','v3')}
        for i,p in sorted(changed.items()):
            t = p['start']; identity = normalize(p['text'])
            lines += ['', f"#### {t:.2f}s：{p['text']}", '']
            for m in ('v2','v3'):
                for w in windows[m]:
                    ws = [x for x in window_words(w) if t-7 <= x['start'] <= t+7]
                    if ws: lines.append(f"- {m} 窗口 {w['index']}："+' '.join(x['token'] for x in ws))
                ds = data[m]
                near = lambda c: normalize(c['text'])==identity and abs(c['start']-t)<=4
                candidates = [c for c in ds['candidates'] if near(c)]
                coarse = [c for c in ds['coarse_occurrences'] if near(c)]
                lines.append(f"- {m} 同文邻近候选数：{len(candidates)}；定时前已选实例："+
                    ', '.join(f"{c['start']:.2f}s" for c in coarse))
                rejected = [c for c in ds['discovery_diagnostics'].get('rejected',[]) if near(c)]
                if rejected:
                    best = max(rejected,key=lambda c:c['similarity'])
                    lines.append(f"- {m} 保留的最佳拒绝记录：{best['rejection']}，相似度 {best['similarity']:.3f}")
        lines += ['', '### 完整漏句清单', '']
        for row in (base, r):
            lines.append(f"**{row['model']}**")
            lines.append('')
            if not row['missing']: lines.append('无。')
            for p in sorted(row['missing'], key=lambda p:p['start']):
                lines.append(f"- {p['start']:.2f}s：{p['text']}")
            lines.append('')
        lines += ['### 时间偏差逐句对照', '',
            '任一模型起点误差超过 0.5 秒的实例列在下表。正数表示晚，负数表示早。缺行表示没有同文配对。', '',
            '| 参考时间 | 歌词 | v2 起点误差 | v3 起点误差 |', '|---:|---|---:|---:|']
        bp = {p['index']:p for p in base['pairs']}
        np = {p['index']:p for p in r['pairs']}
        for i in sorted(bp.keys()|np.keys()):
            a,b = bp.get(i),np.get(i)
            if not any(p and abs(p['error'])>.5 for p in (a,b)): continue
            ref = (a or b)['reference']
            errors = [f"{p['error']:+.2f}s" if p else '缺行' for p in (a,b)]
            lines.append(f"| {ref['start']:.2f}s | {ref['text']} | {errors[0]} | {errors[1]} |")
    atomic_write(folder/'中文逐句对照.md', '\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
