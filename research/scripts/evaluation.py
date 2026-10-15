"""Reference parsing used only by offline research tools, never inference."""
from pathlib import Path
import re
from pyroller.hybrid.matching import normalize


def read_lrc(path):
    raw = Path(path).read_text(encoding='utf-8-sig')
    match = re.search(r'\[offset:([+-]?\d+)\]', raw, re.I)
    offset = int(match.group(1))/1000 if match else 0
    rows = []
    for line in raw.splitlines():
        text = re.sub(r'\[[^\]]+\]', '', line).strip()
        if not text:
            continue
        for minute, second in re.findall(r'\[(\d+):(\d+(?:\.\d+)?)\]', line):
            rows.append(dict(start=60*int(minute)+float(second)+offset,
                             text=text, normalized=normalize(text)))
    return sorted(rows, key=lambda row: row['start'])
