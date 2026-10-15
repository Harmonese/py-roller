import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_write(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle: handle.write(text)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def save_json(path, value):
    atomic_write(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_array(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle: np.save(handle,value,allow_pickle=False)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def decode_audio(path, sample_rate=16000):
    result=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ar',str(sample_rate),'-ac','1',
        '-f','f32le','pipe:1'],capture_output=True,check=True)
    samples=np.frombuffer(result.stdout,dtype='<f4').copy()
    if len(samples)<400 or not np.isfinite(samples).all():
        raise ValueError('Audio is empty, too short or contains nonfinite samples')
    return samples


def lrc_text(rows):
    lines=[]
    for row in rows:
        cents=max(0,round(row['start']*100));minutes,rest=divmod(cents,6000);seconds,fraction=divmod(rest,100)
        lines.append(f'[{minutes:02d}:{seconds:02d}.{fraction:02d}] {row["text"]}')
    return '\n'.join(lines)+ ('\n' if lines else '')
