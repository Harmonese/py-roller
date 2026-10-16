"""Check metadata and distribution contents without importing the neural runtime."""
import argparse
from email.parser import Parser
from pathlib import Path
import re
import tarfile
import zipfile


def check(folder):
    files = sorted(Path(folder).glob('*'))
    wheels = [p for p in files if p.suffix == '.whl']
    sources = [p for p in files if p.name.endswith('.tar.gz')]
    assert wheels and sources, 'Build both wheel and sdist first'
    for path in wheels + sources:
        if path.suffix == '.whl':
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                metadata = Parser().parsestr(archive.read(next(n for n in names if n.endswith('/METADATA'))).decode())
                assert metadata['Version'] == '0.10.2'
                assert 'pyroller/cli.py' in names and 'pyroller/protocol.py' in names
                assert any('pyroller/hybrid/_alignment.' in n and n.endswith(('.so', '.pyd')) for n in names)
        else:
            with tarfile.open(path) as archive:
                names = [n.split('/',1)[-1] for n in archive.getnames()]
                assert 'docs/PROTOCOL_V2.md' in names and 'docs/protocol-v2/request.schema.json' in names
                assert 'pyroller/hybrid/native.cpp' in names
        for name in names:
            assert not re.search(r'(^|/)(artifacts|py-roller-artifacts|src-reconstruct|\.venv|__pycache__)(/|$)',name), name
            assert not name.startswith(('pyroller/transcriber/', 'pyroller/pipeline/', 'pyroller/aligner/', 'pyroller/cli/')), name
            assert not name.endswith(('.mp3', '.wav', '.flac', '.bin', '.safetensors')), name
        print(f'OK {path.name}: {len(names)} entries')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dist',nargs='?',default='dist')
    check(parser.parse_args().dist)
