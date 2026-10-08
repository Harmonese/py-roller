"""Completion receipts validate both inputs/configuration and published bytes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from pyroller import __version__
from pyroller.utils.json import write_json


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest_stream(stream)


def _digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def receipt_path(task):
    path = task.expected_outputs[0]
    return path.with_name('.' + path.name + '.pyroller-complete.json')


def input_fingerprint(task):
    data = asdict(task.request)
    for key in ('intermediate_dir', 'log_level', 'cleanup'):
        data.pop(key, None)
    hashes = {}
    for key, value in data.items():
        if key.endswith('_path') and not key.startswith('output_') and value is not None:
            hashes[key] = digest(Path(value))
    return hashlib.sha256(json.dumps({'version': __version__, 'request': data, 'inputs': hashes},
                                    sort_keys=True, default=str).encode()).hexdigest()


def output_fingerprints(task):
    if not task.expected_outputs:
        raise ValueError('No declared outputs')
    result = {}
    for path in task.expected_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty output: {path}')
        result[str(path.resolve())] = digest(path)
    return result


def completed(task):
    if not task.expected_outputs:
        return False
    try:
        data = json.loads(receipt_path(task).read_text())
        return data['inputs'] == input_fingerprint(task) and data['outputs'] == output_fingerprints(task)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def record_completion(task, fingerprint, quality=None):
    if task.expected_outputs:
        write_json({'inputs': fingerprint, 'outputs': output_fingerprints(task), 'quality': quality}, receipt_path(task))


def completion_quality(task):
    try:
        return json.loads(receipt_path(task).read_text()).get('quality')
    except (OSError, ValueError, AttributeError):
        return None
