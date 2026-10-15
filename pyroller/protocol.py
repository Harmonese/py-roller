"""Protocol v2: validated requests and compact, revision-replacing events."""
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from . import __version__
from .config import EngineConfig

PROTOCOL_VERSION = 2
OPTION_NAMES = frozenset(EngineConfig.__dataclass_fields__)
PATH_OPTIONS = {'cache_dir', 'whisper_model', 'ctc_model'}
LINE_FIELDS = ('occurrence_id', 'line_id', 'text', 'start', 'end', 'status',
               'finalized', 'content_status', 'timing_status', 'timing_method')
TERMINAL_EVENTS = {'completed', 'failed', 'cancelled'}


def envelope(request_id):
    return dict(protocol_version=PROTOCOL_VERSION, engine='py-roller',
                engine_version=__version__, request_id=request_id)


def public_lines(rows):
    return [{key: row[key] for key in LINE_FIELDS if key in row} for row in rows]


def alignment_document(result, request_id, duration):
    return dict(envelope(request_id), status='complete', duration=duration,
                lines=public_lines(result['occurrences']))


def error_code(exc):
    if isinstance(exc, FileExistsError): return 'output_exists'
    if isinstance(exc, FileNotFoundError): return 'file_not_found'
    if isinstance(exc, (ValueError, TypeError)): return 'invalid_input'
    if isinstance(exc, ImportError): return 'dependency_missing'
    return 'execution_failed'


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class Request:
    request_id: str
    audio: Path
    lyrics: Path
    output: Path
    config: EngineConfig


def load_request(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Invalid JSON number: {value}')))
    if not isinstance(data, dict): raise ValueError('Request must be a JSON object')
    required = {'protocol_version', 'request_id', 'audio', 'lyrics', 'output'}
    if required - data.keys(): raise ValueError(f'Missing request fields: {sorted(required-data.keys())}')
    if data.keys() - required - {'options'}: raise ValueError('Unknown request fields')
    if type(data['protocol_version']) is not int or data['protocol_version'] != 2:
        raise ValueError('Only protocol_version 2 is supported')
    identifier = data['request_id']
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128:
        raise ValueError('request_id must be a nonempty string of at most 128 characters')
    if any(ord(char) < 32 for char in identifier): raise ValueError('Invalid request_id characters')

    def resolve(value):
        if not isinstance(value, str) or not value.strip(): raise ValueError('Paths must be nonempty strings')
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else path.parent/candidate).resolve()

    options = data.get('options', {})
    if not isinstance(options, dict) or options.keys() - OPTION_NAMES:
        raise ValueError('Unknown or invalid request options')
    options = dict(options)
    for key in PATH_OPTIONS & options.keys():
        options[key] = resolve(options[key])
    return Request(identifier, resolve(data['audio']), resolve(data['lyrics']),
                   resolve(data['output']), EngineConfig(**options))


class EventStream:
    """One sequence and one terminal outcome per invocation; no internal paths leaked into lines."""
    def __init__(self, request_id=None, sink=None):
        self.request_id = request_id or str(uuid.uuid4())
        self.sink = sink
        self.sequence = 0
        self.terminal = False
        self.file = None
        self.pending = []

    def attach(self, path):
        self.file = Path(path).open('x', encoding='utf-8')
        try:
            for line in self.pending: self.file.write(line)
            self.file.flush()
        except OSError:
            try: self.file.close()
            except OSError: pass
            self.file = None
            raise
        self.pending.clear()

    def emit(self, event):
        if self.terminal: raise RuntimeError('Cannot emit after a terminal event')
        event = dict(event)
        if event['type'] == 'snapshot':
            event.pop('schema_version', None)
            event['lines'] = public_lines(event.pop('occurrences'))
        self.sequence += 1
        payload = dict(event, **envelope(self.request_id), sequence=self.sequence)
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)+'\n'
        if self.file:
            try:
                self.file.write(line)
                self.file.flush()
            except OSError:
                # A full disk must not prevent reporting failure on stdout.
                try: self.file.close()
                except OSError: pass
                self.file = None
                self.sequence -= 1
                raise
        else:
            self.pending.append(line)
        self.terminal = event['type'] in TERMINAL_EVENTS
        if self.sink: self.sink(payload)
        return payload

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
