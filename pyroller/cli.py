"""Human CLI and machine worker share the public file-engine API."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import signal
import sys
import threading

from . import __version__
from .config import EngineConfig
from .protocol import EventStream, error_code, load_request


def align(*args, **kwargs):
    from .engine import align as run_alignment
    return run_alignment(*args, **kwargs)


def _json(value, file=None):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), file=file, flush=True)


def _model_options(parser):
    parser.add_argument('--cache-dir', type=Path, default=EngineConfig.cache_dir)
    parser.add_argument('--whisper-model', type=Path, help='Local Whisper checkpoint directory')
    parser.add_argument('--ctc-model', type=Path, help='Local CTC checkpoint directory')


def parser():
    root = argparse.ArgumentParser(prog='py-roller', description='Align English plain lyrics to songs with Whisper large-v3 and CTC.')
    root.add_argument('--version', action='version', version=f'py-roller {__version__} (protocol 2)')
    commands = root.add_subparsers(dest='command', required=True)
    human = commands.add_parser('align', help='Create LRC from an audio file and untimed lyrics')
    human.add_argument('audio', type=Path)
    human.add_argument('lyrics', type=Path, help='UTF-8 plain lyrics, one line per phrase')
    human.add_argument('-o', '--output', type=Path, required=True, help='New output directory')
    human.add_argument('--audio-mode', choices=['mix', 'vocals'], default='mix')
    human.add_argument('--alignment-mode', choices=['unordered', 'ordered'], default=None)
    human.add_argument('--ctc-device', choices=['auto', 'cpu', 'mps'], default='auto')
    human.add_argument('--cpu-threads', type=int, default=4)
    human.add_argument('--offline', action='store_true')
    human.add_argument('--no-previews', action='store_true')
    human.add_argument('--json-events', action='store_true', help='Protocol v2 JSONL on stdout')
    _model_options(human)
    worker = commands.add_parser('run', help='Run a protocol v2 request; stdout is JSONL')
    worker.add_argument('--request', required=True, type=Path)
    doctor = commands.add_parser('doctor', help='Check dependencies and model availability without downloading')
    doctor.add_argument('--json', action='store_true')
    doctor.add_argument('--audio-mode', choices=['mix', 'vocals'], default='mix')
    _model_options(doctor)
    models = commands.add_parser('models', help='Inspect or download pinned production models')
    models.add_argument('action', choices=['list', 'download', 'migrate', 'remove'])
    models.add_argument('--kind', choices=['all', 'whisper', 'ctc', 'demucs'], default='all')
    models.add_argument('--apply', action='store_true', help='Apply model removal (otherwise preview only)')
    models.add_argument('--json', action='store_true')
    _model_options(models)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    machine = args.command == 'run' or getattr(args, 'json_events', False)
    stdout = sys.stdout
    last_stage = None
    started = False

    def progress(event):
        nonlocal last_stage, started
        started = True
        if machine:
            _json(event, stdout)
        elif event['type'] == 'snapshot':
            print(f"{'Final' if event['final'] else 'Preview'}: {len(event['lines'])} lines", file=sys.stderr)
        elif event['type'] == 'progress' and event['stage'] != last_stage:
            last_stage = event['stage']
            print(f'Processing {last_stage}...', file=sys.stderr)

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.command in ('doctor', 'models'):
            from .doctor import inspect, model_inventory
            config = EngineConfig(cache_dir=args.cache_dir, whisper_model=args.whisper_model,
                                  ctc_model=args.ctc_model, audio_mode=getattr(args, 'audio_mode', 'mix'))
            with redirect_stdout(sys.stderr):
                if args.command == 'doctor':
                    report = inspect(config)
                else:
                    from .backends.catalog import MODELS, resolve_model, migrate_model, remove_model
                    if args.apply and args.action != 'remove':
                        raise ValueError('--apply is only valid for models remove')
                    operations = []
                    for kind in MODELS if args.kind == 'all' else (args.kind,):
                        if args.action == 'download': resolve_model(kind, config)
                        elif args.action == 'migrate': migrate_model(kind, config)
                        elif args.action == 'remove':
                            target = remove_model(kind, config, args.apply)
                            operations.append(dict(kind=kind, path=str(target), applied=args.apply))
                    report = dict(protocol_version=2, engine='py-roller', engine_version=__version__,
                                  type='models', models=model_inventory(config), operations=operations)
                    if args.kind != 'all': report['models'] = [m for m in report['models'] if m['kind'] == args.kind]
            if args.json: _json(report, stdout)
            elif args.command == 'doctor':
                for check in report['checks']: print(f"{'OK' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
                for model in report['models']: print(f"{'OK' if model['available'] else 'MISSING'} {model['kind']}: {model['repo']}")
            else:
                for model in report['models']: print(f"{model['kind']}: {model['repo']} [{model['revision']}]\n  {model['path'] or 'not downloaded'}")
                for operation in report.get('operations', []):
                    print(f"{'Removed' if operation['applied'] else 'Would remove'}: {operation['path']}")
                for model in report['models']:
                    if model.get('legacy_path'): print(f"  Migration source ({model['kind']}): {model['legacy_path']}")
            return 0 if args.command == 'models' or report['ready'] else 1
        if args.command == 'run':
            request = load_request(args.request)
            audio, lyrics, output, config, request_id = request.audio, request.lyrics, request.output, request.config, request.request_id
        else:
            audio, lyrics, output, request_id = args.audio, args.lyrics, args.output, None
            config = EngineConfig(cache_dir=args.cache_dir, whisper_model=args.whisper_model, ctc_model=args.ctc_model,
                **({'alignment_mode': args.alignment_mode} if args.alignment_mode is not None else {}),
                audio_mode=args.audio_mode, ctc_device=args.ctc_device, cpu_threads=args.cpu_threads,
                offline=args.offline, previews=not args.no_previews)
        with redirect_stdout(sys.stderr):
            align(audio, lyrics, output, config, progress, request_id=request_id)
        if not machine: print(output.resolve()/'result.lrc', file=stdout)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        cancelled = isinstance(exc, KeyboardInterrupt)
        if machine and not started:
            stream = EventStream(sink=progress)
            stream.emit(dict(type='started'))
            stream.emit(dict(type='cancelled' if cancelled else 'failed',
                code='cancelled' if cancelled else error_code(exc), message=str(exc) or 'Cancelled'))
        elif getattr(args, 'json', False):
            _json(dict(protocol_version=2, engine='py-roller', engine_version=__version__,
                type=args.command, ready=False, code=error_code(exc), message=str(exc)), stdout)
        print(f'py-roller: {str(exc) or "Cancelled; completed windows remain cached."}', file=sys.stderr)
        return 130 if cancelled else 1
    finally:
        if previous is not None: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
