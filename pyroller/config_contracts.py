"""Backend constructors define supported options; shared rules define constraints."""
import inspect
from pyroller.domain.validation import number

LATIN_LANGUAGES = ["en", "fr", "de", "es", "it", "pt"]
PARSER_LANGUAGES = {"zh_pinyin": ["zh"], "zh_router_pinyin": ["zh"], "en_arpabet": ["en"], "mul_ipa": ["mul"]}
OPTION_RULES = {"repetition": {"choices": ["none", "few", "full"]},
                "min_gap": {"minimum": 0}, "min_match_similarity": {"minimum": 0, "maximum": 1},
                "overlap": {"minimum": 0, "maximum": 1},
                "latin_language": {"choices": LATIN_LANGUAGES},
                "hf_xet": {"choices": ["auto", "on", "off"]}}
QUALITY_OPTIONS = {"mode": {"choices": ["report", "strict"], "default": "report"},
                   "timing_policy": {"choices": ["line", "unit"], "default": "unit"},
                   "min_coverage": {"minimum": 0, "maximum": 1, "default": .8},
                   "max_interpolated_ratio": {"minimum": 0, "maximum": 1, "default": .2},
                   "max_unmatched_seconds": {"minimum": 0, "default": 10}}
WRITER_OPTIONS = {"spacing": {"choices": ["keep", "drop"], "default": "keep"},
                  "by_tag": {"default": "py-roller"}, "tag_type": {"choices": ["k", "K", "kf", "ko"], "default": "kf"},
                  "unmatched_line_duration": {"minimum": .1, "default": .6}}


def validate_rules(config, rules):
    for key, value in config.items():
        rule = rules.get(key, {})
        if "choices" in rule and value not in rule["choices"]:
            raise ValueError(f"Invalid {key}: {value}")
        if "minimum" in rule:
            number(value, key, minimum=rule["minimum"], maximum=rule.get("maximum"))



def constructor_options(factory):
    return {name: parameter for name, parameter in inspect.signature(factory).parameters.items()
            if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)}


def check_options(config, accepted, label, *, allow_backend=True):
    if not isinstance(config, dict):
        raise ValueError(f'{label} config must be an object')
    unknown = set(config) - set(accepted) - ({'backend'} if allow_backend else set())
    if unknown:
        raise ValueError(f'Unknown {label} options: {sorted(unknown)}')


def validate_configs(request):
    from pyroller.aligner.registry import _ALIGNER_FACTORIES
    from pyroller.parser.registry import _PARSER_FACTORIES, _DEFAULT_PARSER_BY_LANGUAGE
    from pyroller.splitter.registry import _SPLITTER_FACTORIES
    from pyroller.transcriber.registry import get_transcriber_config_keys, resolve_transcriber_backend
    from pyroller.filter.registry import build_filter_chain
    from pyroller.writer.registry import list_available_writer_backends
    if not isinstance(request.backend_config, dict):
        raise ValueError('backend_config must be an object')
    configs = request.backend_config
    if set(configs) - {'splitter','filter','transcriber','parser','aligner','writer','quality'}:
        raise ValueError('Unknown backend_config stage')
    for name, config in configs.items():
        if not isinstance(config, dict):
            raise ValueError(f'{name} config must be an object')
        if name in ('parser', 'aligner', 'splitter'):
            factories, default = {'parser': (_PARSER_FACTORIES, _DEFAULT_PARSER_BY_LANGUAGE[request.language.strip().lower()]),
                                  'aligner': (_ALIGNER_FACTORIES, 'global_dp_v1'),
                                  'splitter': (_SPLITTER_FACTORIES, 'demucs')}[name]
            backend = config.get('backend') or default
            if backend not in factories:
                raise ValueError(f'Unsupported {name} backend: {backend}')
            check_options(config, set(constructor_options(factories[backend])) - {"output_dir"}, name)
            if name == 'parser' and request.language.strip().lower() not in PARSER_LANGUAGES[backend]:
                raise ValueError('Parser backend does not support the requested language')
        elif name == 'transcriber':
            _, backend = resolve_transcriber_backend(request.language, config.get('backend'))
            check_options(config, get_transcriber_config_keys(backend), name)
        elif name == 'filter':
            build_filter_chain(config.get('chain', []), request.intermediate_dir, config)
        elif name == 'writer':
            if config.get('backend', 'lrc_ms') not in list_available_writer_backends():
                raise ValueError('Unknown writer backend')
            check_options(config, WRITER_OPTIONS, name)
            validate_rules(config, WRITER_OPTIONS)
            if config.get('spacing', 'keep') not in {'keep','drop'} or config.get('tag_type', 'kf') not in {'k','K','kf','ko'}:
                raise ValueError('Invalid writer spacing or tag_type')
        elif name == 'quality':
            check_options(config, QUALITY_OPTIONS, name, allow_backend=False)
            validate_rules(config, QUALITY_OPTIONS)
            if config.get('mode', 'report') not in {'report','strict'}:
                raise ValueError('Invalid quality mode')
        validate_rules({key: value for key, value in config.items() if value is not None}, OPTION_RULES)
        for key, value in config.items():
            if value is None:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number(value, key, minimum=None)
            if key in {'min_gap','unmatched_line_duration','max_unmatched_seconds','overlap','min_match_similarity','min_coverage','max_interpolated_ratio'}:
                number(value, key, maximum=1 if key in {'overlap','min_match_similarity','min_coverage','max_interpolated_ratio'} else None)
            if key in {'hf_etag_timeout', 'hf_download_timeout', 'segment'}:
                if number(value, key) <= 0:
                    raise ValueError(f'{key} must be positive')
            if key in {'batch_size','target_sample_rate','hf_max_workers','jobs'}:
                if type(value) is not int or value < (0 if key == 'jobs' else 1):
                    raise ValueError(f'{key} must be a valid integer')
            if key == 'latin_language' and (request.language != 'mul' or value not in LATIN_LANGUAGES):
                raise ValueError('latin_language requires mul and a supported explicit language')
            if key in {'local_files_only','vad_filter','trust_remote_code'} and type(value) is not bool:
                raise ValueError(f'{key} must be boolean')


def backend_schemas():
    from pyroller.aligner.registry import _ALIGNER_FACTORIES
    from pyroller.parser.registry import _PARSER_FACTORIES
    from pyroller.splitter.registry import _SPLITTER_FACTORIES
    from pyroller.filter.registry import _FILTER_FACTORIES
    from pyroller.transcriber.specs import TRANSCRIBER_SPECS
    result = {}
    for stage, factories in {'aligner': _ALIGNER_FACTORIES, 'parser': _PARSER_FACTORIES,
                             'splitter': _SPLITTER_FACTORIES, 'filter': _FILTER_FACTORIES}.items():
        result[stage] = {}
        for backend, factory in factories.items():
            result[stage][backend] = {
                name: {'default': p.default if p.default is not inspect.Parameter.empty else None,
                       'type': str(p.annotation), **OPTION_RULES.get(name, {})}
                for name, p in constructor_options(factory).items() if name != 'output_dir'}
    result['transcriber'] = {
        f'{language}:{backend}': {'options': sorted(spec.config_keys - ({'latin_language'} if language != 'mul' else set())), 'language': language,
                                'constraints': {key: rule for key, rule in OPTION_RULES.items() if key in spec.config_keys}}
        for (language, backend), spec in TRANSCRIBER_SPECS.items()}
    result['quality'] = QUALITY_OPTIONS
    from pyroller.writer.registry import list_available_writer_backends
    result['writer'] = {backend: {key: rule for key, rule in WRITER_OPTIONS.items()
                                 if backend == 'ass_karaoke' or key not in {'tag_type', 'unmatched_line_duration'}}
                        for backend in list_available_writer_backends()}
    result['parser_languages'] = PARSER_LANGUAGES
    result['latin_languages'] = LATIN_LANGUAGES
    return result


def cli_option_metadata():
    # Read the actual argparse definitions so frontend discovery cannot drift
    # from flag names, defaults or enum choices.
    from pyroller.cli.main import build_parser
    from pathlib import Path
    _, run_parser, _ = build_parser()
    options = []
    for action in run_parser._actions:
        if not action.option_strings or action.dest == 'help':
            continue
        item = {'name': action.dest, 'flags': action.option_strings,
                'default': str(action.default) if isinstance(action.default, Path) else action.default,
                'required': action.required}
        if action.choices is not None:
            item['choices'] = list(action.choices)
        item['type'] = ('choice' if action.choices is not None else 'boolean' if action.nargs == 0 else
                        'path' if action.type is Path else getattr(action.type, '__name__', 'string'))
        if action.dest == 'writer_spacing':
            item['default'] = WRITER_OPTIONS['spacing']['default']
        if action.dest == 'filter_chain':
            item['type'] = 'string_list'
        prefix = action.dest.split('_')[0]
        stage = {'splitter':'s', 'filter':'f', 'transcriber':'t', 'parser':'p', 'aligner':'a', 'writer':'w'}.get(prefix)
        if stage:
            item['stages'] = [stage]
        if action.dest in {'language', 'stages'}:
            item['stages'] = ['s','f','t','p','a','w']
        if action.dest == 'stages':
            item['type'] = 'stage_chain'
            item['choices'] = ['s','f','t','p','a','w']
        options.append(item)
    return options
