"""Semantic validation shared by persisted artifacts and in-memory alignment."""
import math


def number(value, label, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{label} is outside its allowed range')
    return value


def _times(unit, label):
    start = number(unit.get('start_time'), label + '.start_time')
    end = number(unit.get('end_time'), label + '.end_time')
    if end < start:
        raise ValueError(f'{label} ends before it starts')
    return start, end


def validate_payload(payload, kind):
    if not isinstance(payload, dict):
        raise ValueError('Artifact payload must be an object')
    if not isinstance(payload.get('language'), str):
        raise ValueError('Artifact language must be a string')
    for field in ('metadata', 'report'):
        if field in payload and not isinstance(payload[field], dict):
            raise ValueError(f'{field} must be an object')
    duration = payload.get('metadata', {}).get('audio_duration')
    if duration is not None:
        number(duration, 'audio_duration')
    key = 'units' if kind == 'timed_units' else 'lines'
    items = payload.get(key)
    if not isinstance(items, list):
        raise ValueError(f'{key} must be a list')
    previous = 0.0
    previous_assigned = 0.0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'{key}[{index}] must be an object')
        if not isinstance(item.get('metadata', {}), dict):
            raise ValueError('metadata must be an object')
        for field in ('coverage',):
            if field in item.get('metadata', {}):
                number(item['metadata'][field], field, maximum=1)
        for field in ('matched_start_time', 'matched_end_time'):
            if field in item.get('metadata', {}):
                number(item['metadata'][field], field)
        if kind == 'timed_units':
            for key in ('symbol', 'normalized_symbol', 'unit_type', 'language'):
                if not isinstance(item.get(key), str):
                    raise ValueError(f'Unit {key} must be a string')
            start, end = _times(item, f'units[{index}]')
            if start < previous:
                raise ValueError('Timed units must be in chronological order')
            previous = start
            if item.get('confidence') is not None:
                number(item['confidence'], 'confidence', maximum=1)
        else:
            if type(item.get('line_index')) is not int or item['line_index'] != index:
                raise ValueError('line_index must be contiguous and match list order')
            if not isinstance(item.get('raw_text'), str):
                raise ValueError('raw_text must be a string')
            if kind == 'alignment_result':
                start = number(item.get('start_time'), 'start_time')
                assigned = number(item.get('assigned_time'), 'assigned_time')
                if not math.isclose(assigned, start, rel_tol=0, abs_tol=1e-9):
                    raise ValueError('assigned_time must match start_time')
                if assigned < previous_assigned:
                    raise ValueError('Alignment assigned_time values must be chronological')
                previous_assigned = assigned
                if start < previous:
                    raise ValueError('Alignment lines must be chronological')
                previous = start
                if item.get('end_time') is not None and number(item['end_time'], 'end_time') < start:
                    raise ValueError('Line ends before it starts')
                number(item.get('confidence', 0), 'confidence', maximum=1)
            units = item.get('units' if kind == 'parsed_lyrics' else 'aligned_units', [])
            if not isinstance(units, list):
                raise ValueError('Line units must be a list')
            previous_span_start = 0
            previous_unit_start = None
            for i, unit in enumerate(units):
                if not isinstance(unit, dict):
                    raise ValueError('Unit must be an object')
                if type(unit.get('unit_index_in_line')) is not int or unit['unit_index_in_line'] != i:
                    raise ValueError('unit_index_in_line must match list order')
                if kind == 'parsed_lyrics':
                    for key in ('symbol', 'normalized_symbol', 'unit_type', 'language'):
                        if not isinstance(unit.get(key), str):
                            raise ValueError(f'Unit {key} must be a string')
                    if unit.get('line_index', index) != index:
                        raise ValueError('Unit line_index does not match its line')
                    span = unit.get('source_text_span')
                    if span is not None and (not isinstance(span, (list, tuple)) or len(span) != 2 or
                                            any(type(v) is not int for v in span) or not 0 <= span[0] < span[1] <= len(item['raw_text'])):
                        raise ValueError('Invalid source_text_span')
                    if span is not None:
                        if span[0] < previous_span_start:
                            raise ValueError('Source spans must be in original text order')
                        previous_span_start = span[0]
                else:
                    unit_start, unit_end = _times(unit, f'aligned_units[{i}]')
                    if previous_unit_start is not None and unit_start < previous_unit_start - 1e-9:
                        raise ValueError('Aligned units must be chronological')
                    previous_unit_start = unit_start
                    if unit_start < start - 1e-9 or (item.get('end_time') is not None and unit_end > item['end_time'] + 1e-9):
                        raise ValueError('Aligned unit lies outside its line interval')
                    if not isinstance(unit.get('metadata', {}), dict):
                        raise ValueError('Aligned unit metadata must be an object')
                    number(unit.get('confidence', 0), 'unit confidence', maximum=1)
    if payload.get('overall_confidence') is not None:
        number(payload['overall_confidence'], 'overall_confidence', maximum=1)


def validate_alignment_inputs(transcription, parsed):
    validate_payload(transcription.to_dict(), 'timed_units')
    validate_payload(parsed.to_dict(), 'parsed_lyrics')
    if transcription.language != parsed.language:
        raise ValueError('Transcription and lyrics languages must match')
    if any(unit.unit_type != parsed.unit_type for unit in transcription.units):
        raise ValueError('Transcription and lyrics unit types must match')
    if any(unit.unit_type != parsed.unit_type for line in parsed.lines for unit in line.units):
        raise ValueError('Parsed lyric unit types are inconsistent')
