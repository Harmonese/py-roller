"""Expose unsupported text and approximate pronunciation routes to consumers."""
import re
import unicodedata


def route_warnings(text, summary):
    warnings = []
    normalized = unicodedata.normalize('NFKC', text)
    covered = [False] * len(normalized)
    cursor = 0
    for segment in summary.get('segments', []):
        route = segment.get('route')
        value = segment.get('text', '')
        normalized_value = unicodedata.normalize('NFKC', value)
        position = normalized.find(normalized_value, cursor) if normalized_value else -1
        if position >= 0:
            covered[position:position + len(normalized_value)] = [True] * len(normalized_value)
            cursor = position + len(normalized_value)
        if segment.get('assumed_language'):
            warnings.append({'code': 'assumed_english', 'text': value, 'language': 'en'})
        if not segment.get('unit_count') and any(c.isalnum() for c in value):
            warnings.append({'code': 'unsupported_text', 'text': value, 'language': segment.get('language', 'und')})
        if route in {'lexicon', 'arpabet_proxy', 'grapheme_letter_name', 'grapheme_fallback', 'acronym_letter_name', 'mixed_foreign'}:
            warnings.append({'code': 'approximate_pronunciation', 'text': value, 'route': route})
    # Segmenters may omit standalone digits or unsupported alphabets entirely.
    # Account for these omissions, not only segments with empty pronunciation.
    missing = ''.join(char if not covered[i] and char.isalnum() else ' ' for i, char in enumerate(normalized))
    warnings.extend({'code': 'unsupported_text', 'text': match.group(), 'language': 'und'}
                    for match in re.finditer(r'\S+', missing))
    return warnings


def english_warnings(text, phones):
    segments = {}
    for phone in phones:
        span = phone.get('source_text_span')
        if span is None:
            continue
        span = tuple(span)
        segment = segments.setdefault(span, {'text': text[span[0]:span[1]], 'unit_count': 0, 'language': 'en', 'route': 'dictionary'})
        segment['unit_count'] += 1
        if str(phone.get('symbol', '')).startswith('GR_'):
            segment['route'] = 'grapheme_fallback'
    return route_warnings(text, {'segments': list(segments.values())})
