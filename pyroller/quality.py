"""Quality is independent of successful execution or file publication."""
from __future__ import annotations


class AlignmentQualityError(ValueError):
    code = 'alignment_quality_failed'


def attach_timing_provenance(alignment, transcription):
    for line in alignment.lines:
        for unit in line.aligned_units:
            index = unit.source_audio_unit_index
            if index is None:
                source = 'unresolved' if unit.end_time <= unit.start_time else 'interpolated'
                trace = {'timing_basis': 'lyric_gap_interpolation'}
                unit.metadata['match_status'] = 'unmatched'
            else:
                original = transcription.units[index]
                trace = {key: value for key, value in original.metadata.items()
                         if key.startswith('timing_') or key in {'source_segment_index', 'source_word_index', 'engine_span_id'}}
                semantics = str(transcription.metadata.get('unit_timing_semantics') or 'unknown')
                if trace.get('timing_is_interpolated') or 'interpolated' in semantics:
                    source = 'interpolated'
                elif trace.get('timing_is_acoustic') or semantics == 'model_native_segment':
                    source = 'acoustic'
                else:
                    source = 'unknown'
                trace['unit_timing_semantics'] = semantics
                unit.metadata['match_status'] = 'matched'
            unit.metadata['timing_source'] = source
            unit.metadata['timing_provenance'] = trace


def evaluate_quality(alignment, config=None):
    from pyroller.config_contracts import QUALITY_OPTIONS, validate_rules
    validate_rules(config or {}, QUALITY_OPTIONS)
    config = {**{key: item["default"] for key, item in QUALITY_OPTIONS.items()}, **(config or {})}
    lines = [line for line in alignment.lines if line.raw_text.strip() and not line.metadata.get('is_structural')]
    count = len(lines)
    matched = sum(line.confidence > 0 and line.method != 'interpolate' for line in lines)
    interpolated = sum(line.method == 'interpolate' or line.confidence <= 0 for line in lines)
    coverage = sum(float(line.metadata.get('coverage', 1.0 if line.confidence > 0 else 0.0)) for line in lines) / count if count else 0.0
    longest = 0.0
    run_start = None
    for i, line in enumerate(lines):
        if line.confidence <= 0 or line.method == 'interpolate':
            if run_start is None:
                run_start = line.start_time
            end = lines[i+1].start_time if i+1 < count else (line.end_time or line.start_time)
            longest = max(longest, end - run_start)
        else:
            run_start = None
    reasons = []
    if not count:
        reasons.append('no_lyric_lines')
    if coverage < config.get('min_coverage', 0.8):
        reasons.append('low_coverage')
    ratio = interpolated / count if count else 1.0
    if ratio > config.get('max_interpolated_ratio', 0.2):
        reasons.append('excessive_interpolation')
    if longest > config.get('max_unmatched_seconds', 10.0):
        reasons.append('long_unmatched_region')
    warnings = alignment.metadata.get('language_warnings', [])
    if warnings:
        reasons.append('language_approximation_or_unsupported')
    timing_counts = dict.fromkeys(('acoustic', 'interpolated', 'unresolved', 'unknown'), 0)
    timing_diagnostics = []
    timing_anomalies = 0
    missing_timing_lines = [line.line_index for line in lines if not line.aligned_units]
    for line in lines:
        previous_end = line.start_time
        for unit in line.aligned_units:
            source = unit.metadata.get('timing_source', 'unknown')
            if source not in timing_counts:
                source = 'unknown'
            timing_counts[source] += 1
            issues = []
            if unit.start_time < previous_end - 1e-9:
                issues.append('overlapping_unit_times')
            if unit.end_time <= unit.start_time:
                issues.append('zero_duration')
            previous_end = max(previous_end, unit.end_time)
            timing_anomalies += bool(issues)
            if source != 'acoustic' or issues:
                timing_diagnostics.append({'line_index': line.line_index, 'unit_index_in_line': unit.unit_index_in_line,
                                           'source_text_span': unit.metadata.get('source_text_span'),
                                           'timing_source': source, 'match_status': unit.metadata.get('match_status', 'unknown'),
                                           'issues': issues})
    unit_count = sum(timing_counts.values())
    timing_review = bool(not unit_count or timing_counts['acoustic'] != unit_count or timing_anomalies or missing_timing_lines)
    if config['timing_policy'] == 'unit' and timing_review:
        reasons.append('unit_timing_requires_review')
    quality = {'status': 'degraded' if reasons else 'ok', 'needs_review': bool(reasons),
               'matched_line_ratio': matched / count if count else 0.0,
               'coverage': coverage, 'interpolated_line_ratio': ratio,
               'longest_unmatched_seconds': longest, 'reasons': reasons,
               'language_warnings': warnings,
               'timing_policy': config['timing_policy'], 'timing_needs_review': timing_review,
               'timing_source_counts': timing_counts,
               'timing_anomaly_unit_count': timing_anomalies, 'missing_unit_timing_lines': missing_timing_lines,
               'interpolated_unit_ratio': timing_counts['interpolated'] / unit_count if unit_count else 0.0,
               'acoustic_unit_ratio': timing_counts['acoustic'] / unit_count if unit_count else 0.0,
               'unresolved_unit_ratio': timing_counts['unresolved'] / unit_count if unit_count else 0.0,
               'unknown_timing_unit_ratio': timing_counts['unknown'] / unit_count if unit_count else 1.0,
               'unit_timing_diagnostics': timing_diagnostics}
    alignment.report['quality'] = quality
    return quality


def enforce_quality(alignment, config=None):
    quality = evaluate_quality(alignment, config)
    if (config or {}).get('mode', 'report') == 'strict' and quality['needs_review']:
        raise AlignmentQualityError('Alignment requires review: ' + ', '.join(quality['reasons']))
    return quality
