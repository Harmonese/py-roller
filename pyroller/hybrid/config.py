from dataclasses import dataclass
from ..config import EngineConfig as RuntimeConfig


@dataclass(frozen=True)
class AlignmentSettings:
    alignment_mode: str = 'ordered'
    scoring_mode: str = 'weighted'
    stitching_mode: str = 'evidence'
    matching_mode: str = 'robust'
    timing_mode: str = 'legacy'
    profile_score_tolerance: float = 2.0
    profile_time_tolerance: float = .5
    candidate_limit_per_line_start: int = 4
    diagnostic_bin_seconds: float = 10.0
    overlap_time_tolerance: float = 2.0
    minimum_similarity: float = .72
    edit_fraction: float = .30
    edit_penalty: float = 2.0
    instance_cost: float = .5
    alternatives: int = 3
    max_line_seconds: float = 30.0
    max_word_gap_seconds: float = 6.0
    group_seconds: float = 32.0
    padding_seconds: float = 2.0
    anchor_tolerance: float = .5
    ordinary_onset_shift: float = 2.0
    unknown_cost: float = 6.0
    joint_stage: str = 'decision'
    joint_padding_seconds: float = 4.0
    joint_shift_seconds: tuple = (-4., -2., 0., 2., 4.)
    joint_time_cost_cap: float = 3.0
    joint_shift_cost: float = .5
    joint_margin: float = 2.0
    conditional_candidates: str = 'endpoints'
    posterior_mass: float = .9
    posterior_max_width: float = 1.0

    def __post_init__(self):
        if self.alignment_mode not in ('unordered', 'ordered'):
            raise ValueError('Invalid alignment mode')
        if self.scoring_mode not in ('original', 'consistent', 'weighted', 'power'):
            raise ValueError('Invalid scoring mode')
        if self.stitching_mode not in ('legacy','evidence'):
            raise ValueError('Invalid stitching mode')
        if self.stitching_mode=='evidence' and self.matching_mode!='robust':
            raise ValueError('Evidence stitching requires robust matching')
        if self.matching_mode not in ('legacy', 'robust') or self.timing_mode not in ('legacy', 'profile', 'joint', 'conditional', 'posterior'):
            raise ValueError('Invalid alignment mode')
        if not 0 < self.posterior_mass < 1 or not 0 < self.posterior_max_width <= 2:
            raise ValueError('Invalid posterior decision settings')
        if self.joint_stage not in ('structure','prior','decision'):
            raise ValueError('Invalid joint stage')
        if self.conditional_candidates not in ('sparse','dense','endpoints'):
            raise ValueError('Invalid conditional candidate policy')


@dataclass(frozen=True)
class EngineConfig(RuntimeConfig):
    scoring_mode: str = 'weighted'
    stitching_mode: str = 'evidence'
    matching_mode: str = 'robust'
    timing_mode: str = 'legacy'
    whisper_context: str = 'none'

    def __post_init__(self):
        super().__post_init__()
        if self.whisper_context not in ('none','lyrics'):
            raise ValueError('Invalid Whisper context mode')
        AlignmentSettings(matching_mode=self.matching_mode,timing_mode=self.timing_mode,stitching_mode=self.stitching_mode,
                          scoring_mode=self.scoring_mode)
