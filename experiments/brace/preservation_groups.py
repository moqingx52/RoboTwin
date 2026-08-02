"""Shared BRACE sample-source and preservation-group label constants."""

from __future__ import annotations

# episode_source / sample_source: data provenance for SFT loss budgeting
SOURCE_EXPERT = 0
SOURCE_ROLLOUT = 1
SOURCE_PREFIX = 2

# episode_preservation_group / sample_preservation_group: capability groups for anchor
PRESERVATION_NONE = 0
PRESERVATION_BASE_SOLVED = 1
PRESERVATION_BOUNDARY = 2
PRESERVATION_HARD_MONITOR = 3

PRESERVATION_GROUP_NAMES = {
    PRESERVATION_NONE: "none",
    PRESERVATION_BASE_SOLVED: "base_solved",
    PRESERVATION_BOUNDARY: "boundary",
    PRESERVATION_HARD_MONITOR: "hard_monitor",
}

ANCHOR_GRADIENT_GROUPS = frozenset({PRESERVATION_BASE_SOLVED, PRESERVATION_BOUNDARY})
MONITOR_ONLY_GROUPS = frozenset({PRESERVATION_HARD_MONITOR})
