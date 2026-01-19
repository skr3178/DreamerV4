"""
Imagination module for DreamerV4 Phase 3.

Generates imagined trajectories from the frozen world model
for policy optimization with PMPO.
"""

from .rollout import ImaginationRollout

__all__ = ["ImaginationRollout"]
