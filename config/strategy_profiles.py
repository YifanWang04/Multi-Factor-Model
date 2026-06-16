"""Compatibility wrapper for strategy profile configuration.

The authoritative profile configuration lives in ``qqq_config.strategy_profiles``.
This module remains only so legacy imports of ``config.strategy_profiles`` keep
working without maintaining a second, divergent copy.
"""

from qqq_config.strategy_profiles import *  # noqa: F401,F403
