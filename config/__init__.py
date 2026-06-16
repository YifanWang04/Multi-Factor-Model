"""Repository-level configuration package.

Compatibility note:
Some legacy single-factor scripts import ``SingleFactorConfig`` with
``from config import SingleFactorConfig``. Now that root ``config`` is a real
package, keep that import working while newer code imports strategy profiles
from ``config.strategy_profiles``.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "SingleFactorConfig":
        from analysis.single_factor.config import SingleFactorConfig

        return SingleFactorConfig
    raise AttributeError(f"module 'config' has no attribute {name!r}")


__all__ = ["SingleFactorConfig"]
