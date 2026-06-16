"""Core infrastructure helpers shared by QQQ research entry points."""

from .paths import ProjectPaths, resolve_output_path
from .run_context import RunContext

__all__ = ["ProjectPaths", "RunContext", "resolve_output_path"]
