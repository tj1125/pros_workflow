"""Zero-dependency .env loader for the 3090 A2A services.

The three server entrypoints call :func:`load_env` at startup so a single
`3090server/pros_workflow/.env` is the source of truth for the GPU host and its
derived service URLs. Values already present in ``os.environ`` are kept, so the
legacy ``EXTERNAL_IP=<gpu-host> python -m ...`` command-line prefix still wins.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# tool/runtime/env.py -> parents[2] == 3090server/pros_workflow (SERVER_ROOT)
SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = SERVER_ROOT / ".env"

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand(value: str, parsed: dict[str, str]) -> str:
    """Expand ${VAR}/$VAR using os.environ first, then values parsed so far."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, parsed.get(name, ""))

    return _VAR_PATTERN.sub(repl, value)


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into ``os.environ``.

    Args:
        path: .env path; defaults to ``SERVER_ROOT/.env``.
        override: when False (default) existing environment variables are kept,
            so a command-line ``EXTERNAL_IP=...`` prefix takes precedence.

    Returns:
        The parsed key/value mapping (after ``${VAR}`` expansion).
    """
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    parsed: dict[str, str] = {}
    if not env_path.exists():
        return parsed

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = _expand(value, parsed)
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return parsed
