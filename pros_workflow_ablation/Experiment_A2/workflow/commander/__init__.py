"""Commander package — Brain, Orchestrator, State, Logger."""

from __future__ import annotations

import os
from pathlib import Path

_WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _WORKFLOW_ROOT.parent
_PROJECT_ENV = _PROJECT_ROOT / ".env"
if not _PROJECT_ENV.exists():
    _PROJECT_ENV = _WORKFLOW_ROOT / ".env"


def _fallback_load_dotenv(path: Path, *, override: bool) -> bool:
    if not path.exists():
        return False

    loaded = False
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ[key] = value
        loaded = True
    return loaded


def load_project_env(*, override: bool = True) -> bool:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return _fallback_load_dotenv(_PROJECT_ENV, override=override)

    return bool(load_dotenv(dotenv_path=_PROJECT_ENV, override=override))


load_project_env()

__all__ = ["load_project_env"]
