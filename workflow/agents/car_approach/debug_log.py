"""Small CLI debug logging helper for the car approach pipeline."""

from __future__ import annotations

from typing import Any


def debug_stage(component: str, stage_message: str, **fields: Any) -> None:
    """Print one human-readable car_approach stage line to CLI."""
    details = " ".join(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    suffix = f" | {details}" if details else ""
    print(f"[car_approach][{component}] {stage_message}{suffix}", flush=True)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        compact_items = list(value.items())[:4]
        compact = ", ".join(f"{key}: {_format_value(item)}" for key, item in compact_items)
        if len(value) > 4:
            compact += ", ..."
        return "{" + compact + "}"
    return str(value)
