"""
a2a_utils/response.py — Unified A2A response helpers

Use the SDK's new_agent_text_message() to send results,
following the official a2a-samples/helloworld convention.
"""

import json

from a2a.utils import new_agent_text_message


def build_success(data: dict) -> object:
    """Wrap a result dict into an A2A text message event."""
    return new_agent_text_message(json.dumps(data))


def build_error(message: str) -> object:
    """Wrap an error string into an A2A text message event."""
    return new_agent_text_message(json.dumps({"error": message}))
