"""
commander/prompts.py — Prompt templates for the chat front-end.

The VLM reasoning system prompt has been removed (fixed-flow ablation, no VLM).
Only the related-object selection prompts (used by the input/chat flow) remain.
"""

# ---------------------------------------------------------------------------
# Related object selection: choose object ids from objects.yaml only
# ---------------------------------------------------------------------------

RELATED_OBJECT_SELECTION_SYSTEM_PROMPT = (
    "Select all related graspable objects from the provided labels only. "
    "Return 1-based indices ordered by label relevance. Do not invent categories."
)

RELATED_OBJECT_SELECTION_HUMAN_TEMPLATE = """Objects from config:
{listing}

Human request: {task_text}

Return related_object_indices ordered by relevance."""

