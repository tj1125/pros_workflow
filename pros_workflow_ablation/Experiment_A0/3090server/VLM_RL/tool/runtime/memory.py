from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_cuda_memory() -> None:
    """Best-effort CUDA memory cleanup for request-scoped inference."""
    gc.collect()

    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("torch.cuda.empty_cache() failed: %s", exc)

    try:
        torch.cuda.ipc_collect()
    except Exception as exc:
        logger.debug("torch.cuda.ipc_collect() failed: %s", exc)
