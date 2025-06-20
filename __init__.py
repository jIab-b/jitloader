"""
jitloader
=========

A tiny, work-in-progress package that provides:

    • SafetensorLoader  – streams individual tensors from disk.
    • InferenceScheduler – orchestrates layer-by-layer execution of Flux.

Both classes are skeletal. Refer to in-file TODO markers in
`jitloader/scheduler.py` for guidance on completing the implementation.
"""

from .scheduler import SafetensorLoader, InferenceScheduler

__all__ = [
    "SafetensorLoader",
    "InferenceScheduler",
]