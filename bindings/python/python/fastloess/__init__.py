"""
fastloess: High-performance LOESS Smoothing for Python.

A high-performance LOESS (Locally Estimated Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLoess Rust crate.
"""

from .__version__ import __version__
from ._core import (
    AddPointResult,
    Diagnostics,
    Loess,
    LoessResult,
    OnlineLoess,
    OnlineOutput,
    PredictOutput,
    StreamingLoess,
)

__all__ = [
    "AddPointResult",
    "Diagnostics",
    "Loess",
    "LoessResult",
    "OnlineLoess",
    "OnlineOutput",
    "PredictOutput",
    "StreamingLoess",
    "__version__",
]
