"""
fastloess: High-performance LOESS Smoothing for Python.

A high-performance LOESS (Locally Weighted Scatterplot Smoothing) implementation
with parallel execution via Rayon and NumPy integration. Built on top of the
fastLoess Rust crate.
"""

from .__version__ import __version__

from ._core import (
    LoessResult,
    Diagnostics,
    Loess,
    OnlineLoess,
    StreamingLoess,
)

__all__ = [
    "LoessResult",
    "Diagnostics",
    "Loess",
    "OnlineLoess",
    "StreamingLoess",
    "__version__",
]
