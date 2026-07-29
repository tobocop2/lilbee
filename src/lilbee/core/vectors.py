"""The embedding vector type shared by providers, retrieval, and the store."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
"""A single embedding, float32 to match the store's vector column."""
