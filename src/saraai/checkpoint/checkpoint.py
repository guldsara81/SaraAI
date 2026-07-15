"""
SaraAI Checkpoint

Defines the data stored in a model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Checkpoint:
    """
    Represents a complete training checkpoint.
    """

    model_state: dict[str, Any]

    optimizer_state: dict[str, Any]

    vocabulary: dict[str, int]

    model_config: dict[str, Any]

    epoch: int

    loss: float

    version: str = "0.1"