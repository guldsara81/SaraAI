"""
SaraAI Layer Normalization
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization used throughout the Transformer.
    """

    def __init__(
        self,
        embedding_dim: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.bias = nn.Parameter(
            torch.zeros(embedding_dim)
        )

        self.epsilon = epsilon

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x:
                Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of identical shape.
        """

        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        variance = x.var(
            dim=-1,
            unbiased=False,
            keepdim=True,
        )

        normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.epsilon
        )

        return (
            self.weight * normalized
            + self.bias
        )