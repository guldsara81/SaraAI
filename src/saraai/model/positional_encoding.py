"""
SaraAI Positional Encoding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from saraai.config.model_config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional information to token embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        position = torch.arange(
            config.max_sequence_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                config.embedding_dim,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / config.embedding_dim)
        )

        encoding = torch.zeros(
            config.max_sequence_length,
            config.embedding_dim,
        )

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            embeddings:
                Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of identical shape.
        """

        sequence_length = embeddings.size(1)

        return (
            embeddings
            + self.encoding[:, :sequence_length]
        )