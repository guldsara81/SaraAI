"""
SaraAI Feed Forward Network
"""

from __future__ import annotations

import torch
import torch.nn as nn

from saraai.config.model_config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in the Transformer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.feed_forward_dim,
                bias=config.bias,
            ),
            nn.GELU(),
            nn.Linear(
                config.feed_forward_dim,
                config.embedding_dim,
                bias=config.bias,
            ),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the feed-forward network.

        Args:
            x:
                Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of identical shape.
        """

        return self.network(x)