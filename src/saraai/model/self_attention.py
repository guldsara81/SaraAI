"""
SaraAI Self Attention

Single-head causal self-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from saraai.config.model_config import ModelConfig


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention layer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding_dim = config.embedding_dim

        self.query = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )

        self.key = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )

        self.value = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.max_sequence_length,
                    config.max_sequence_length,
                    dtype=torch.bool,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal self-attention.

        Args:
            x:
                Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of identical shape.
        """

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.embedding_dim)

        sequence_length = x.size(1)

        mask = self.mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
        )

        weights = self.dropout(weights)

        return weights @ value
