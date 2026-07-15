"""
SaraAI Transformer Block
"""

from __future__ import annotations

import torch
import torch.nn as nn

from saraai.config.model_config import ModelConfig
from saraai.model.feed_forward import FeedForward
from saraai.model.layer_norm import LayerNorm
from saraai.model.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    """
    A single GPT-style Transformer block using
    Pre-Layer Normalization.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_norm = LayerNorm(
            config.embedding_dim,
        )

        self.attention = SelfAttention(config)

        self.feed_forward_norm = LayerNorm(
            config.embedding_dim,
        )

        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute one Transformer block.

        Args:
            x:
                Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of identical shape.
        """

        # Self-attention with residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward network with residual connection
        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x