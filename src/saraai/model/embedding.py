"""
SaraAI Embedding Layer
"""

from __future__ import annotations

import torch
import torch.nn as nn

from saraai.config.model_config import ModelConfig


class EmbeddingLayer(nn.Module):
    """
    Converts token IDs into trainable embedding vectors.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=config.embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: Tensor of shape
                (batch_size, sequence_length)

        Returns:
            Tensor of shape
                (batch_size, sequence_length, embedding_dim)
        """

        return self.embedding(token_ids)