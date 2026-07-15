"""
SaraAI Language Model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from saraai.config.model_config import ModelConfig
from saraai.model.embedding import EmbeddingLayer
from saraai.model.layer_norm import LayerNorm
from saraai.model.positional_encoding import PositionalEncoding
from saraai.model.transformer_block import TransformerBlock


class LanguageModel(nn.Module):
    """
    GPT-style language model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embedding = EmbeddingLayer(config)

        self.positional_encoding = PositionalEncoding(config)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = LayerNorm(
            config.embedding_dim,
        )

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.vocabulary_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass through the language model.
        """

        x = self.embedding(token_ids)

        x = self.positional_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.output_projection(x)

        return logits