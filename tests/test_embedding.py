from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.embedding import EmbeddingLayer


def test_embedding_shape() -> None:
    config = ModelConfig(
        vocabulary_size=100,
        embedding_dim=32,
    )

    layer = EmbeddingLayer(config)

    token_ids = torch.tensor([[1, 5, 10]])

    output = layer(token_ids)

    assert output.shape == (1, 3, 32)