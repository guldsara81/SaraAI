from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.transformer_block import TransformerBlock


def test_transformer_block_shape() -> None:
    config = ModelConfig(
        embedding_dim=32,
        feed_forward_dim=64,
        max_sequence_length=128,
    )

    block = TransformerBlock(config)

    x = torch.randn(
        2,
        10,
        32,
    )

    output = block(x)

    assert output.shape == x.shape