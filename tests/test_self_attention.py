from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.self_attention import SelfAttention


def test_self_attention_shape() -> None:
    config = ModelConfig(
        embedding_dim=32,
        max_sequence_length=128,
    )

    layer = SelfAttention(config)

    x = torch.randn(
        2,
        8,
        32,
    )

    output = layer(x)

    assert output.shape == x.shape


def test_self_attention_has_mask() -> None:
    config = ModelConfig()

    layer = SelfAttention(config)

    assert layer.mask is not None