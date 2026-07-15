from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.feed_forward import FeedForward


def test_feed_forward_shape() -> None:
    config = ModelConfig(
        embedding_dim=32,
        feed_forward_dim=64,
    )

    layer = FeedForward(config)

    x = torch.randn(
        2,
        10,
        32,
    )

    output = layer(x)

    assert output.shape == x.shape