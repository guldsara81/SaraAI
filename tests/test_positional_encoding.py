from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.positional_encoding import PositionalEncoding


def test_positional_encoding_shape() -> None:
    config = ModelConfig(
        embedding_dim=32,
        max_sequence_length=128,
    )

    layer = PositionalEncoding(config)

    embeddings = torch.zeros(
        (2, 10, 32)
    )

    output = layer(embeddings)

    assert output.shape == embeddings.shape


def test_positional_encoding_changes_values() -> None:
    config = ModelConfig(
        embedding_dim=32,
        max_sequence_length=128,
    )

    layer = PositionalEncoding(config)

    embeddings = torch.zeros(
        (1, 5, 32)
    )

    output = layer(embeddings)

    assert not torch.equal(
        embeddings,
        output,
    )