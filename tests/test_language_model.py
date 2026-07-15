from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.language_model import LanguageModel


def test_language_model_output_shape() -> None:
    config = ModelConfig(
        vocabulary_size=100,
        embedding_dim=32,
        feed_forward_dim=64,
        num_layers=2,
        max_sequence_length=32,
    )

    model = LanguageModel(config)

    token_ids = torch.randint(
        0,
        config.vocabulary_size,
        (2, 10),
    )

    logits = model(token_ids)

    assert logits.shape == (
        2,
        10,
        config.vocabulary_size,
    )