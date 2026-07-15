from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.generator import TextGenerator
from saraai.model.language_model import LanguageModel


def test_generator_output_length() -> None:
    config = ModelConfig(
        vocabulary_size=100,
        embedding_dim=32,
        feed_forward_dim=64,
        num_layers=2,
        max_sequence_length=32,
    )

    model = LanguageModel(config)

    generator = TextGenerator(model)

    input_ids = torch.randint(
        0,
        config.vocabulary_size,
        (1, 5),
    )

    output = generator.generate(
        input_ids=input_ids,
        max_new_tokens=3,
    )

    assert output.shape == (1, 8)