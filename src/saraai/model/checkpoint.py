"""
SaraAI Checkpoint Manager
"""

from __future__ import annotations

import torch

from saraai.config.model_config import ModelConfig
from saraai.model.language_model import LanguageModel


class CheckpointManager:
    """
    Save and load SaraAI models.
    """

    @staticmethod
    def save(
        model: LanguageModel,
        config: ModelConfig,
        path: str,
    ) -> None:

        torch.save(
            {
                "config": config,
                "state_dict": model.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(
        path: str,
    ) -> tuple[LanguageModel, ModelConfig]:

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        config = checkpoint["config"]

        model = LanguageModel(config)

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        model.eval()

        return model, config