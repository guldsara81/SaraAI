"""
SaraAI Checkpoint Manager
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Optimizer

from saraai.checkpoint.checkpoint import Checkpoint
from saraai.config.model_config import ModelConfig
from saraai.model.language_model import LanguageModel
from saraai.tokenizer.vocabulary import Vocabulary


class CheckpointManager:
    """
    Saves and loads SaraAI checkpoints.
    """

    def save(
        self,
        *,
        path: str | Path,
        model: LanguageModel,
        optimizer: Optimizer,
        vocabulary: Vocabulary,
        config: ModelConfig,
        epoch: int,
        loss: float,
    ) -> None:
        """
        Save a complete training checkpoint.
        """

        checkpoint = Checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            vocabulary=vocabulary.word_to_id,
            model_config=asdict(config),
            epoch=epoch,
            loss=loss,
        )

        torch.save(
            asdict(checkpoint),
            Path(path),
        )

    def load(
        self,
        path: str | Path,
    ) -> Checkpoint:
        """
        Load a checkpoint from disk.
        """

        data = torch.load(
            Path(path),
            map_location="cpu",
        )

        return Checkpoint(**data)