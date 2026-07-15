"""
SaraAI Trainer
"""

from __future__ import annotations

from typing import Iterable

import torch

from saraai.model.training_step import TrainingStep


class Trainer:
    """
    Trains a language model.
    """

    def __init__(
        self,
        training_step: TrainingStep,
    ):
        self.training_step = training_step

    def train_epoch(
        self,
        dataloader: Iterable[
            tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> float:
        """
        Train the model for one epoch.
        """

        total_loss = 0.0
        batches = 0

        for inputs, targets in dataloader:
            loss = self.training_step(
                inputs,
                targets,
            )

            total_loss += loss
            batches += 1

        if batches == 0:
            return 0.0

        return total_loss / batches