"""
SaraAI Training Step
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from saraai.model.language_model import LanguageModel
from saraai.model.loss import LanguageModelLoss


class TrainingStep:
    """
    Performs a single optimization step.
    """

    def __init__(
        self,
        model: LanguageModel,
        optimizer: optim.Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = LanguageModelLoss()

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Perform one training step.
        """

        self.model.train()

        self.optimizer.zero_grad()

        logits = self.model(inputs)

        loss = self.loss_function(
            logits,
            targets,
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()