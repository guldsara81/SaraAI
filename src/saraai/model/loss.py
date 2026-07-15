"""
SaraAI Loss Functions
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    """
    Cross-entropy loss for next-token prediction.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the language modeling loss.

        Args:
            logits:
                Tensor of shape
                (batch_size, sequence_length, vocabulary_size)

            targets:
                Tensor of shape
                (batch_size, sequence_length)

        Returns:
            Scalar loss tensor.
        """

        vocabulary_size = logits.size(-1)

        logits = logits.view(
            -1,
            vocabulary_size,
        )

        targets = targets.view(-1)

        return self.loss(
            logits,
            targets,
        )