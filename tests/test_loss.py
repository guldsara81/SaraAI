from __future__ import annotations

import torch

from saraai.model.loss import LanguageModelLoss


def test_loss_returns_scalar() -> None:
    loss_function = LanguageModelLoss()

    logits = torch.randn(
        2,
        5,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 5),
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert loss.ndim == 0