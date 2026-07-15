"""
SaraAI Text Generator
"""

from __future__ import annotations

import torch

from saraai.model.language_model import LanguageModel


class TextGenerator:
    """
    Generates text using a trained language model.
    """

    def __init__(
        self,
        model: LanguageModel,
    ):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate new tokens using greedy decoding.
        """

        self.model.eval()

        tokens = input_ids.clone()

        for _ in range(max_new_tokens):

            logits = self.model(tokens)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1,
            )

        return tokens
