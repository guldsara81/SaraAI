"""
SaraAI Text Dataset

Converts text into training sequences for language modeling.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from saraai.tokenizer.tokenizer import Tokenizer


class TextDataset(Dataset):
    """
    Dataset for next-token prediction.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        sequence_length: int,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.tokens: list[int] = []

    def load_text(
        self,
        text: str,
    ) -> None:
        """
        Add text to the dataset.
        """

        token_ids = self.tokenizer.encode(text)

        self.tokens.extend(token_ids)

    def load_file(
        self,
        filename: str,
    ) -> None:
        """
        Load a UTF-8 text file.
        """

        text = Path(filename).read_text(
            encoding="utf-8",
        )

        self.load_text(text)

    def __len__(
        self,
    ) -> int:

        return max(
            0,
            len(self.tokens)
            - self.sequence_length,
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        input_tokens = self.tokens[
            index : index + self.sequence_length
        ]

        target_tokens = self.tokens[
            index + 1 : index + self.sequence_length + 1
        ]

        return (
            torch.tensor(
                input_tokens,
                dtype=torch.long,
            ),
            torch.tensor(
                target_tokens,
                dtype=torch.long,
            ),
        )