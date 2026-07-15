"""
SaraAI Data Loader
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from saraai.config.model_config import ModelConfig


class LanguageDataLoader(DataLoader):
    """
    DataLoader for language model training.
    """

    def __init__(
        self,
        dataset,
        config: ModelConfig,
        shuffle: bool = True,
    ):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
        )