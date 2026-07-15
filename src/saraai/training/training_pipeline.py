"""
SaraAI Training Pipeline
"""

from __future__ import annotations

from torch.optim import Adam

from saraai.config.model_config import ModelConfig
from saraai.datasets.text_dataset import TextDataset
from saraai.model.data_loader import LanguageDataLoader
from saraai.model.language_model import LanguageModel
from saraai.model.training_step import TrainingStep
from saraai.model.trainer import Trainer
from saraai.tokenizer.tokenizer import Tokenizer
from saraai.tokenizer.vocabulary import Vocabulary


class TrainingPipeline:
    """
    Coordinates the complete language model training pipeline.
    """

    def __init__(
        self,
        config: ModelConfig,
    ):
        self.config = config

        self.vocabulary = Vocabulary()

        self.tokenizer = Tokenizer(
            self.vocabulary,
        )

        self.dataset: TextDataset | None = None

        self.dataloader: LanguageDataLoader | None = None

        self.model: LanguageModel | None = None

        self.training_step: TrainingStep | None = None

        self.trainer: Trainer | None = None

    def load_text(
        self,
        text: str,
    ) -> None:
        """
        Build the vocabulary and create the training dataset.
        """

        self.tokenizer.build_vocabulary(
            [text]
        )

        self.config.vocabulary_size = (
            self.tokenizer.vocabulary_size
        )

        self.dataset = TextDataset(
            tokenizer=self.tokenizer,
            sequence_length=self.config.max_sequence_length,
        )

        self.dataset.load_text(text)

    def build_model(
        self,
    ) -> None:
        """
        Create the language model and training components.
        """

        self.model = LanguageModel(
            self.config,
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.training_step = TrainingStep(
            self.model,
            optimizer,
        )

        self.trainer = Trainer(
            self.training_step,
        )

    def train(
        self,
    ) -> float:
        """
        Train the language model for one epoch.

        Returns:
            Average training loss.
        """

        if self.dataset is None:
            raise RuntimeError(
                "No dataset loaded."
            )

        if self.trainer is None:
            raise RuntimeError(
                "Model has not been built."
            )

        self.dataloader = LanguageDataLoader(
            dataset=self.dataset,
            config=self.config,
        )

        return self.trainer.train_epoch(
            self.dataloader,
        )