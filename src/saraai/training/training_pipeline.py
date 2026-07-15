"""
SaraAI Training Pipeline
"""

from __future__ import annotations

from torch.optim import Adam

from saraai.checkpoint.checkpoint_manager import CheckpointManager
from saraai.config.model_config import ModelConfig
from saraai.datasets.text_dataset import TextDataset
from saraai.model.data_loader import LanguageDataLoader
from saraai.model.language_model import LanguageModel
from saraai.model.trainer import Trainer
from saraai.model.training_step import TrainingStep
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

        self.optimizer = None

        self.training_step: TrainingStep | None = None

        self.trainer: Trainer | None = None

        self.checkpoint_manager = CheckpointManager()

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

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.training_step = TrainingStep(
            self.model,
            self.optimizer,
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

    def save(
        self,
        path: str,
        epoch: int = 0,
        loss: float = 0.0,
    ) -> None:
        """
        Save the current training checkpoint.
        """

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer has not been created."
            )

        self.checkpoint_manager.save(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            vocabulary=self.vocabulary,
            config=self.config,
            epoch=epoch,
            loss=loss,
        )
        