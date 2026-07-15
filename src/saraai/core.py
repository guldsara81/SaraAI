"""
SaraAI Core

High-level public API for the SaraAI framework.
"""

from __future__ import annotations

from saraai.config.model_config import ModelConfig
from saraai.training.training_pipeline import TrainingPipeline


class SaraAI:
    """
    Main entry point for the SaraAI framework.
    """

    def __init__(
        self,
        name: str = "SaraAI",
        config: ModelConfig | None = None,
    ):
        self.name = name
        self.version = "0.1.0"

        if config is None:
            config = ModelConfig()

        self.config = config

        self.pipeline = TrainingPipeline(
            self.config,
        )

    def info(self) -> str:
        """
        Return framework information.
        """

        return (
            f"{self.name} "
            f"(version {self.version})"
        )

    def train(
        self,
        text: str,
    ) -> float:
        """
        Train SaraAI on a text string.

        Returns:
            Training loss.
        """

        self.pipeline.load_text(text)

        self.pipeline.build_model()

        return self.pipeline.train()

    def chat(
        self,
        message: str,
    ) -> str:
        """
        Placeholder until text generation is implemented.
        """

        return (
            "Chat generation has not been "
            "implemented yet.\n\n"
            f"You wrote:\n{message}"
        )

    def save(
        self,
        path: str,
    ) -> None:
        """
        Save the current model checkpoint to disk.
        """

        self.pipeline.save(path)
