"""
SaraAI Training Script
"""

from __future__ import annotations

from saraai.config.model_config import ModelConfig
from saraai.training.training_pipeline import TrainingPipeline


def main() -> None:
    """
    Entry point for model training.
    """

    config = ModelConfig()

    pipeline = TrainingPipeline(config)

    text = (
        "Hello SaraAI! "
        "This is the first training dataset. "
        "SaraAI will learn language one token at a time."
    )

    pipeline.load_text(text)

    pipeline.build_model()

    loss = pipeline.train()

    print(f"Training loss: {loss:.6f}")


if __name__ == "__main__":
    main()