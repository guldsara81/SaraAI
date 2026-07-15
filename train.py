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

    # Small window + a longer sample text so the dataset produces
    # training sequences. With max_sequence_length=128 you would need
    # 128+ tokens of text before a single sample exists.
    config.max_sequence_length = 16
    config.batch_size = 8

    pipeline = TrainingPipeline(config)

    text = (
        "Hello SaraAI. This is the first training dataset. "
        "SaraAI will learn language one token at a time. "
        "The model reads text, predicts the next token, and "
        "adjusts its weights so the prediction improves. "
        "Every epoch the training loss should get a little smaller. "
        "This small run proves the whole pipeline works end to end, "
        "from tokenizer to transformer to optimizer step."
    )

    pipeline.load_text(text)

    pipeline.build_model()

    print(f"Vocabulary size : {config.vocabulary_size}")
    print(f"Training samples: {len(pipeline.dataset)}")

    for epoch in range(1, 6):
        loss = pipeline.train()
        print(f"Epoch {epoch}  training loss: {loss:.6f}")


if __name__ == "__main__":
    main()
