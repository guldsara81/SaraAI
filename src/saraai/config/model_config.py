"""
SaraAI Model Configuration

All language model settings are defined here.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """
    Configuration for a SaraAI language model.
    """

    # General
    model_name: str = "SaraAI-0.1"

    # Vocabulary
    vocabulary_size: int = 0

    # Model architecture
    embedding_dim: int = 128
    max_sequence_length: int = 128
    num_layers: int = 4
    num_heads: int = 4
    feed_forward_dim: int = 512

    # Regularization
    dropout: float = 0.1
    bias: bool = False

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4

    # Randomness
    seed: int = 42