from saraai.config.model_config import ModelConfig


def test_default_config():

    config = ModelConfig()

    assert config.embedding_dimension == 128
    assert config.transformer_layers == 4
    assert config.attention_heads == 4