from saraai.config.model_config import ModelConfig


def test_default_config():

    config = ModelConfig()

    assert config.embedding_dim == 128
    assert config.num_layers == 4
    assert config.num_heads == 4
