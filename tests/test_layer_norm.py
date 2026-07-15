import torch

from saraai.model.layer_norm import LayerNorm


def test_layer_norm_shape():

    layer = LayerNorm(64)

    x = torch.randn(2, 10, 64)

    output = layer(x)

    assert output.shape == x.shape


def test_layer_norm_changes_values():

    layer = LayerNorm(64)

    x = torch.randn(2, 10, 64)

    output = layer(x)

    assert not torch.equal(x, output)