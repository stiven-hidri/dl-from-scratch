"""Cross-check MLP's forward/backward against an equivalent torch computation."""

import random

import pytest
import torch
from micrograd.nn import MLP

ACTIVATIONS = {"relu": torch.relu, "tanh": torch.tanh}


def test_mlp_matches_torch_equivalent():
    random.seed(42)
    net = MLP(2, [3, 2], activation="relu", output_activation="tanh")
    x = [1.0, -0.5]
    labels = [1.0, -1.0]

    pred = net(x)
    loss = net.mse_loss(pred=pred, labels=labels)
    net.zero_grad()
    loss.backward()

    # Rebuild the exact same computation in torch, reusing net's own weights.
    h = torch.tensor(x, dtype=torch.float64)
    torch_params = []
    for layer in net.layers:
        act = ACTIVATIONS[layer.neurons[0].activation]
        w = torch.tensor(
            [[wi.data for wi in n.w] for n in layer.neurons],
            dtype=torch.float64,
            requires_grad=True,
        )
        b = torch.tensor(
            [n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True
        )
        torch_params.append((w, b))
        h = act(w @ h + b)

    t_labels = torch.tensor(labels, dtype=torch.float64)
    t_loss = ((h - t_labels) ** 2).mean()
    t_loss.backward()

    assert loss.data == pytest.approx(t_loss.item())
    for layer, (w, b) in zip(net.layers, torch_params):
        assert w.grad is not None
        assert b.grad is not None
        for ni, neuron in enumerate(layer.neurons):
            for wi, weight in enumerate(neuron.w):
                assert weight.grad == pytest.approx(w.grad[ni, wi].item())
            assert neuron.b.grad == pytest.approx(b.grad[ni].item())
