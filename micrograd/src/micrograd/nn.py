import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin: int, activation: str = "relu"):
        self.w = [Value(data=random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(data=random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x: list) -> Value:
        out = sum((x_i * w_i for x_i, w_i in zip(x, self.w)), self.b)
        out.activation_name = self.activation
        return out.activation()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, activation: str = "relu"):
        self.neurons = [Neuron(nin=nin, activation=activation) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(
        self,
        nin: int,
        layers_dims: list[int],
        activation: str = "relu",
        output_activation: str = "tanh",
    ):
        sz = [nin] + layers_dims
        n_layers = len(layers_dims)
        self.layers = [
            Layer(
                nin=sz[i],
                nout=sz[i + 1],
                activation=activation if i < n_layers - 1 else output_activation,
            )
            for i in range(n_layers)
        ]

    def __call__(self, x) -> list[Value]:
        for l in self.layers:
            x = l(x)
        return x

    def mse_loss(self, pred: list[Value], labels):
        assert len(pred) == len(labels)
        return sum((pi - li) ** 2 for pi, li in zip(pred, labels)) / len(pred)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
