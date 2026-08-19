import random

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(data=random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(data=random.uniform(-1, 1))

    def __call__(self, x: list):
        activation = sum((x_i * w_i for x_i, w_i in zip(x, self.w)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin=nin) for _ in range(nout)]

    def __call__(self, x: list[Value]):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, layers_dims: list[int]):
        sz = [nin] + layers_dims
        self.layers = [
            Layer(nin=sz[i], nout=sz[i + 1]) for i, d in enumerate(layers_dims)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = [x] if isinstance(x, (Value, float)) else x
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
