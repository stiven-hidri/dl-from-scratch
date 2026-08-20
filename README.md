# dl-from-scratch

A personal collection of deep learning concepts implemented from scratch, for learning purposes. Each module lives in its own package under `src/` and re-implements a core idea without relying on a framework to do the heavy lifting.

## Modules

### `micrograd`

A tiny scalar-valued autograd engine and a small neural net library built on top of it, following the spirit of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). `Value` wraps a number and records the operations that produced it in a computation graph; `backward()` walks that graph in reverse topological order to compute gradients via backpropagation. `nn.py` builds `Neuron` / `Layer` / `MLP` on top of `Value`.

More modules will be added here over time as new concepts are worked through.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync
```

## Running

```sh
uv run dev
```

## Development

```sh
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type-check
uv run pytest            # test
```

Tests cross-check `Value`'s and `MLP`'s forward/backward against `torch`'s autograd on the same computations, rather than asserting hardcoded numbers — this is what actually caught the sign/aliasing bugs found while building `micrograd`.

## License

MIT
