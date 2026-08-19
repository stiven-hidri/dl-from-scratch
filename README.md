# dl-from-scratch

A personal collection of deep learning concepts implemented from scratch, for learning purposes. Everything currently lives in the `dl_from_scratch` package under `src/`; as more concepts get added they'll grow out into their own modules within it.

## What's inside

A tiny scalar-valued autograd engine and a small neural net library built on top of it (`engine.py`, `nn.py`), following the spirit of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). `Value` wraps a number and records the operations that produced it in a computation graph; `backward()` walks that graph in reverse topological order to compute gradients via backpropagation. `nn.py` builds `Neuron` / `Layer` / `MLP` on top of `Value`.

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
```

## License

MIT
