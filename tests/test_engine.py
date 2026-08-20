"""Cross-check Value's forward/backward against torch's autograd."""

import pytest
import torch

from micrograd.engine import Value


def t(x: float) -> torch.Tensor:
    """A float64 scalar tensor requiring grad, matching Value's float64 precision."""
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def grad(tensor: torch.Tensor) -> float:
    """tensor.grad is typed as Tensor | None until backward() has run."""
    assert tensor.grad is not None
    return tensor.grad.item()


def test_add():
    a, b = Value(2.0), Value(-3.0)
    out = a + b
    out.backward()

    ta, tb = t(2.0), t(-3.0)
    (ta + tb).backward()

    assert out.data == pytest.approx(ta.item() + tb.item())
    assert a.grad == pytest.approx(grad(ta))
    assert b.grad == pytest.approx(grad(tb))


def test_sub():
    a, b = Value(2.0), Value(-3.0)
    out = a - b
    out.backward()

    ta, tb = t(2.0), t(-3.0)
    (ta - tb).backward()

    assert out.data == pytest.approx(2.0 - -3.0)
    assert a.grad == pytest.approx(grad(ta))
    assert b.grad == pytest.approx(grad(tb))


def test_mul():
    a, b = Value(2.0), Value(-3.0)
    out = a * b
    out.backward()

    ta, tb = t(2.0), t(-3.0)
    (ta * tb).backward()

    assert out.data == pytest.approx(ta.item() * tb.item())
    assert a.grad == pytest.approx(grad(ta))
    assert b.grad == pytest.approx(grad(tb))


def test_pow():
    a = Value(3.0)
    out = a**3
    out.backward()

    ta = t(3.0)
    tout = ta**3
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    assert a.grad == pytest.approx(grad(ta))


def test_truediv():
    a, b = Value(6.0), Value(-2.0)
    out = a / b
    out.backward()

    ta, tb = t(6.0), t(-2.0)
    (ta / tb).backward()

    assert out.data == pytest.approx(ta.item() / tb.item())
    assert a.grad == pytest.approx(grad(ta))
    assert b.grad == pytest.approx(grad(tb))


def test_neg():
    a = Value(4.0)
    out = -a
    out.backward()

    ta = t(4.0)
    (-ta).backward()

    assert out.data == pytest.approx(-ta.item())
    assert a.grad == pytest.approx(grad(ta))


@pytest.mark.parametrize("x", [-1.5, 0.7, 3.0])
def test_tanh(x):
    a = Value(x)
    out = a.tanh()
    out.backward()

    ta = t(x)
    tout = torch.tanh(ta)
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    assert a.grad == pytest.approx(grad(ta))


@pytest.mark.parametrize("x", [-2.3, 0.0, 1.8])
def test_relu(x):
    a = Value(x)
    out = a.relu()
    out.backward()

    ta = t(x)
    torch.relu(ta).backward()

    assert out.data == pytest.approx(max(0.0, x))
    assert a.grad == pytest.approx(grad(ta))


def test_composite_expression():
    """The classic a, b, c, f expression chain from Karpathy's micrograd walkthrough."""
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    e = a * b
    d = e + c
    loss = d * f
    loss.backward()

    ta, tb, tc, tf = t(2.0), t(-3.0), t(10.0), t(-2.0)
    tloss = (ta * tb + tc) * tf
    tloss.backward()

    assert loss.data == pytest.approx(tloss.item())
    for mg, tt in [(a, ta), (b, tb), (c, tc), (f, tf)]:
        assert mg.grad == pytest.approx(grad(tt))


def test_composite_expression_with_activations():
    """A harder chain mixing relu, tanh and a squared-error term, as used in mse_loss."""
    x1, w1, x2, w2, b = Value(1.5), Value(0.8), Value(2.0), Value(0.4), Value(0.3)
    pre = x1 * w1 + x2 * w2 + b
    out = pre.relu().tanh()
    label = Value(1.0)
    loss = (out - label) ** 2
    loss.backward()

    tx1, tw1, tx2, tw2, tb = t(1.5), t(0.8), t(2.0), t(0.4), t(0.3)
    tlabel = torch.tensor(1.0, dtype=torch.float64)
    tpre = tx1 * tw1 + tx2 * tw2 + tb
    tloss = (torch.tanh(torch.relu(tpre)) - tlabel) ** 2
    tloss.backward()

    assert loss.data == pytest.approx(tloss.item())
    for mg, tt in [(x1, tx1), (w1, tw1), (x2, tx2), (w2, tw2), (b, tb)]:
        assert mg.grad == pytest.approx(grad(tt))
