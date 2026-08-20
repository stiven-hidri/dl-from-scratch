import math

from graphviz import Digraph


class Value:
    def __init__(
        self,
        data: float,
        label: str = "data",
        grad: float = 0.0,
        activation_name: str = "relu",
        _children: tuple[Value, ...] = (),
        _op: str | None = None,
    ) -> None:
        self.data = data
        self.label = label
        self.grad = grad
        self.activation_name: str = activation_name
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"{self.data:.4f}"

    def __radd__(self, other):
        return self + other

    def __neg__(self):

        out = Value(
            data=-self.data,
            label=self.label,
            _children=(self,),
            _op="neg",
        )

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward

        return out

    def __rsub__(self, other: Value | float):
        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        return other + -self

    def __sub__(self, other: Value | float):
        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        return self + -other

    def __add__(self, other: Value | float) -> Value:

        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other: Value | float) -> Value:

        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rpow__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(float(other))
        return other**self

    def __pow__(self, other: Value | float):
        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        out = Value(data=self.data**other.data, _children=(self, other), _op="**")

        def _backward():
            self.grad += (other.data) * self.data ** (other.data - 1) * out.grad

        out._backward = _backward

        return out

    def __rtruediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(float(other))
        return other / self

    def __truediv__(self, other: Value | float):
        if isinstance(other, (float, int)):
            other = Value(data=float(other))

        return self * other ** (-1)

    def tanh(self) -> Value:
        tanh_res = math.tanh(self.data)
        out = Value(data=tanh_res, _children=(self,), label="tanh", _op="tanh")

        def _backward() -> None:
            self.grad += out.grad * (1.0 - tanh_res**2)

        out._backward = _backward

        return out

    def relu(self) -> Value:

        rel_res = max(0, self.data)

        out = Value(data=rel_res, _children=(self,), label="relu", _op="relu")

        def _backward() -> None:
            self.grad += out.grad * ((self.data > 0.0) * 1.0)

        out._backward = _backward

        return out

    def sigmoid(self):

        sigmoid_res = 1 / (1 + math.exp(-self.data))

        out = Value(data=sigmoid_res, label="sigmoid", _op="sigmoid")

        def _backward():
            self.grad = out.grad * (sigmoid_res * (1 - sigmoid_res))

        return out

    def _topological_sort(self) -> list[Value]:
        ts_nodes, visited = [], set()

        def dfs(node: Value = self) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    dfs(child)
                ts_nodes.append(node)

        dfs()

        return list(reversed(ts_nodes))

    def _trace(self, root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
        """Builds a set of all nodes and edges in the graph using DFS."""
        nodes, edges = set(), set()

        def build(v: Value):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))  # Edge from input to output
                    build(child)

        build(root)
        return nodes, edges

    def backward(self) -> None:
        self.grad = 1.0
        for node in self._topological_sort():
            node._backward()

    def draw_dot(self) -> Digraph:
        """Creates a Graphviz Digraph rendering node values and operations."""
        root = self

        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # Left to Right

        nodes, edges = self._trace(root)
        for n in nodes:
            uid = str(id(n))
            content = f"data {n.data:.4f} | grad {n.grad:.4f}"
            label_text = (
                f"{n.label} | {content}"
                if n.label
                else f"data {n.data:.4f} | grad {n.grad:.4f} | {content}"
            )
            dot.node(
                name=uid,
                label=f"{{ {label_text} }}",
                shape="record",
            )

            # If this node was created by an operation, create an intermediate op node
            if n._op:
                op_uid = uid + n._op
                dot.node(name=op_uid, label=n._op, shape="circle")
                dot.edge(op_uid, uid)

        for n1, n2 in edges:
            # Connect input node n1 to output operation node (n2 + op)
            target_uid = str(id(n2)) + n2._op if n2._op else str(id(n2))
            dot.edge(str(id(n1)), target_uid)

        return dot

    def activation(self) -> Value:
        if self.activation_name == "relu":
            return self.relu()
        elif self.activation_name == "tanh":
            return self.tanh()
        else:
            raise NotImplementedError(f"Invalid activation: {self.activation_name}")
