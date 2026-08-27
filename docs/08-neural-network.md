# 8. Neurons, layers, and MLPs

> Implemented in [`nabla/nn/`](../nabla/nn/).
> Tested by [`tests/test_nn.py`](../tests/test_nn.py) (86 tests).

Here is the payoff. Everything from here is built out of `Value` and nothing
else — no new mathematics, no new backward rules. A neural network turns out to
be an *arrangement* of the arithmetic we already have.

---

## 1. The neuron

A neuron computes a weighted sum of its inputs, adds a bias, and applies a
nonlinearity:

$$y = \sigma\left(\sum_{i} w_i x_i + b\right)$$

```python
class Neuron(Module):
    def __init__(self, n_in, activation='tanh', ...):
        self.w = [Parameter(...) for _ in range(n_in)]
        self.b = Parameter(0.0)

    def forward(self, xs):
        z = sum(wi * xi for wi, xi in zip(self.w, xs)) + self.b
        return self.activation(z)
```

That is the whole thing. `wi * xi` builds a `Value` node; `sum` builds more;
`self.activation` builds one more. **The graph is constructed as a side effect
of the arithmetic**, and `backward()` will find its way through all of it
without the neuron knowing anything about derivatives.

A neuron with `n` inputs has `n + 1` parameters and contributes about `2n + 1`
nodes to the graph.

### Why the bias

Without `b`, every neuron's decision boundary passes through the origin. The
bias translates it, and it is why a neuron can represent "fire when the input
exceeds *this* threshold" rather than only "exceeds zero".

Biases initialise to **zero**, and that is safe even though zero-initialising
weights would be a disaster — see below.

## 2. The layer

A layer is a list of neurons, all fed the same inputs:

```python
class Layer(Module):
    def __init__(self, n_in, n_out, ...):
        self.neurons = [Neuron(n_in, ...) for _ in range(n_out)]

    def forward(self, xs):
        return [n(xs) for n in self.neurons]
```

`n_out` neurons, each with `n_in + 1` parameters. A `784 → 128` layer holds
100,480 parameters and — on the scalar engine — creates roughly 200,000 graph
nodes per sample.

That number is worth sitting with. It is the entire motivation for
[chapter 15](15-tensors.md).

## 3. The MLP

Stack layers, feed each one's output to the next:

```python
model = MLP(784, [128, 64], 10, activation='relu')
```

gives `784 → 128 → 64 → 10`, configurable rather than hard-coded. The output
layer is **linear by default** — no activation. Classifiers emit *logits* and
the softmax lives inside the loss, for numerical-stability reasons covered in
[chapter 10](10-losses.md).

## 4. `Module`: the one abstraction that earns its place

Every component inherits from `Module`, which does one important thing:
`__setattr__` notices when you assign a `Parameter` or another `Module` and
registers it. That makes `model.parameters()` work by recursive traversal, so
the optimiser never needs to know anything about network structure.

```python
model.parameters()      # every Parameter, recursively
model.num_parameters()  # 109,386
model.zero_grad()
model.state_dict()      # for checkpointing
model.gradient_stats()  # {'mean': ..., 'max': ..., 'zeros': ...}
model.summary()
```

The test that this abstraction is drawn in the right place: the tensor engine
([chapter 15](15-tensors.md)) reuses `Module`, `Optimizer`, `Trainer` and the
checkpoint code **completely unchanged**. Only the leaf type differs. If those
boundaries had been wrong, that reuse would have been impossible.

`Parameter` is simply a `Value` marked as learnable — the marker is what
`Module.parameters()` looks for, so `Module` needs no knowledge of which engine
is running.

## 5. Initialisation is not a detail

> Implemented in [`nabla/nn/init.py`](../nabla/nn/init.py).

**Zero-initialising weights breaks the network completely.** If every weight in
a layer is zero, every neuron computes the same output, receives the same
gradient, and takes the same update. They stay identical forever. A 128-unit
layer behaves as one unit, permanently. This is the *symmetry problem*, and
random initialisation exists to break it.

Biases can safely be zero, because each sits behind a different random weight
vector.

**But the scale matters too.** Consider a sum of `n` random terms: its variance
grows with `n`. Initialise a 784-input layer with weights of variance 1 and the
pre-activations have standard deviation ~28 — deep in tanh's saturated region
where the derivative is ~0, so almost no gradient flows and training stalls
before it starts.

The schemes correct for exactly this:

| scheme | variance | for |
|---|---|---|
| **He** | $2/n_{in}$ | ReLU — the 2 compensates for ReLU zeroing half its inputs |
| **Xavier** | $U(\pm\sqrt{6/(n_{in}+n_{out})})$ | tanh, sigmoid — balances forward and backward variance |
| **LeCun** | $1/n_{in}$ | linear |

The library picks automatically from the activation, because pairing them
wrongly is a subtle and common mistake.

## 6. The whole network is still just `Value`

Worth restating, because it is the point of the chapter. When you call

```python
loss = softmax_cross_entropy([model(x) for x in batch], labels)
loss.backward()
```

there is no special "neural network backward pass". `loss` is a `Value` like
any other. `backward()` topologically sorts a graph of ~328,000 nodes and walks
it, applying `mul`, `add`, `relu`, `exp` and `log` rules — the same six lines
from [chapter 5](05-backpropagation.md).

**The network is not a special case. It is a large expression.** That
realisation is most of what this project exists to produce.

## 7. Verified end to end

```python
model = MLP(3, [4, 4], 1, seed=0)
result = check_parameter_gradients(
    lambda: mse_loss([model(x) for x in xs], ys),
    model.parameters(),
)
assert result.passed          # every parameter, vs finite differences
```

Every parameter's gradient, through two hidden layers, an activation, and a
loss — checked against finite differences that share no code with the engine.

---

**Next:** [9. Activation functions](09-activations.md)
