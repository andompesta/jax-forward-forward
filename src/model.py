from typing import Any, Callable, Sequence
from jax import numpy as jnp
import jax
import optax
from flax import linen as nn, core
from flax.training.train_state import TrainState


class FFConvLayer(nn.Module):
    out_channels: int
    kernel_size: Sequence[int]
    stride: int
    eps: float = 1e-6
    kernel_init: Callable = nn.initializers.uniform(0.02)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.dense = nn.Conv(
            self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def forward(self, x: jnp.DeviceArray) -> jnp.DeviceArray:
        x_norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + self.eps

        # keep orientation only
        x = x / x_norm
        x = self.dense(x)
        x = nn.relu(x)
        return x

    def __call__(self, x_pos: jnp.DeviceArray, x_neg: jnp.DeviceArray):
        x_pos = self.forward(x_pos)
        x_neg = self.forward(x_neg)

        goodness_pos = jnp.power(x_pos, 2).sum(axis=-1)
        goodness_neg = jnp.power(x_neg, 2).sum(axis=-1)

        return x_pos, x_neg, goodness_pos, goodness_neg


class FFNet(nn.Module):
    layers: tuple[str] = ("ff_1", "ff_2", "ff_3", "classification")


    def setup(self):
        # Submodule names are derived by the attributes you assign to. In this
        # case, "dense1" and "dense2". This follows the logic in PyTorch.
        self.ff_1 = FFConvLayer(
            out_channels=128,
            kernel_size=(10, 10),
            stride=6,
        )
        self.ff_2 = FFConvLayer(
            out_channels=220,
            kernel_size=(3, 3),
            stride=1,
        )
        self.ff_3 = FFConvLayer(
            out_channels=512,
            kernel_size=(2, 2),
            stride=1,
        )
        self.classification = nn.Dense(10)

    def __call__(
        self,
        x_pos: jnp.DeviceArray,
        x_neg: jnp.DeviceArray,
    ):
        x_pos, x_neg, goodness_pos_1, goodness_neg_1 = self.ff_1(
            x_pos,
            x_neg,
        )
        x_pos, x_neg, goodness_pos_2, goodness_neg_2 = self.ff_2(
            x_pos,
            x_neg,
        )

        x_pos, x_neg, goodness_pos_3, goodness_neg_3 = self.ff_3(
            x_pos,
            x_neg,
        )

        logits = self.classification(x_pos)
        return logits, goodness_pos_1, goodness_neg_1, goodness_pos_2, goodness_neg_2, goodness_pos_3, goodness_neg_3

# TODO: create single train_step function that uses multiple optimizers
# TODO: user optax multi_transform

def create_training_states(
    rng,
    learning_rate: float = 3e-4,
) -> Sequence[TrainState]:
    net = FFNet()
    init_state = net.init(
        rng,
        jnp.ones([1, 28, 28, 1]),
        jnp.zeros([1, 28, 28, 1]),
    )
    params = init_state["params"]
    layers_params = [params[layer] for layer in net.layers]
    return [
        TrainState.create(
            apply_fn=net.__getattribute__(l).apply,
            params=p,
            tx=optax.adam(learning_rate),
        )
        for (l, p) in zip(net.layers, layers_params)
    ]


def compute_ff_metrics(logits, labels):
    loss = optax.sigmoid_binary_cross_entropy(
        logits=logits,
        labels=labels,
    ).mean()
    # accuracy = jnp.sum(logits > 0.5 == labels)
    metrics = {
        "loss": loss,
        # 'accuracy': accuracy,
    }
    return metrics


@jax.jit
def ff_train_step(
    state: TrainState,
    batch,
):
    def make_ff_loss_fn(
        apply_fn: Callable,
        threshold: float,
    ):
        def loss_fn(params: core.FrozenDict[str, Any]):
            x_pos, x_neg, goodness_pos, goodness_neg = apply_fn(
                params,
                batch["pos"],
                batch["neg"],
            )

            pos_logits = goodness_pos - threshold
            neg_logits = goodness_neg - threshold
            logits = jnp.concatenate((pos_logits, neg_logits), axis=0)
            labels = jnp.concatenate(
                (jnp.ones_like(pos_logits), jnp.zeros_like(neg_logits)), axis=0
            )

            loss = optax.sigmoid_binary_cross_entropy(
                logits=logits,
                labels=labels,
            ).mean()

            return loss, (x_pos, x_neg, logits, labels)

        return loss_fn

    loss_fn = make_ff_loss_fn(state.apply_fn, threshold=0.2)

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, (x_pos, x_neg, logits, labels) = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_ff_metrics(logits=logits, labels=labels)
    return (x_pos, x_neg), state, metrics


@jax.jit
def softmax_train_step(
    state: TrainState,
    batch,
):
    def make_loss_fn(
        apply_fn: Callable,
        num_classes: int = 10,
    ):
        def loss_fn(params: core.FrozenDict[str, Any]):
            logits = apply_fn(
                params,
                batch["pos"],
            )

            labels = jax.nn.one_hot(
                batch["labels"],
                num_classes=num_classes,
            )

            loss = optax.softmax_cross_entropy(
                logits=logits,
                labels=labels,
            ).mean()

            return loss, (logits, labels)

        return loss_fn

    loss_fn = make_loss_fn(
        state.apply_fn,
    )

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, (logits, labels) = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=labels)
    return state, metrics


def compute_metrics(logits, labels):
    loss = optax.softmax_cross_entropy(
        logits=logits,
        labels=labels,
    ).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics
