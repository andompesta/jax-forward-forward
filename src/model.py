from typing import Any, Callable, Sequence
from flax import linen as nn
from jax import numpy as jnp
import optax
import flax

from flax import core
from flax import struct


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
        x_norm = jnp.linalg.norm(
            x,
            ord=2,
            axis=-1,
            keepdims=True
        ) + self.eps

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
   
    def setup(self):
        # Submodule names are derived by the attributes you assign to. In this
        # case, "dense1" and "dense2". This follows the logic in PyTorch.
        self.ff_1 = FFConvLayer(
            out_channels=128,
            kernel_size=(10, 10),
            stride=6
        )
        self.ff_2 = FFConvLayer(
            out_channels=220,
            kernel_size=(3, 3),
            stride=1
        )
        self.ff_3 = FFConvLayer(
            out_channels=512,
            kernel_size=(2, 2),
            stride=1
        )
        self.classification = nn.Dense(self.output_size)

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
        return logits



# def create_train_state(hidden_size, num_classes, rng, learning_rate, momentum):
#     """Creates initial `TrainState`."""
#     net = FFDenseNet(hidden_size, num_classes)
#     # initialize parameters by passing a template image
#     params = net.init(rng, jnp.ones([1, 28, 28, 1]))['params']
#     tx = optax.sgd(learning_rate, momentum)
#     return train_state.TrainState.create(
#         apply_fn=net.apply, params=params, tx=tx)


# def train_step(
#     state,
#     batch,
#     num_classes: int,
#     goodness_threshold: float,
# ):
#     # split in 3 differente training states
#     def cross_entropy_loss(
#         logits: jnp.DeviceArray,
#         labels: jnp.DeviceArray,
#     ) -> jnp.DeviceArray:
#         labels_one_hot = nn.one_hot(labels, num_classes=num_classes)
#         loss = optax.softmax_cross_entropy(
#             logits=logits,
#             labels=labels_one_hot,
#         ).mean()
#         return loss

#     def goodness_loss(
#         logits: jnp.DeviceArray,
#         labels: jnp.DeviceArray,
#     ) -> jnp.DeviceArray:
#         goodness_logits = logits - goodness_threshold
#         loss = optax.sigmoid_binary_cross_entropy(
#             logits=goodness_logits,
#             labels=labels
#         ).mean()
#         return loss

#     def loss_fn(params):
#         (
#             logits,
#             goodness_pos_1,
#             goodness_neg_1,
#             goodness_pos_2,
#             goodness_neg_2,
#         ) = FFDenseNet().apply(
#             {'params': params},
#             batch['pos'],
#             batch['neg'],
#         )
#         cross_loss = cross_entropy_loss(logits=logits, labels=batch['label'])
#     return loss, logits
