from src.dataset import get_mnist_dataset, ToTensor, Normalize
from src.utils import show_image
from src.model import FFNet
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torchvision.transforms import Compose

import numpy as np
import flax
import jax
import optax


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        default=Path.cwd(),
        type=lambda x: Path(x)
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dataset_path = args.base_path.joinpath(
        "data",
        "mnist"
    )

    img_transforms = Compose([
        ToTensor(),
        Normalize(0.1307, 0.3081),
    ])

    train_dataset = get_mnist_dataset(
        dataset_path.as_posix(),
        train=True,
        transform=img_transforms,
    )

    net = FFNet()

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # initialize parameters by passing a template image
    init_state = net.init(rng, np.ones([1, 28, 28, 1]))
    params = init_state["params"]

    tx = optax.sgd(3e-5, 0.9)
    ts = flax.train_state.TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=tx
    )

    for img, l in train_dataset:
        show_image(img)
        
        break
    
    test_dataset = get_mnist_dataset(
        dataset_path.as_posix(),
        train=False,
        transform=img_transforms,
    )

    