from src.dataset import (
    get_mnist_dataset,
    get_mnist_dataloader,
    ToTensor,
    Normalize,
)
from src.utils import show_image
from src.model import FFNet, create_training_states
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

    parser.add_argument(
        "--learning_rate",
        default=3e-4,
        type=float
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

    train_dl = get_mnist_dataloader(
        train_dataset,
        False,
        10,
    )

    for batch in train_dl:
        show_image(batch["pos"][0])
        show_image(batch["neg"][0])
        print(batch["labels"][0])
        break

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    training_states = create_training_states(
        rng,
        args.learning_rate,
    )

    
    
    test_dataset = get_mnist_dataset(
        dataset_path.as_posix(),
        train=False,
        transform=img_transforms,
    )

    