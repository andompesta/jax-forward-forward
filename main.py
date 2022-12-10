from src.dataset import NumpyLoader, get_mnist_dataset, ToTensor, Normalize
from src.utils import show_image
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torchvision.transforms import Compose

import numpy as np
import jax
from flax import linen as nn


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

    for img, l in train_dataset:
        show_image(img)
        break
    
    test_dataset = get_mnist_dataset(
        dataset_path.as_posix(),
        train=False,
        transform=img_transforms,
    )

    