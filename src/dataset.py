import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import jax.numpy as jnp
from typing import Callable, Any


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: data.Sampler = None,
        batch_sampler: data.BatchSampler = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn: Callable[[Any], None] = None
    ):
        super(
            self.__class__,
            self
        ).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class ToTensor(object):
    def __call__(self, pic) -> Any:
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]
                return pic.astype(np.float32)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = np.array(
            pic,
            mode_to_nptype.get(pic.mode, np.float32),
            copy=True
        )

        if pic.mode == "1":
            img = 255 * img
        img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
        return img


class Normalize(object):
    def __init__(
        self,
        mean: float,
        std: float,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.array) -> np.array:
        img = img / 255
        return (img - self.mean) / self.std


def get_mnist_dataset(
    path: str,
    train: bool,
    transform: Callable[[Any], Any] = None,
) -> data.Dataset:
    return MNIST(
        path,
        download=True,
        train=train,
        transform=transform
    )


def mnist_collate_fn(batch):
    pos, labels = zip(*batch)
    pos = np.stack(pos).astype(np.float32)

    shape = pos.shape
    img_size = shape[1] * shape[2] * shape[3]

    # get positive index
    idx = [(i * img_size) + l for i, l in enumerate(labels)]
    # get negative index
    neg_idx = [
        (i * img_size) + l for i, l in enumerate(
            np.random.randint(0, 10, size=len(idx))
        )
    ]

    # set all labels for positive cases to zero
    pos[:, 0, 0:10, :] = 0
    # flatten for add label infpor
    pos = pos.flatten()
    # create neg examples
    neg = np.copy(pos)
    # add correct labels information
    pos[idx] = 1
    # add random label informaton to negative cases
    neg[neg_idx] = 1
    # reshape
    pos = pos.reshape(shape)
    neg = neg.reshape(shape)
    labels = np.array(labels).astype(np.float32)

    return dict(
        pos=pos,
        neg=neg,
        labels=labels
    )


def get_mnist_dataloader(
    dataset: data.Dataset,
    shuffle: bool,
    batch_size: int,
):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mnist_collate_fn
    )
