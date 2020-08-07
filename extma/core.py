import numpy as np

from typing import Tuple


class Core(object):
    def __init__(
        self, idx: int, labels: np.ndarray,
    ):
        self._labels = labels
        self.idx = idx

    @property
    def image(self) -> np.ndarray:
        return self._labels == self.idx

    @property
    def mesh(self) -> Tuple[np.ndarray]:
        image = self.image
        return np.ix_(np.any(image, axis=0), np.any(image, axis=1))

    @property
    def bounds(self) -> np.ndarray:
        mesh = self.mesh
        return np.vstack(([np.amin(m) for m in mesh], [np.amax(m) for m in mesh]))

    @property
    def box_size(self) -> int:
        bounds = self.bounds
        return np.prod(bounds[1] - bounds[0])

    @property
    def width(self) -> int:
        bounds = self.bounds
        return np.diff(bounds[:, 0])

    @property
    def height(self) -> int:
        bounds = self.bounds
        return np.diff(bounds[:, 1])

    @property
    def center(self) -> np.ndarray:
        return np.sum(self.bounds, axis=0) / 2.0

    @property
    def size(self) -> int:
        return np.count_nonzero(self.image)

    @property
    def roundness(self) -> float:
        rnd = self.size / (np.pi * self.box_size / 4.0)
        return rnd if rnd <= 1.0 else 2.0 - rnd  # range 0 - 1
