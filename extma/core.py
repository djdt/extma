import numpy as np

from typing import Tuple


class Core(object):
    def __init__(
        self,
        idx: int,
        labels: np.ndarray,
    ):
        self._labels = labels
        self.idx = idx

    @property
    def image(self) -> np.ndarray:
        return self._labels == self.idx

    @property
    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        image = self.image
        return np.ix_(np.any(image, axis=1), np.any(image, axis=0))

    @property
    def bounds(self) -> np.ndarray:
        ys, xs = self.mesh
        return np.stack(
            ([np.amin(xs), np.amax(xs)], [np.amin(ys), np.amax(ys)]), axis=0
        )

    @property
    def box_size(self) -> int:
        return self.width * self.height

    @property
    def width(self) -> int:
        return int(np.diff(self.bounds[0]))

    @property
    def height(self) -> int:
        return int(np.diff(self.bounds[1]))

    @property
    def center(self) -> np.ndarray:
        return np.sum(self.bounds, axis=1) / 2.0

    @property
    def size(self) -> int:
        return np.count_nonzero(self.image)

    @property
    def roundness(self) -> float:
        rnd = self.size / (np.pi * self.box_size / 4.0)
        return rnd if rnd <= 1.0 else 2.0 - rnd  # range 0 - 1
