import numpy as np


def cluster_idx(x: np.ndarray, diff: float) -> np.ndarray:
    i = np.argsort(x)
    idx = np.split(i, np.where(np.diff(x[i]) > diff)[0] + 1)
    order = np.empty(x.shape, dtype=int)
    for i, j in enumerate(idx):
        order[j] = i
    return order


def enlarge_border(x: np.ndarray, size: int) -> np.ndarray:
    shape = np.array(x.shape) + (2 * size)
    border = np.full(shape, np.median(x), dtype=x.dtype)
    border[size:-size, size:-size] = x
    return border


def normalise(x: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Normalise an array.

    Args:
        x: Array
        vmin: New minimum
        vmax: New maxmimum
    """
    x = (x - x.min()) / x.max()
    x *= vmax - vmin
    x += vmin
    return x


def remove_border(x: np.ndarray, size: int) -> np.ndarray:
    return x[size:-size, size:-size]
