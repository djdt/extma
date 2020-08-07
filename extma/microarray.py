import numpy as np

from scipy import ndimage as ndi
from skimage import filters, feature, segmentation

from pew.lib.calc import normalise

from extma.core import Core
from extma.lib import cluster_idx, enlarge_border, remove_border

from typing import Callable, List


def normalised_total_counts(
    image: np.ndarray,
    names: List[str] = None,
    out_min: float = 0.0,
    out_max: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    if names is None:
        names = image.dtype.names

    ntic = np.zeros(image.shape, dtype=np.float64)
    for name in names:
        if name in image.dtype.names:
            x = image[name]
            if clip:
                x = np.clip(x, 0.0, np.percentile(x, 95))
            try:
                ntic += normalise(x, out_min, out_max)
            except ValueError:
                pass

    return ntic / len(names)


def remove_objects(x: np.ndarray, size: int) -> np.ndarray:
    x = x.copy()
    objects, _ = ndi.label(x, ndi.generate_binary_structure(x.ndim, 1))
    object_sizes = np.bincount(objects.ravel())
    objects_mask = np.array(object_sizes < size)[objects]
    x[objects_mask] = 0
    return x


# def individual_multiotsu_threshold_mask(
#     x: np.ndarray, labels: np.ndarray
# ) -> np.ndarray:
#     mask = np.zeros(x.shape, dtype=bool)
#     idx = np.unique(labels)
#     for i in idx[idx > 0]:
#         label = labels == i
#         mask[label] = x[label] > filters.threshold_multiotsu(x[label])[0]
#     return mask


class MicroArray(object):
    APLHA_LABELS = [s for s in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    NUMERIC_LABELS = [str(i) for i in range(1, 50)]

    def __init__(
        self,
        data: np.ndarray,
        core_size: int = 50,
        threshold_method: str = "local",
        threshold_names: List[str] = None,
        threshold_value: float = None,
        background_value: float = None,
        # masks: List[str] = None,
    ):
        self.data = data
        self.size = core_size

        if threshold_names is None:
            threshold_names = self.data.dtype.names

        self.normal = normalised_total_counts(self.data, names=threshold_names)
        self.labels = self._segment_cores(
            self.normal, self.size, threshold_method, threshold_value, background_value
        )

#         if masks is not None:
#             mask = individual_multiotsu_threshold_mask(
#                 normalised_total_counts(self.data, names=masks), self.labels
#             )
#             self.labels[~mask] = 0

        self.cores = self._generate_cores()

    @property
    def label_idx(self) -> np.ndarray:
        return np.unique(self.labels)

    @property
    def coreslist(self) -> List[Core]:
        return self.cores[np.nonzero(self.cores)]

    def core_text(self, core: Core, swap_xy: bool = False) -> str:
        y, x = np.argwhere(self.cores == core)[0]
        if swap_xy:
            x, y = y, x
        return MicroArray.APLHA_LABELS[x] + MicroArray.NUMERIC_LABELS[y]

    def core_op(self, core: Core, op: Callable, *args) -> float:
        v = tuple(
            op(self.data[core.image][name], *args) for name in self.data.dtype.names
        )
        return np.array(v, dtype=self.data.dtype)

    def _generate_cores(self) -> np.ndarray:
        idx = self.label_idx
        idx = idx[idx > 0]

        corelist = np.empty(idx.size, dtype=object)
        for i in idx:
            corelist[i - 1] = Core(i, self.labels)

        # Sort the cores by position
        centers = np.vstack([core.center for core in corelist])
        x_cluster = cluster_idx(centers[:, 0], self.size / 2.0)
        y_cluster = cluster_idx(centers[:, 1], self.size / 2.0)

        cores = np.empty((np.amax(x_cluster) + 1, np.amax(y_cluster) + 1), dtype=object)
        cores[x_cluster, y_cluster] = corelist

        return cores

    def _segment_cores(
        self,
        x: np.ndarray,
        size: int,
        method: str,
        method_value: float = None,
        min_value: float = None,
    ) -> np.ndarray:
        mask = np.zeros(np.array(x.shape) + (2 * size), dtype=bool)
        if method == "local":
            block = size + 1 if size % 2 == 0 else size
            thresh = filters.threshold_local(x, block, mode="wrap", param=size)
        elif method == "otsu":
            thresh = filters.threshold_otsu(x, size)
        elif method == "percentile":
            thresh = np.percentile(x, method_value)
        else:
            raise ValueError("Invalid method.")

        mask = enlarge_border(x >= thresh, size)

        # Remove single pixels and filaments
        mask = ndi.binary_opening(mask)

        # Close holes
        mask = ndi.binary_fill_holes(mask)

        # Smooth
        mask = ndi.binary_closing(mask, structure=np.ones((2, 2)))

        # Shrink selection to avoid background
        mask = ndi.binary_erosion(mask)

        # Remove any small objects: from skimage remove_small_objects
        mask = remove_objects(mask, int(size ** 2 * 0.25))

        mask = remove_border(mask, size)

        mask = self._separate_cores(mask, size)

        return ndi.label(mask)[0]

    def _separate_cores(self, mask: np.ndarray, size: int) -> np.ndarray:
        labels, num_labels = ndi.label(mask)
        lrg_obj = np.zeros_like(mask, dtype=bool)
        for label in range(1, num_labels):
            if np.count_nonzero(labels == label) > size ** 2 * 1.5:
                lrg_obj[labels == label] = True

        dist = ndi.distance_transform_edt(lrg_obj)

        peak_idx = feature.peak_local_max(
            dist, labels=lrg_obj, min_distance=size, exclude_border=0,
        )
        peak_mask = np.zeros_like(labels, dtype=bool)
        peak_mask[peak_idx[:, 0], peak_idx[:, 1]] = True
        peak_mask = ndi.binary_dilation(peak_mask, structure=np.ones((3, 3)))

        ws = segmentation.watershed(
            -dist,
            ndi.label(peak_mask)[0],
            mask=lrg_obj,
            connectivity=2,
            watershed_line=True,
        )
        mask[lrg_obj] = ws[lrg_obj]
        return mask
