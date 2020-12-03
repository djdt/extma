import argparse
import sys

import numpy as np
from skimage.color import label2rgb

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.backend_bases import KeyEvent
from matplotlib.patheffects import withStroke

from pewlib import io

from extma import MicroArray

from typing import List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Parse, segment and extrac data from micro-arrays.",
    )

    parser.add_argument(
        "infile", help="Pew '.npz' archive.",
    )
    parser.add_argument("size", type=int, default=50, help="Size of the segments.")
    parser.add_argument(
        "--isotopes", nargs="+", help="Limit which isotopes are output."
    )
    args = parser.parse_args(sys.argv[1:])
    return args


class SegmentTracker(object):
    def __init__(
        self,
        ax: Axes,
        data: np.ndarray,
        size: int,
        methods: List[Tuple[str, Optional[float]]],
        names: List[str],
    ):
        self.ax = ax

        self.coretexts: List[Text] = []
        self.image: AxesImage = None

        self.data = data
        self.size = size

        self.methods = methods
        self.names = names

        self.midx = 0
        self.iidx = 0
        self.ntic = False

        self.init_ax()

    def init_ax(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.ax.text(
            0.05,
            0.05,
            (
                "'up' next method\n'down' previous method\n'left' previous isotope\n"
                "'right' next isotope\n't' use total counts"
            ),
            ha="left",
            va="top",
            color="white",
            fontsize=7,
            path_effects=[withStroke(linewidth=1.5, foreground="black")],
        )

    def get_tma(self) -> MicroArray:
        names = self.names if self.ntic else [self.names[self.iidx]]
        method, value = self.methods[self.midx]
        return MicroArray(
            self.data,
            self.size,
            threshold_names=names,
            threshold_method=method,
            threshold_value=value,
        )

    def draw_tma(self, tma: MicroArray) -> None:
        if self.image is not None:
            self.image.remove()

        self.image = self.ax.imshow(
            label2rgb(tma.labels, image=tma.normal, alpha=0.5, bg_label=0)
        )

        for text in self.coretexts:
            text.remove()
        self.coretexts.clear()

        for core in tma.coreslist:
            self.coretexts.append(
                axes.text(
                    core.center[0],
                    core.center[1],
                    tma.core_text(core),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=14,
                    path_effects=[withStroke(linewidth=1.5, foreground="black")],
                )
            )
        method = self.methods[self.midx]
        mname = method[0] if method[1] is None else f"{method[0]}[{method[1]}]"
        iname = "total counts" if self.ntic else self.names[self.iidx]
        self.ax.set_title(f"{mname} - {iname}")
        self.ax.figure.canvas.draw()

    def on_keypress(self, event: KeyEvent) -> None:
        if event.key == "up":
            self.midx += 1
            if self.midx > len(self.methods) - 1:
                self.midx = 0
        elif event.key == "down":
            self.midx -= 1
            if self.midx < 0:
                self.midx = len(self.methods) - 1
        elif event.key == "right":
            self.ntic = False
            self.iidx += 1
            if self.iidx > len(self.methods) - 1:
                self.iidx = 0
        elif event.key == "left":
            self.ntic = False
            self.iidx -= 1
            if self.iidx < 0:
                self.iidx = len(self.names) - 1
        elif event.key == "t":
            self.ntic = True
        else:
            return
        self.draw_tma(self.get_tma())


if __name__ == "__main__":
    args = parse_args()

    data = io.npz.load(args.infile)[0].get(calibrate=False)

    if args.isotopes is None:
        args.isotopes = data.dtype.names

    seg_methods = [
        ("local", None),
        ("otsu", None),
        ("percentile", 60.0),
        ("percentile", 65.0),
        ("percentile", 70.0),
        ("percentile", 75.0),
        ("percentile", 80.0),
    ]

    figsize = (6 * data.shape[1] / data.shape[0], 6)

    fig, axes = plt.subplots(figsize=figsize, dpi=100, frameon=False, tight_layout=True)

    seg = SegmentTracker(axes, data, args.size, seg_methods, args.isotopes)
    seg.draw_tma(seg.get_tma())

    fig.canvas.mpl_connect("key_press_event", seg.on_keypress)

    plt.show()
