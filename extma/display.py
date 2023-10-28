from typing import List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent
from matplotlib.image import AxesImage
from matplotlib.patheffects import withStroke
from matplotlib.text import Text
from skimage.color import label2rgb

from extma import MicroArray


class SegmentDisplay(object):
    def __init__(
        self,
        ax: Axes,
        data: np.ndarray,
        size: int,
        methods: List[Tuple[str, Optional[float]]],
        names: List[str],
        swap_xy_label: bool = False,
    ):
        self.ax = ax

        self.coretexts: List[Text] = []
        self.image: Optional[AxesImage] = None

        self.data = data
        self.size = size

        self.methods = methods
        self.names = names

        self.midx = 0
        self.iidx = 0
        self.ntic = False

        self.swap_xy_label = swap_xy_label

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
                self.ax.text(
                    core.center[0],
                    core.center[1],
                    tma.core_text(core, swap_xy=self.swap_xy_label),
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
            if self.iidx > len(self.names) - 1:
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
