import argparse
import sys

import numpy as np

from skimage.color import label2rgb

from pew import io
from pew.lib.colocal import pearsonr, li_icq

from extma import Core, MicroArray

from typing import Callable, List, TextIO


op_dict = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "count": np.size,
}
colocal_dict = {
    "pearson": pearsonr,
    "li": li_icq,
}


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        prog="exTMA",
        description="Parse, segment and extrac data from micro-arrays.",
    )
    # Input
    parser.add_argument(
        "infile", help="Pew '.npz' archive, may be calibrated.",
    )
    parser.add_argument("size", type=int, default=50, help="Size of the cores in pixels.")
    parser.add_argument(
        "--nocalibration", action="store_true", help="Don't use calibration."
    )
    # Filtering
    parser.add_argument(
        "--cores", type=str, help="Limit output to specifed cores. Format is A1:B2"
    )
    parser.add_argument(
        "--outnames", nargs="+", help="Limit which isotopes are output."
    )
    # parser.add_argument(
    #     "--outmasks", nargs="+", help="Isotopes used for individual core otsu masking."
    # )
    # Segmentating
    parser.add_argument(
        "--names", type=str, nargs="+", help="Isotopes used for creating mask during segmentation.",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="local",
        choices=["local", "otsu", "percentile"],
        help="Threshold method used for mask during segmentation.",
    )
    parser.add_argument(
        "--value",
        type=float,
        help="Value used for some thresholding methods. percentile=<%>.",
    )
    # Options
    parser.add_argument(
        "--rotate", type=int, choices=[90, 180, 270], help="Rotate the data, in degrees."
    )
    parser.add_argument(
        "--swapxy", action="store_true", help="Swap row and column labels.",
    )
    # Output
    parser.add_argument(
        "--draw", action="store_true", help="Use matplotlib to draw the tma."
    )
    parser.add_argument(
        "--print",
        nargs="?",
        const="info",
        choices=["info"] + list(op_dict.keys()) + list(colocal_dict.keys()),
        help="Print result of operation performed on segmented data.",
    )
    parser.add_argument(
        "--output", help="Output to a file instead of stdout.",
    )

    args = parser.parse_args(argv[1:])
    if args.print in colocal_dict.keys():
        if args.outnames is None or len(args.outnames) > 1:
            parser.error(
                "Colocalisation operations can only be perfromed on one isotope."
            )
    if args.threshold == "percentile":
        if args.value is None:
            parser.error("'value' required for threshold method 'percentile'.")
    if not args.draw and args.print is None:
        args.print = "info"

    return args


def core_text_to_position(text: str, swap_xy: bool = False) -> np.ndarray:
    if ":" in text:
        xy0, xy1 = text.split(":")
        y0, x0 = ord(xy0[0]) - 64 - 1, int(xy0[1:]) - 1
        y1, x1 = ord(xy1[0]) - 64 - 1, int(xy1[1:]) - 1
        xs, ys = np.s_[x0 : x1 + 1], np.s_[y0 : y1 + 1]
    else:
        y, x = ord(text[0]) - 64 - 1, int(text[1:]) - 1
        xs, ys = np.s_[x : x + 1], np.s_[y : y + 1]

    if swap_xy:
        xs, ys = ys, xs
    return xs, ys


def draw_tma(tma: MicroArray, output: str = None, swap_xy_label: bool = False) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    fig, axes = plt.subplots()
    axes.axis("off")

    axes.imshow(label2rgb(tma.labels, image=tma.normal, alpha=0.6, bg_label=0))

    for core in tma.coreslist:
        pos = core.center
        axes.text(
            pos[0],
            pos[1],
            tma.core_text(core, swap_xy=swap_xy_label),
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            path_effects=[withStroke(linewidth=1.5, foreground="black")],
        )

    plt.tight_layout()
    if output is not None:
        plt.save(output)
    else:
        plt.show()


def print_tma(
    tma: MicroArray,
    cores: List[Core],
    op: Callable,
    isotopes: List[str],
    output: TextIO = None,
) -> None:
    if output is None:
        output = sys.stdout

    names = np.array([tma.core_text(c) for c in cores], dtype=[("Name", str, 8)])
    values = np.hstack([tma.core_op(c, op) for c in cores])
    data = np.lib.recfunctions.merge_arrays(
        (names, values), flatten=True, usemask=False
    )
    header = ",".join(data.dtype.names)
    fmt = "%s" + ",%.10f" * len(values.dtype.names)
    np.savetxt(output, data, fmt=fmt, header=header, comments="")


def print_tma_colocal(
    tma: MicroArray,
    cores: List[Core],
    op: Callable,
    isotopes: List[str],
    output: TextIO = None,
) -> None:
    if output is None:
        output = sys.stdout

    names = np.array([tma.core_text(c) for c in cores], dtype=[("Name", str, 8)])
    base = isotopes[0]

    values = np.empty(len(cores), dtype=tma.data.dtype)
    for name in values.dtype.names:
        values[name] = [
            op(tma.data[core.image][base], tma.data[core.image][name]) for core in cores
        ]

    data = np.lib.recfunctions.merge_arrays(
        (names, values), flatten=True, usemask=False
    )
    header = ",".join([f"{base}::{name}" for name in data.dtype.names])
    fmt = "%s" + ",%.10f" * len(values.dtype.names)
    np.savetxt(output, data, fmt=fmt, header=header, comments="")


def print_tma_info(tma: MicroArray, cores: List[Core], output: TextIO = None,) -> None:
    if output is None:
        output = sys.stdout

    dtype = [
        ("Name", str, 8),
        ("Size", int),
        ("Width", int),
        ("Height", int),
        ("X", int),
        ("Y", int),
        ("Roundness", float),
    ]
    data = np.empty(len(cores), dtype=dtype)

    for i, core in enumerate(cores):
        data[i] = (
            tma.core_text(core),
            core.size,
            core.width,
            core.height,
            *core.center,
            core.roundness,
        )

    header = ",".join(data.dtype.names)
    fmt = ("%s", "%d", "%d", "%d", "%d", "%d", "%.2f")
    np.savetxt(output, data, delimiter=",", fmt=fmt, header=header, comments="")


def validate_names(names: List[str], valid_names: List[str]) -> None:
    for name in names:
        if name not in valid_names:
            raise ValueError(f"Invalid name '{name}'.")


def main():
    args = parse_args(sys.argv)

    laser = io.npz.load(args.infile)[0]
    data = laser.get(calibrate=not args.nocalibration)

    if args.outnames is None:
        args.outnames = data.dtype.names
    else:
        validate_names(args.outnames, data.dtype.names)

    if args.names is not None:
        validate_names(args.names, data.dtype.names)

    # if args.outmasks is not None:
    #     validate_names(args.outmasks, data.dtype.names)

    if args.rotate:
        data = np.rot90(data, k=4 - args.rotate // 90, axes=(0, 1))

    tma = MicroArray(
        data,
        args.size,
        threshold_names=args.names,
        threshold_method=args.threshold,
        threshold_value=args.value,
        # masks=args.outmasks,
    )

    cores = tma.cores
    if args.cores is not None:
        cores = cores[core_text_to_position(args.cores)]
    cores = cores[np.nonzero(cores)].ravel()

    if args.print:
        if args.print == "info":
            print_tma_info(tma, cores, args.output)
        elif args.print in op_dict.keys():
            op = op_dict[args.print]
            print_tma(tma, cores, op, args.outnames, args.output)
        elif args.print in colocal_dict.keys():
            op = colocal_dict[args.print]
            print_tma_colocal(tma, cores, op, args.outnames, args.output)
    if args.draw:
        draw_tma(tma, args.output, args.swapxy)


if __name__ == "__main__":
    main()
