import sys
import os
import random
import warnings
import mrcfile
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from cycler import cycler
from coord_converter import process_conversion
import matplotlib  # we import pyplot later

MRC_PLOT_MARGIN = 64
GT_COLOR = "blue"
PICKER_COLORS = [
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#e41a1c",
    "#dede00",
]

matplotlib.rcParams["axes.prop_cycle"] = cycler(color=[GT_COLOR] + PICKER_COLORS)
from matplotlib import patches


# utils


def _log(msg, lvl=0, quiet=False):
    """
    Format and print message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if quiet=True)
    1: warning (print and continue execution)
    2: error (print and exit with code 1)
    """

    if lvl == 0 and quiet:
        return

    prefix = ""
    if lvl == 0:
        prefix = "INFO: "
    elif lvl == 1:
        prefix = "WARN: "
    elif lvl == 2:
        prefix = "CRITICAL: "

    print(f"{prefix}{msg}")
    if lvl == 2:
        sys.exit(1)


def _hist_equalize(arr, normalized_max=255):
    """Perform histogram equalization on a 2D numpy array (i.e. micrograph image
    representation).
    """

    from matplotlib import pyplot as hplt

    hist_fig = hplt.figure()
    hist_arr, bins, _ = hplt.hist(arr.ravel(), bins="auto", density=True)
    cdf = hist_arr.cumsum()  # cumulative distribution function
    cdf = normalized_max * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    hplt.close(hist_fig)
    img_equalized = np.interp(arr.ravel(), bins[:-1], cdf)
    res = np.array(img_equalized.reshape(arr.shape))

    return res


def _invert(arr):
    a = arr.copy()
    rng = a.max() - a.min()
    a = np.vectorize(lambda x: rng - x)(a)
    return a


def _read_mrc(path, mmap=False):
    """Use the mrcfile module to read data from micrograph at given path."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mmap:
            with mrcfile.mmap(path, mode="r", permissive=True) as f:
                mrc = f.data
        else:
            with mrcfile.open(path, mode="r", permissive=True) as f:
                mrc = f.data

    return mrc


# plotting


def single_mrc_overlay(
    mrcimg_norm,
    box_lists,
    legend_labels,
    mrc_name,
    out_dir=None,
    samp_size=None,
    stack_dim=None,
    stack_idx=None,
    do_hist_eq=True,
    do_flip_contrast=False,
    do_force=False,
    quiet=False,
):

    if out_dir:
        matplotlib.use("Agg")
    else:
        # print out to display
        # requires pycairo and pygobject
        matplotlib.use("GTK3Agg")

    from matplotlib import pyplot as plt

    overlay_fig, overlay_ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=400)
    # np.flipud flips image vertically; origin='lower' flips it again and puts the origin at bottom left
    # overlay_ax[0, 0].imshow(np.flipud(hist_equalize(mrcimg_norm)), cmap=plt.cm.gray, origin='lower')
    # ...but on the other hand, it appears RELION keeps the origin at top left, so don't bother

    if len(mrcimg_norm.shape) == 3:
        while stack_dim not in [0, 1, 2]:
            stack_dim = int(
                input(
                    f"This image has shape {mrcimg_norm.shape} and is most likely a "
                    f".mrcs stack. Which is the stacking dimension? [0, 1, 2]: "
                )
            )
        while stack_idx not in range(mrcimg_norm.shape[stack_dim]):
            num_slices = mrcimg_norm.shape[stack_dim]
            stack_idx = int(
                input(
                    f"There are {num_slices} slices in this stack. "
                    f"Which should be plotted? [0...{num_slices - 1}]: "
                )
            )

        mrcimg_norm = np.moveaxis(mrcimg_norm, stack_dim, 0)[stack_idx, :, :]
        _log(
            f"slicing dimension {stack_dim} at index {stack_idx}; "
            f"new shape is {mrcimg_norm.shape}",
            quiet=quiet,
        )

    # flip contrast if necessary
    mrcimg_disp = _invert(mrcimg_norm) if do_flip_contrast else mrcimg_norm

    # histogram-equalize if necesary
    mrcimg_disp = _hist_equalize(mrcimg_disp) if do_hist_eq else mrcimg_disp

    overlay_ax.imshow(mrcimg_disp, cmap=plt.get_cmap("gray"), aspect="equal")
    overlay_ax.axis("off")
    colors = [GT_COLOR] + PICKER_COLORS

    # add boxes to micrograph
    for i, arr in enumerate(box_lists):
        downsamp_boxarr = random.sample(arr, len(arr))[
            :samp_size
        ]  # slicing with None gives whole list
        for box in downsamp_boxarr:
            rect = patches.Rectangle(
                (box.x, box.y),
                box.w,
                box.h,
                linewidth=0.4,
                edgecolor=colors[i],
                facecolor="none",
            )
            overlay_ax.add_patch(rect)

    # generate micrograph title
    fig_title = "%s\n(Sample Size: %s)" % (
        mrc_name,
        "All" if samp_size is None else str(samp_size),
    )
    overlay_ax.set_title(fig_title, fontsize=4, pad=3)

    extra_artists = []

    # generate legend (.box1 are the blue boxes) if we're doing multiple plots
    if legend_labels:
        legend_list = [
            (name, box_lists[idx][0].w) for idx, name in enumerate(legend_labels)
        ]
        legend_patches = []
        for idx, item in enumerate(legend_list):
            lbl = "%s (%s calls, box size: %s)" % (
                str(item[0]),
                str(len(box_lists[idx])),
                str(item[1]),
            )
            legend_patches.append(
                patches.Patch(color=([GT_COLOR] + PICKER_COLORS)[idx], label=lbl)
            )
        legend = overlay_fig.legend(
            handles=legend_patches,
            bbox_to_anchor=(0.5, 0),
            loc="lower center",
            fontsize=5,
            ncol=1,
            frameon=False,
        )
        extra_artists.append(legend)

    # resize and write out

    if out_dir:
        overlay_fig.tight_layout(
            rect=(0, 0.18, 1, 0.98)
        )  # rect=(left, bot, right, top)
        save_destination = Path(out_dir) / f"overlay_{mrc_name.lower()}.png"
        if save_destination.is_file() and not do_force:
            _log("re-run with the force flag to replace existing files", 2)
        overlay_fig.savefig(save_destination, bbox_extra_artists=extra_artists)
        _log(f"figure saved to: {save_destination}", quiet=quiet)
    else:
        plt.figure(overlay_fig.number)
        plt.show()


if __name__ == "__main__":
    # handle CLI calls
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", required=False, help="Path to micrograph file")
    parser.add_argument("-b", required=False, nargs="+", help="Path to input BOX files")
    parser.add_argument(
        "-o",
        required=False,
        help="Output directory (if omitted, will open result in display window)",
    )
    parser.add_argument(
        "--samp_size",
        required=False,
        help="Number of boxes per boxfile to plot (default: 50)",
    )
    parser.add_argument(
        "--stack_dim",
        required=False,
        type=int,
        help="Stacking dimension of .mrcs stack",
    )
    parser.add_argument(
        "--stack_idx",
        required=False,
        type=int,
        help="Index to pull from stacking dimension of .mrcs stack",
    )
    parser.add_argument(
        "--no_hist_eq", action="store_false", help="Turn off histogram equalization"
    )
    parser.add_argument(
        "--flip_contrast", action="store_true", help="Invert micrograph contrast"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow files in output directory to be overwritten and make output "
        "directory if it does not exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence info-level output",
    )

    a = parser.parse_args()

    # input verification
    if not a.m and not a.b:
        _log("please specify micrographs or boxfiles or both", 2)
    if a.b and len(a.b) > len(PICKER_COLORS):
        _log("more box files were provided than colors in the available cycle", 2)

    if a.b:
        a.b = [Path(p).resolve() for p in a.b]
    if a.o:
        a.o = Path(a.o).resolve()
        if not a.o.is_dir() and not a.force:
            _log("re-run with the force flag to create output directory", 2)
        a.o.mkdir(parents=True, exist_ok=True)
        _log(f"using output directory: {a.o}", quiet=a.quiet)

    # convert all input boxfiles to lists of Box namedtuples
    if a.b:
        box_dfs = process_conversion(a.b, "box", "box", out_dir=None, quiet=True)
        for df in box_dfs.values():
            if "conf" not in df.columns:
                df["conf"] = 1
        box_lists = [
            list(df.itertuples(name="Box", index=False)) for df in box_dfs.values()
        ]

    # load mrc
    if a.m:
        try:
            a.m = Path(a.m).resolve()
        except KeyError:
            _log("invalid path", 2)
        mrc_img = _read_mrc(a.m)
        mrc_name = Path(a.m).stem
    else:
        box_lists_flat = [b for bs in box_lists for b in bs]
        max_x = max([b.x + b.w for b in box_lists_flat]) + MRC_PLOT_MARGIN
        max_y = max([b.y + b.h for b in box_lists_flat]) + MRC_PLOT_MARGIN
        mrc_img = np.ones((max_y, max_x))
        mrc_name = "[boxes only]"
        _log(f"assuming mrc shape of {max_y} rows, {max_x} cols")

    # report run options used
    _log(f"building overlay_fig: {mrc_name}")

    single_mrc_overlay(
        mrc_img,
        box_lists,
        a.b,
        mrc_name,
        out_dir=a.o,
        samp_size=a.samp_size,
        stack_dim=a.stack_dim,
        stack_idx=a.stack_idx,
        do_hist_eq=a.no_hist_eq,
        do_flip_contrast=a.flip_contrast,
        do_force=a.force,
        quiet=a.quiet,
    )
