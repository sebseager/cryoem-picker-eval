import sys
import warnings
import mrcfile
import pickle
import numpy as np
from pathlib import Path
from collections import namedtuple


Box = namedtuple("Box", ["x", "y", "w", "h", "conf"])
Intersection = namedtuple("Intersection", ["box1", "box2", "jac"])

# set defaults starting from rightmost positional arg (i.e. confidence)
Box.__new__.__defaults__ = (0.0,)

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
MRC_EXTS = [".mrc", ".mrcs"]
BOXFILE_EXTS = [".box", ".cbox", ".star", ".tsv", ".coord"]
TSV_SEP = "\t"


def norm_path(path):
    """Normalize path and return as a pathlib.Path object."""
    return Path(path).expanduser().resolve()


def log(*msgs, lvl=0, quiet=False):
    """Format and print message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if quiet=True)
    1: warning (print and continue execution)
    2: error (print and exit with code 1)
    """

    if lvl == 0 and quiet:
        return

    prefix = ""
    if lvl == 0:
        prefix = "INFO:"
    elif lvl == 1:
        prefix = "WARN:"
    elif lvl == 2:
        prefix = "CRITICAL:"

    print(f"{prefix}", *msgs)
    if lvl == 2:
        sys.exit(1)


def style_ax(ax, xlab="", ylab="", aspect="auto", xlim=None, ylim=None):
    for axis in ("top", "bottom", "left", "right"):
        # hide axes
        if axis in ("top", "right"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

        # general formatting
        ax.spines[axis].set_linewidth(1.0)
        ax.set_xlabel(xlab, fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            direction="out",
            length=4.0,
            width=1.0,
            color="k",
            labelsize=12,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            direction="out",
            length=4.0,
            width=1.0,
            color="k",
            labelsize=12,
        )
        ax.grid(color="gray", ls=":", lw=0.5, alpha=0.5)

        # set aspect ratio
        # values can be "auto", "equal" (same as 1), or a float
        ax.set_aspect(aspect)

        # set x and y limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # useful for chaining (is ax modified in place?)
        return ax


def read_mrc(path, mmap=False, with_header=False):
    """Use the mrcfile module to read data from micrograph at given path."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mmap:
            with mrcfile.mmap(path, mode="r", permissive=True) as f:
                mrc = f.data
        else:
            with mrcfile.open(path, mode="r", permissive=True) as f:
                mrc = f.data

    if with_header:
        return f.header, mrc
    else:
        return mrc.astype(np.float32)


def flatten(x, keep_inmost_depth=None):
    """Flatten the given list via recursive generator, preserving specified number of
    inmost list dimensions (tuples are not considered lists).
    """

    if keep_inmost_depth:
        is_list = isinstance(x[keep_inmost_depth - 1], list)
    else:
        is_list = isinstance(x, list)
    if x and not is_list:
        yield x
    else:
        for a in x:
            yield from flatten(a)


def read_from_pickle(path):
    """Return the object stored in the given pickle file."""

    with open(path, "rb") as f:
        return pickle.load(f)


def write_to_pickle(out_dir, obj, filename, rename_on_collision=True, force=False):
    """Write serializable object to pickle file with filename in given directory.

    Args:
        out_dir (str): Path to output directory
        obj (object): Object of any serializable type (None, True, False, str, int,
            byte, bytearray, non-lambda functions, and any tuple, list, set, or dict
            of these)
        filename (str): Target filename (including extension) that obj will be written to
        rename_on_collision (bool, optional): If force is False and a filename
            collision occurs, rename the file by appending the current datetime to
            obj always gets written to a file. Defaults to True.
        force (bool, optional): Overwrite target file. Defaults to False.
    """

    def perform_write(path):
        # write mode "x" raises an exception if file already exists
        write_mode = "wb" if force else "xb"
        with open(path, write_mode) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"wrote to {path}")

    # make sure output directory exists
    out_dir = norm_path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
    out_path = out_dir / filename

    try:
        perform_write(out_path)
    except FileExistsError:
        if rename_on_collision:
            log(f"file already exists at {out_path}; attempting to rename")
            time_str = datetime.now().strftime("_%y%m%d%H%M%S")
            out_path = out_path.with_suffix("") + time_str + out_path.suffix
            perform_write(out_path)
        else:
            log(f"file already exists at {out_path}; skipping", lvl=1)
            return


def linear_normalize(vals, new_min=0, new_max=1, always_norm=False):
    """Ensure all numbers in vals fit between new_min and new_max. If the existing
    range is within the new range and always_norm is False, do nothing. Otherwise,
    linearly normalize values to new min and max.
    """

    vals = np.array(vals)
    old_max, old_min = vals.max(), vals.min()
    old_range, new_range = old_max - old_min, new_max - new_min
    if always_norm or old_min < new_min or old_max > new_max:
        if old_range == 0:
            # if the old range was 0, arbitrarily set everything to new_min
            return vals * 0 + new_min
        else:
            # otherwise do linear normalization
            return (vals - old_min) * new_range / old_range + new_min

    # otherwise do nothing
    return vals
