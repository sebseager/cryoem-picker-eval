import sys
import os
import numpy as np
import argparse
from pathlib import Path
import mrcfile
import warnings
from common import log


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join MRCS stacks")
    parser.add_argument("mrcs_files", nargs="+", help="MRCS files to join")
    parser.add_argument(
        "-o",
        help="Path to output file (default: joined.mrcs)",
        default="joined.mrcs",
    )
    parser.add_argument(
        "--force",
        help="Allow overwriting output file",
        action="store_true",
    )

    a = parser.parse_args()

    mrcs_paths = [Path(x) for x in a.mrcs_files]
    mrcs_files = [_read_mrc(p) for p in mrcs_paths]

    # make sure all files are 3d with stack dimension as 0th axis
    for i, mrc in enumerate(mrcs_files):
        if len(mrc.shape) == 2:
            mrcs_files[i] = mrc[np.newaxis, :, :]

    try:
        log(f"joining input shapes: \n{[x.shape for x in mrcs_files]}")
        joined_mrc = np.concatenate(mrcs_files, axis=0)
        new_shp = joined_mrc.shape
        log(f"output shape: {new_shp}")
        with mrcfile.new_mmap(a.o, shape=new_shp, mrc_mode=0, overwrite=a.force) as f:
            log(f"writing output to {a.o}")
            for z in range(new_shp[0]):
                f.data[z, :, :] = joined_mrc[z, :, :]
    except AttributeError:
        log("failed to open one or more mrcs files", 2)

    log("done")
