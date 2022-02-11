import sys
import os
import numpy as np
import argparse
from pathlib import Path
import mrcfile
from common import log, read_mrc


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
    mrcs_files = [read_mrc(p, mmap=True) for p in mrcs_paths]

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
        log("failed to open one or more mrcs files", lvl=2)

    log("done.")
