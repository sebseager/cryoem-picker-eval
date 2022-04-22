import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import warnings
from common import *


def save_to_img(arr, save_path=None):
    scaled = (arr * 255 / np.max(arr)).astype(np.uint8)  # scale to 0...255
    png = Image.fromarray(scaled)
    if save_path:
        png.save(save_path)
    return png


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("mrc", help="Micrograph file(s) to convert to PNG", nargs="+")
    parser.add_argument(
        "-o",
        help="Output directory for new PNG files (will be created if it does not exist)",
        required=True,
    )
    parser.add_argument(
        "-f",
        help="Output format (must be supported by PIL, including: png, gif, jpg, tiff)",
        required=False,
        default="png",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow files in output directory to be overwritten",
    )

    a = parser.parse_args()
    a.mrc = [norm_path(p) for p in np.atleast_1d(a.mrc).tolist()]
    a.o = norm_path(a.o)
    a.o.mkdir(parents=True, exist_ok=True)

    for mrc_path in tqdm(a.mrc):
        out_path = a.o / str(mrc_path.stem + "." + a.f)
        if out_path.is_file() and not a.force:
            log(f"skipping {mrc_path} (re-run with --force to allow overwriting)")
            continue
        mrc = read_mrc(Path(mrc_path).resolve())
        save_to_img(mrc, save_path=out_path)

    log(f"Done. Output is in {a.o}")
