import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import mrcfile
import warnings


# utils


def _log(msg, lvl=0, quiet=False):
    """Format and print message to console with one of the following logging levels:
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


def _save_to_img(arr, save_path=None, force=False):
    scaled = (arr * 255 / np.max(arr)).astype(np.uint8)  # scale to 0...255
    png = Image.fromarray(scaled)
    if save_path:
        if Path(save_path).is_file() and not force:
            _log("re-run with the force flag to allow overwriting", 2)
        png.save(save_path)
    return png


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("mrc", help="Micrograph file(s) to convert to PNG", nargs="+")
    parser.add_argument(
        "-o",
        help="Output directory for new PNG files",
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
        help="Allow files in output directory to be overwritten and make output "
        "directory if it does not exist",
    )

    a = parser.parse_args()
    a.mrc = np.atleast_1d(a.mrc).tolist()
    a.o = Path(a.o).resolve()

    if a.force:
        a.o.mkdir(parents=True, exist_ok=True)

    for mrc_path in tqdm(a.mrc):
        mrc = _read_mrc(Path(mrc_path).resolve())
        out_path = os.path.join(a.o, Path(mrc_path).stem + f".{a.f}")
        _save_to_img(mrc, save_path=out_path, force=a.force)

    _log(f"Done. Output is in {a.o}")
