import argparse
import mrcfile
import numpy as np
from pathlib import Path
from common import log, read_mrc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice micrograph images or movies")
    parser.add_argument("mrcs_files", nargs="+", help="Micrograph file paths")
    parser.add_argument(
        "--sw",
        help="Numpy-style slice for width (e.g., ':-1' to remove the last column)",
        default=":",
    )
    parser.add_argument(
        "--sh",
        help="Numpy-style slice for height (e.g., ':-1' to remove the last row)",
        default=":",
    )
    parser.add_argument(
        "-o",
        required=True,
        help="Output directory (will be created if it does not exist)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    a = parser.parse_args()

    mrcs_paths = [Path(x) for x in a.mrcs_files]

    for i, path in enumerate(mrcs_paths):
        log(f"processing {path}")

        mrc = read_mrc(path, mmap=True)

        # single images
        if mrc.ndim == 2:
            new_mrc = eval(f"mrc[{a.sh}, {a.sw}]")
        elif mrc.ndim == 3:
            new_mrc = eval(f"mrc[:, {a.sh}, {a.sw}]")
        else:
            log(f"unsupported shape for {mrcs_paths[i]}: {mrc.shape}", lvl=1)
            continue

        log(f"resizing (h, w): {mrc.shape} -> {new_mrc.shape}")
        new_mrc_path = Path(a.o) / mrcs_paths[i].name

        try:
            with mrcfile.new(new_mrc_path, overwrite=a.force) as f_out:
                f_out.set_data(new_mrc)
            log(f"saved to {new_mrc_path}")
        except ValueError:
            log(f"{new_mrc_path} already exists; use --force to overwrite", lvl=1)
            continue
