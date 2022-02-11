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

    a = parser.parse_args()

    mrcs_paths = [Path(x) for x in a.mrcs_files]
    mrcs_files = [read_mrc(p) for p in mrcs_paths]

    for i, mrc in enumerate(mrcs_files):
        # single images
        if mrc.data.ndim == 2:
            new_mrc = eval(f"mrc.data[{a.sh}, {a.sw}]")
        elif mrc.data.ndim == 3:
            new_mrc = eval(f"mrc.data[:, {a.sh}, {a.sw}]")
        else:
            log(f"Unsupported shape for {mrcs_paths[i]}: {mrc.data.shape}")
            continue

        log(f"output shape: {new_mrc.shape}")
        new_mrc_path = Path(a.o) / mrcs_paths[i].name

        log(f"saving to {new_mrc_path}")
        with mrcfile.new(new_mrc_path, overwrite=a.force) as f_out:
            f_out.set_data(new_mrc)
            f_out.set_header(mrc.header)
