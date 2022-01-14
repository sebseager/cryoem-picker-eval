import os
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm


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


def split_coord_files(paths, out_dir, weights, n_header_lines, do_force=False):
    for path in tqdm(paths):
        with open(path, "r") as f:
            lines = f.readlines()

        lines = lines[n_header_lines:]  # remove header lines
        random.shuffle(lines)  # randomize order of lines in each path
        num_lines = len(lines)

        # calculate index split points from decimal weights
        split_points = [0]
        for i, w in enumerate(weights):
            spl = split_points[i] + int(w * num_lines)
            spl = spl if spl < num_lines else num_lines
            split_points.append(spl)
        split_points.append(num_lines)

        # split lines into chunks
        chunks = [
            lines[split_points[i] : split_points[i + 1]]
            for i in range(len(split_points) - 1)
        ]

        # write chunks to files
        for i, chunk in enumerate(chunks):
            weight = weights[i] if i < len(weights) else 1.0 - sum(weights)
            subdir = out_dir / f"split{i}_{weight * 100:.0f}_percent"
            subdir.mkdir(parents=False, exist_ok=True)
            write_path = subdir / path.name
            if not do_force and write_path.exists():
                _log(f"skipping existing file {write_path}", 1)
                continue
            with open(write_path, "w") as f:
                f.writelines(chunk)


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Split input coordinate files randomly by line to form n output "
        "coordinate file sets (e.g., x input files with approx. y lines each, split "
        "two ways, will form two groups of x files with approx. y/2 lines each). "
        "Optionally specify group weighting. NOTE: assumes one coordinate"
    )

    parser.add_argument(
        "input_files",
        help="Coordinate files to split (files will not be modified)",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        help="Output directory (will be created if it does not exist)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        help="Number of splits (default is 2)",
        type=int,
        default=2,
        required=True,
    )
    parser.add_argument(
        "--weights",
        help="List of decimal weights for each split; takes n - 1 arguments, as the "
        "nth split will get the remaining coordinates (default is to split evenly)",
        nargs="+",
    )
    parser.add_argument(
        "--n_header_lines",
        help="Ignore this many header lines in each file",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--force",
        help="Overwrite files in output if necessary",
        action="store_true",
    )

    a = parser.parse_args()

    # validation
    if a.n < 2:
        _log("n must be 2 or greater", 2)
    if a.weights is None:
        a.weights = [1.0 / a.n] * (a.n - 1)  # split evenly
    if len(a.weights) != a.n - 1:
        _log("n - 1 weights must be provided", 2)
    if sum(a.weights) >= 1.0:
        _log("weights cannot sum to >= 1.0", 2)

    a.o = Path(a.o).expanduser().resolve()
    a.input_files = [Path(f).expanduser().resolve() for f in a.input_files]

    if not all(f.exists() for f in a.input_files):
        _log("bad input paths", 2)

    # make output directory
    a.o.mkdir(parents=True, exist_ok=True)

    split_coord_files(a.input_files, a.o, a.weights, a.n_header_lines, a.force)

    _log(f"output in {a.o}", 0)
