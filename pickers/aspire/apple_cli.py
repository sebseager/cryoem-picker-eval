import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from aspire.apple.apple import Apple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line interface for the " "ASPIRE APPLE picker."
    )

    parser.add_argument("mrc", help="Micrograph file(s)", nargs="+")
    parser.add_argument(
        "-o",
        help="Output directory for coordinate files",
        required=True,
    )
    parser.add_argument("--particle_size", type=int)
    parser.add_argument("--max_particle_size", type=int)
    parser.add_argument("--min_particle_size", type=int)
    parser.add_argument("--query_image_size", type=int)
    parser.add_argument("--tau1", type=int)
    parser.add_argument("--tau2", type=int)
    parser.add_argument("--minimum_overlap_amount", type=int)
    parser.add_argument("--container_size", type=int)

    a = parser.parse_args()

    # validation
    a.mrc = [str(Path(f).resolve()) for f in np.atleast_1d(a.mrc)]
    a.o = str(Path(a.o).resolve())

    apple_picker = Apple()

    for attr in (
        "particle_size",
        "max_particle_size",
        "min_particle_size",
        "query_image_size",
        "tau1",
        "tau2",
        "minimum_overlap_amount",
        "container_size",
    ):
        val = getattr(a, attr)
        if val is None:
            print(f"Using default for param '{attr}': {getattr(apple_picker, attr)}")
        else:
            setattr(apple_picker, attr, val)

    setattr(apple_picker, "output_dir", a.o)

    for f in a.mrc:
        apple_picker.process_micrograph(f, show_progress=True)

    print(f"Done. Output directory: {a.o}")
