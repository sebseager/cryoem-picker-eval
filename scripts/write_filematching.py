import os
import sys
import argparse
from pathlib import Path
from common import *


def file_matching(primary_path_key, force=False, **paths):
    """Make name-based file matches. File paths are provided as keyword arguments,
    where the key is the name of a group of files and the value is those paths as a
    list (e.g., mrc=[path1, path2, ...], gt=[...], picker1=[...], ...). Specify the
    name of the file group against which all others will be matched with the
    primary_path_key argument. Returns a dictionary where keys are file group names
    and values are lists of file paths in matched order.
    """

    # make sure paths were passed in
    if not paths:
        log("no paths were provided", lvl=1)
        return

    # make sure primary_path_key is valid
    if primary_path_key not in paths:
        log(f"primary_path_key {primary_path_key} not in paths", lvl=1)
        return

    # normalize paths
    paths = {
        name: [Path(x).expanduser().resolve() for x in paths]
        for name, paths in paths.items()
    }

    # let the user know what we found
    for name, paths in paths.items():
        log(f"{name}: found {len(paths)} files")

    # find matching file groups
    match_groups = {p: [] for p in pckr_paths}
    for m_path in paths[primary_path_key]:
        m_stem = m_path.stem.lower()
        matches = {
            name: next((p for p in paths if p.stem.lower().startswith(m_stem)), None)
            for name, paths in enumerate(pckr_paths)
        }
        if all(matches.values()):
            for name, path in matches.items():
                match_groups[name].append(path)

    if not match_groups[match_groups.keys()[0]]:
        log("no match groups found", lvl=1)
        return

    log(f"matched {len(match_df)} micrographs")
    return match_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script matches named file groups by common file name to "
        "a given 'primary' path set. In addition to the provided arguments, specify "
        "file path groups by adding your own keyword arguments. For example: "
        f"{Path(__file__).name} out/dir --primary_path_key mrc "
        "--mrc path1 path2 ... --gt ... --picker1 ..."
    )
    parser.add_argument(
        "out_dir", help="Output directory (will be created if it does not exist)"
    )
    parser.add_argument(
        "--primary_path_key",
        help="Name of file group against which to match",
        required=True,
    )
    parser.add_argument(
        "--force",
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )

    a, paths = parser.parse_known_args()

    # make sure file path groups were provided
    if not paths:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # make sure output directory exists
    a.out_dir = Path(a.out_dir)
    if not a.out_dir.isdir():
        a.out_dir.mkdir()

    # skip if matchings file already exists
    tsv_path = Path(a.out_dir) / FILE_MATCHING_NAME
    if not a.force and tsv_path.exists():
        log("set force to True to overwrite existing file matches", lvl=2)
        exit(1)

    paths_dict = {}
    current_key = None
    for p in paths:
        if p.startswith("--"):
            current_key = p[2:]
            continue
        if current_key:
            paths_dict[current_key] = paths_dict.get(current_key, []) + [p]

    # write matchings to tsv
    matches = file_matching(a.primary_path_key, force=a.force, **paths_dict)
    match_df = pd.DataFrame(matches)
    df.to_csv(tsv_path, sep=TSV_SEP, index=False)
    log(f"wrote file matches to {tsv_path}")