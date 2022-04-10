import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from coord_converter import tsv_to_df
from common import *


def file_matching(primary_path_key, **paths):
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
    paths = {name: [norm_path(x) for x in paths] for name, paths in paths.items()}

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


def read_file_matching(out_dir):
    """Read file matching from file as written by file_matching, and return as a
    dictionary (where keys are file group names and values are lists of file names).
    """

    df = pd.read_csv(norm_path(out_dir) / FILE_MATCHES_NAME, sep=TSV_SEP)
    return df.to_dict(orient="list")


def read_boxfiles(file_matches, mrc_key="mrc"):
    """Provided lists of box file paths (must follow the EMAN box file format, with
    columns: x, y, width, height, confidence) as returned by file_matching, load all
    boxes into a dictionary of Box objects with the following structure:
    {picker_name: {mrc_path: [box1, box2, ...]}}.
    """

    boxes = {}
    for name, paths in file_matches.items():
        if name == mrc_key:
            continue
        boxes[name] = {}
        for mrc_path, boxfile_path in zip(file_matches[mrc_key], paths):
            df = tsv_to_df(boxfile_path)
            try:
                # Box._fields = ['x', 'y', 'w', 'h', 'conf']
                new_boxes = [Box(*row) for row in df[list(Box._fields)].values]
            except KeyError as e:
                log(f"box file does not have all required columns ({f})", lvl=2)
                return
            try:
                boxes[name][mrc_path].append(new_boxes)
            except KeyError:
                boxes[name][mrc_path] = [new_boxes]

    return boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script matches named file groups by common file name to "
        "a given 'primary' path set. In addition to the provided arguments, specify "
        "file path groups by adding your own keyword arguments. For example: "
        f"{Path(__file__).name} out_dir/ --primary_path_key mrc "
        "--mrc path1 path2 ... --gt ... --picker1 ..."
    )
    parser.add_argument(
        "out_dir", help="Output directory (will be created if it does not exist)"
    )
    parser.add_argument(
        "--primary_key",
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
    a.out_dir = norm_path(a.out_dir)
    if not a.out_dir.isdir():
        a.out_dir.mkdir()

    # skip if matchings file already exists
    tsv_path = a.out_dir / FILE_MATCHING_NAME
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

    # write matchings to tsv (mode "x" raises an error if file already exists)
    matches = file_matching(a.primary_key, **paths_dict)
    match_df = pd.DataFrame(matches)
    try:
        df.to_csv(tsv_path, sep=TSV_SEP, index=False, mode="w" if a.force else "x")
    except FileExistsError:
        log(f"file {tsv_path} already exists", lvl=2)
        exit(1)
    log(f"wrote file matches to {tsv_path}")
