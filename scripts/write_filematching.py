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
    paths = {name: [norm_path(x) for x in lst] for name, lst in paths.items()}

    # let the user know what we found
    for name, lst in paths.items():
        log(f"{name}: found {len(lst)} files")

    # find matching file groups
    match_groups = {p: [] for p in paths}
    for m_path in paths[primary_path_key]:
        m_stem = m_path.stem.lower()
        matches = {
            name: next((p for p in lst if p.stem.lower().startswith(m_stem)), None)
            for name, lst in paths.items()
        }
        if all(matches.values()):
            for name, path in matches.items():
                match_groups[name].append(path)

    n_matches = len(match_groups[list(match_groups.keys())[0]])

    if not n_matches:
        log("no match groups found", lvl=1)
        return

    log(f"matched {n_matches} micrographs")
    return match_groups


def read_file_matching(path):
    """Read file matching from file as written by file_matching, and return as a
    dictionary (where keys are file group names and values are lists of file names).
    """

    df = pd.read_csv(norm_path(path), sep=TSV_SEP)
    return df.to_dict(orient="list")


def write_file_matching(out_dir, matches, filename="file_matches.tsv", force=False):
    """Write file matching (as returned by file_matching) to file."""

    # make sure output directory exists
    out_dir = norm_path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # skip if matchings file already exists
    out_path = out_dir / filename
    if not force and out_path.exists():
        log("set force to True to overwrite existing file matches", lvl=2)
        exit(1)

    # write matchings to tsv (mode "x" raises an error if file already exists)
    df = pd.DataFrame(matches)
    try:
        df.to_csv(out_path, sep=TSV_SEP, index=False, mode="w" if force else "x")
    except FileExistsError:
        log(f"file {out_path} already exists", lvl=2)
        exit(1)
    log(f"wrote file matches to {out_path}")


def read_boxfiles(file_matches, mrc_key="mrc", norm_conf=None):
    """Provided lists of box file paths (must follow the EMAN box file format, with
    columns: x, y, width, height, confidence) as returned by file_matching, load all
    boxes into a dictionary of Box objects with the following structure:
    {picker_name: {mrc_path: [box1, box2, ...]}}.
    If a (new_min, new_max) tuple is provided to norm_conf, forcibly normalize all
    incoming confidences to that range.
    """

    boxes = {}
    for name, paths in file_matches.items():
        if name == mrc_key:
            continue
        boxes[name] = {}
        for mrc_path, boxfile_path in zip(file_matches[mrc_key], paths):
            df = tsv_to_df(boxfile_path)

            # if df columns don't have names, infer column names
            if all(df.columns == range(len(df.columns))):
                n_cols = len(df.columns)
                if n_cols == 4:
                    df.columns = list(Box._fields[:-1])
                elif n_cols == 5:
                    df.columns = list(Box._fields)
                elif n_cols < 4:
                    log(f"box file needs at least 4 columns ({boxfile_path})", lvl=2)
                    return
                else:
                    log(f"box file has {n_cols} columns; keeping the first 5", lvl=1)
                    df = df.drop(list(range(5, n_cols)), axis=1)
                    df.columns = list(Box._fields)

            if len(df.columns) == 4:
                df["conf"] = 0.0
            if norm_conf:
                df["conf"] = linear_normalize(df["conf"], *norm_conf, always_norm=True)

            try:

                # Box._fields = ['x', 'y', 'w', 'h', 'conf']
                new_boxes = [Box(*row) for row in df[list(Box._fields)].values]
            except KeyError as e:
                log(f"box file does not have all required columns ({e})", lvl=2)
                return

            try:
                boxes[name][mrc_path].extend(new_boxes)
            except KeyError:
                boxes[name][mrc_path] = new_boxes

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
        "-o",
        help="Output directory (will be created if it does not exist)",
        required=True,
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

    paths_dict = {}
    current_key = None
    for p in paths:
        if p.startswith("--"):
            current_key = p[2:]
            continue
        if current_key:
            paths_dict[current_key] = paths_dict.get(current_key, []) + [p]

    matches = file_matching(a.primary_key, **paths_dict)
    write_file_matching(a.o, matches, force=a.force)
