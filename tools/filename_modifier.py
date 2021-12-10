import os
import sys
import shutil
import re
import argparse
from tqdm import tqdm
from pathlib import Path


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


def replace_filename(
    path,
    substring,
    replacement,
    skip_suffixes=0,
    skip_first=0,
    skip_last=0,
):
    """Replace substring with replacement in path.
    Ignore skip_suffixes number of path suffixes.
    Ignore first skip_first occurrences and last skip_last occurrences.
    """

    path = Path(path).expanduser().resolve()
    if skip_suffixes > len(path.suffixes):
        _log(
            f"skip_suffixes {skip_suffixes} > len({path.suffixes})",
            1,
        )

    substring = re.escape(substring)
    exts_len = sum(len(s) for s in path.suffixes)
    from_name = path.name[:-exts_len] + "".join(path.suffixes[: -skip_suffixes or None])
    matches = [x for x in re.finditer(substring, from_name)]
    matches = matches[skip_first : -skip_last or None]

    if len(matches) == 0:
        _log(f"no matches for substring {substring} in {from_name}", 1)
        return path

    to_name = from_name[: matches[0].start()]
    for i, m in enumerate(matches):
        next_start = matches[i + 1].start() if i + 1 < len(matches) else None
        to_name += replacement + from_name[m.end() : next_start]

    # put skipped suffixes back
    to_name += "".join(path.suffixes[-skip_suffixes or None :])

    tqdm.write(f"{from_name} -> {to_name}")

    return path.parent / to_name


def move_file(from_path, to_path, do_force):
    """Move file from from_path to to_path.
    If do_force=True, overwrite existing file.
    """

    from_path = Path(from_path).expanduser().resolve()
    to_path = Path(to_path).expanduser().resolve()

    if not do_force and to_path.exists():
        _log(f"use --force to overwrite existing file {to_path}", lvl=2)

    os.rename(from_path, to_path)


def copy_file(from_path, to_path, do_force):
    """Copy file from from_path to to_path.
    If do_force=True, overwrite existing file.
    """

    from_path = Path(from_path).expanduser().resolve()
    to_path = Path(to_path).expanduser().resolve()

    if not do_force and to_path.exists():
        _log(f"use --force to overwrite existing file {to_path}", lvl=2)

    shutil.copy(str(from_path), str(to_path))


def link_file(from_path, to_path, do_force):
    """Link file from from_path to to_path.
    If do_force=True, overwrite existing file.
    """

    from_path = Path(from_path).expanduser().resolve()
    to_path = Path(to_path).expanduser().resolve()

    if not do_force and to_path.exists():
        _log(f"use --force to overwrite existing file {to_path}", lvl=2)

    os.symlink(from_path, to_path)


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Modify filenames by substring replacement."
    )

    parser.add_argument("files", help="Files to modify", nargs="+")
    parser.add_argument(
        "-b",
        help="Write behavior",
        choices=("inplace", "link", "copy"),
        default="inplace",
    )
    parser.add_argument(
        "-f",
        help="Substring to find (NOTE: single-quote this in bash to prevent parameter expansion)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        help="Substring to replace with (NOTE: single-quote this in bash to prevent parameter expansion)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--skip_suffixes",
        help="Number of extensions (successive dot-delimted strings at end of filename) to ignore",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--skip_first",
        help="Number of occurrences of found substring to skip from beginning of filename",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--skip_last",
        help="Number of occurrences of found substring to skip from end of filename",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--force",
        help="If write behavior is not inplace, allow overwriting of existing files",
        action="store_true",
    )

    a = parser.parse_args()

    # verification
    if a.skip_first < 0 or a.skip_last < 0 or a.skip_suffixes < 0:
        _log("skip_* args must be >= 0", lvl=2)

    _log(f"find substring: {a.f}")
    _log(f"replace with substring: {a.r}")

    for from_path in tqdm(a.files):
        to_path = replace_filename(
            from_path, a.f, a.r, a.skip_suffixes, a.skip_first, a.skip_last
        )

        # behavior
        if a.b == "inplace":
            move_file(from_path, to_path, a.force)
        elif a.b == "link":
            link_file(from_path, to_path, a.force)
        elif a.b == "copy":
            copy_file(from_path, to_path, a.force)
        else:
            _log(f"unrecognized behavior {a.b}", lvl=2)

    _log(f"done.")
