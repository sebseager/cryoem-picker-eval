import argparse
import numpy as np
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
from sklearn.metrics._ranking import _binary_clf_curve
from common import *
from write_filematching import read_file_matching, read_boxfiles


def jaccard(box1, box2):
    """Jaccard index (intersection-over-union) min-max algorithm. Finds intersection
    of two rectangles by combining the rightmost of the two left edges, bottommost of
    the top edges, topmost of the bottom edges, and leftmost of the right edges.
    """

    box1_xmin = box1.x
    box1_xmax = box1.x + box1.w
    box1_ymin = box1.y
    box1_ymax = box1.y + box1.h
    box2_xmin = box2.x
    box2_xmax = box2.x + box2.w
    box2_ymin = box2.y
    box2_ymax = box2.y + box2.h
    intersection_xmin = max(box1_xmin, box2_xmin)
    intersection_xmax = min(box1_xmax, box2_xmax)
    intersection_ymin = max(box1_ymin, box2_ymin)
    intersection_ymax = min(box1_ymax, box2_ymax)
    intersection_width = float(intersection_xmax) - float(intersection_xmin)
    intersection_height = float(intersection_ymax) - float(intersection_ymin)

    if intersection_width > 0 and intersection_height > 0:
        num = intersection_width * intersection_height
        denom = box1.w * box1.h + box2.w * box2.h - num
        if denom > 0:
            jac = num / denom
        else:
            jac = 0
    else:
        jac = 0

    return jac


def box_intersects(box, box_list, max_only=True, reverse_box_order=False):
    """Get a box's intersections (or max intersection only, if max_only is True) with
    a box list. If reverse_box_order is True, put make `box` the second member of
    returned Intersections.
    """

    intersections = []
    for box2 in box_list:
        jac = jaccard(box, box2)
        if jac > 0:
            if reverse_box_order:
                intersections.append(Intersection(box2, box, jac))
            else:
                intersections.append(Intersection(box, box2, jac))
    if max_only:
        res = max(intersections, key=lambda x: x.jac, default=None)
    else:
        res = intersections
    return res


def many_one_matching(gt_boxes, pckr_boxes, conf_rng=(0, 1)):
    """Give each box in pckr_boxes its best pairing in gt_boxes, and allow many-to-one
    gt-to-pckr pairings (i.e. don't remove anything and allow reuse of boxes from the
    ground truth set). Iterating over gt_boxes instead would measure the "goodness"
    of the ground truth set (not desired here).

    Args:
        gt_boxes (list):  List of ground truth boxes
        pckr_boxes (list): List of particle picker boxes
        conf_rng (tuple, optional): Tuple of length 2, representing minimum and maximum
            confidences (between 0 and 1), inclusive, within which a box will be
            returned in intersections list. Defaults to (0, 1).

    Returns:
        list: List of Intersection namedtuples. If an intersection with gt_boxes is
            not found for a particular pckr_box, an Intersection with
            Box(None, None, None, None, None) as box1 will be returned.
    """

    intersections = []
    for pckr_box in pckr_boxes:
        pckr_box_max_intersection = max_box_intersection(
            pckr_box, gt_boxes, reverse_box_order=True
        )
        if pckr_box_max_intersection is not None:
            if conf_rng[0] <= pckr_box.conf <= conf_rng[1]:
                intersections.append(pckr_box_max_intersection)
        else:
            intersections.append(
                Intersection(Box(None, None, None, None, None), pckr_box, jac=0)
            )

    return intersections


def maxbpt_matching(gt_boxes, pckr_boxes, conf_rng=(0, 1)):
    """Uses scipy implementation of Hungarian (Kuhn-Munkres) maximum bipartite matching
    algorithm to find a perfect matching, maximizing edge weights (Jaccard indices,
    here) and number of connections. Same as nx.minimum_weight_full_matching (docs:
    "this implementation defers the calculation of the assignment to SciPy").
    Note that, since this is a maximum matching, some non-overlapping gt_ and
    pckr_boxes may be paired to satisfy constraints of the algorithm; these can be
    identified by a Jaccard of exactly 0.
    """

    cost_matrix = None  # will be an NxM cost matrix (gt_boxes vs pckr_boxes)

    for gt_box in gt_boxes:
        gt_box_jacs = [
            jaccard(gt_box, p) if conf_rng[0] <= p.conf <= conf_rng[1] else 0
            for p in pckr_boxes
        ]

        if cost_matrix is None:
            cost_matrix = np.array(gt_box_jacs)
        else:
            cost_matrix = np.vstack((cost_matrix, gt_box_jacs))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    intersections = [
        Intersection(
            Box(
                gt_boxes[row_ind[i]].x,
                gt_boxes[row_ind[i]].y,
                gt_boxes[row_ind[i]].w,
                gt_boxes[row_ind[i]].h,
                gt_boxes[row_ind[i]].conf,
            ),
            Box(
                pckr_boxes[col_ind[i]].x,
                pckr_boxes[col_ind[i]].y,
                pckr_boxes[col_ind[i]].w,
                pckr_boxes[col_ind[i]].h,
                pckr_boxes[col_ind[i]].conf,
            ),
            cost_matrix[row_ind[i]][col_ind[i]],
        )
        for i in range(len(row_ind))
    ]

    return intersections


def jac_table(
    boxes,
    matching_func,
    gt_key="gt",
    box_precision=0,
    conf_precision=2,
    jac_precision=3,
    **matching_kwargs,
):
    """Generate a single table of ground-truth-to-picker boxfile matches, using the
    matching strategy provided by matching_func. Additional keyword arguments for
    the matching function can be passed in via matching_kwargs.

    Args:
        boxes (dict): Dictionary of Box objects, as returned by read_boxfiles.
        matching_func (function): Function that takes two lists of Box objects
            and returns a list of Intersection objects. Provide optional keyword
            arguments to this function via matching_kwargs.
        gt_key (str, optional): They key in boxes to use as the ground truth set.
            Defaults to "gt".
        box_precision (int, optional): Number of decimal places to round box
            coordinates and dimensions to. Set to None to do no rounding. Defaults to 0.
        conf_precision (int, optional): Number of decimal places to round confidence
            to. Set to None to do no rounding. Defaults to 2.
        jac_precision (int, optional): Number of decimal places to round Jaccard
            indices to. Set to None to do no rounding. Defaults to 3.

    Returns:
        dict: Dictionary with the following structure:
            {
                "mrc": [path1, path2, ...],
                "gt": [box1, box2, ...],
                "picker1": [(box1, jaccard_overlap1), ...]
            },
            in which corresponding list indices (i.e., each "row") indicate each
            picker's match for a given ground truth. If a picker list contains None at
            some index, it means that the picker did not have a matched box for the
            ground truth at that index. If the ground truth list contains None at some
            index, it means that there is no correlation between any picker boxes at
            that index (i.e., picker boxes not matched with any ground truth).
    """

    # table is a dict keyed by (mrc_path, gt_box) pairs, with
    # {"picker1": box1, "picker2": box2, ...} dicts as values
    table = {}
    pickers = list(set(boxes.keys()) - set(gt_key))

    try:
        mrc_paths = list(boxes[gt_key].keys())
    except KeyError:
        log("gt_key not found in boxes", lvl=2)
        return

    # helper function to round a box, jac pair to the provided precision
    def round_if_needed(box, jac):
        round_jac = jac_precision is not None
        new_box = []
        for attr in ("x", "y", "w", "h", "conf"):
            val = getattr(box, attr)
            do_round, precision = (
                (conf_precision is not None, conf_precision)
                if attr == "conf"
                else (box_precision is not None, box_precision)
            )
            try:
                if not do_round:
                    raise TypeError  # abuse TypeError to skip rounding
                if precision == 0:
                    new_box.append(int(round(val, precision)))
                else:
                    new_box.append(round(val, precision))
            except TypeError:  # avoid rounding None
                new_box.append(val)
        try:
            new_jac = round(jac, jac_precision) if jac_precision is not None else jac
        except TypeError:  # avoid rounding None
            new_jac = jac
        return Box(*new_box), new_jac

    # add all boxes to matching table
    for mrc in mrc_paths:
        for picker in pickers:
            gt_boxes = boxes[gt_key][mrc]
            pckr_boxes = boxes[picker][mrc]
            matches = matching_func(gt_boxes, pckr_boxes, **matching_kwargs)
            for match in matches:
                key = (mrc, match.box1)
                try:
                    # if there's already a Box here, something's gone very wrong
                    # and we may have a duplicate of the key (mrc, match.box1)
                    assert table[key][picker] is None
                    table[key][picker] = round_if_needed(match.box2, match.jac)
                except KeyError:
                    # set the default for this key to a dict of Nones
                    table[key] = {p: None for p in pickers}

    # rearrange matching table into the return format:
    # {"mrc": [path1, ...], "gt": [box1, ...], "picker1": [box1, ...]},
    flat_table = {"mrc": [], "gt": []} + {p: [] for p in pickers}
    for (mrc, gt_box), picker_boxes in table.items():
        flat_table["mrc"].append(mrc)
        flat_table["gt"].append(gt_box)
        for picker, (box, jac) in picker_boxes.items():
            flat_table[picker].append((box, jac))

    return flat_table


def read_jac_table(path, mrc_key="mrc", gt_key="gt"):
    """Read table of box matches from file (as generated by write_jac_table) at the
    given path and convert to a dictionary of the following format:
    {"mrc": [path1, ...], "gt": [box1, ...], "picker1": [(box1, jaccard_overlap1), ...]}
    """

    df = pd.read_csv(norm_path(path), sep=TSV_SEP)
    table = {}

    try:
        table[mrc_key] = df[mrc_key].tolist()
        table[gt_key] = df[gt_key].tolist()
    except KeyError as e:
        log(f"required key not found in table: {e}", lvl=2)
        return

    picker_cols = list(set(df.columns) - set([mrc_key, gt_key]))
    try:
        # get picker names without _x, _y, ... suffixes
        pickers = list(set([c[: c.rindex("_")] for c in picker_cols]))
    except ValueError as e:
        log(f"improperly formatted picker column in table: {e}", lvl=2)
        return

    for picker in pickers:
        try:
            # filter columns for this picker's boxes and convert to list of Box objects
            picker_df = df[[f"{picker}_{attr}" for attr in Box._fields]]
            picker_df.columns = Box._fields
            boxes = picker_df.itertuples(index=False, name="Box")
            # also recover jaccard indices
            jacs = df[f"{picker}_jac"]
            # add to table
            table[picker] = list(zip(boxes, jacs))
        except KeyError as e:
            log(f"expected columns not found in table for '{picker}': {e}", lvl=2)
            continue

    return table


def write_jac_table(out_dir, table, filename, mrc_key="mrc", gt_key="gt", force=False):
    """Write Jaccard table (as returned by jac_table) to file."""

    # make sure output directory exists
    out_dir = norm_path(out_dir)
    if not out_dir.isdir():
        out_dir.mkdir(parents=True)

    # skip if matchings file already exists
    tsv_path = out_dir / filename
    if not force and tsv_path.exists():
        log("set force to True to overwrite existing file matches", lvl=2)
        exit(1)

    # unpack all tuples into separate columns in table
    table_df = pd.DataFrame()
    table_df[mrc_key] = table[mrc_key]
    table_df[gt_key] = table[gt_key]
    for picker in set(table.keys()) - set((mrc_key, gt_key)):
        # at this point table[picker] is a list of (box, jac) tuples
        # first unzip it into a list of boxes and a list of jaccard indices
        boxes, jacs = zip(*table[picker])
        # then make temporary dataframe for the unpacked boxes
        df = pd.DataFrame(boxes, columns=[f"{picker}_{attr}" for attr in Box._fields])
        table_df = pd.concat([table_df, df], axis=1)
        # finally add the jaccard index column last
        table_df[f"{picker}_jac"] = jacs

    # write matchings to tsv (mode "x" raises an error if file already exists)
    try:
        table_df.to_csv(tsv_path, sep=TSV_SEP, index=False, mode="w" if force else "x")
    except FileExistsError:
        log(f"file {tsv_path} already exists", lvl=2)
        exit(1)
    log(f"wrote file matches to {tsv_path}")


if __name__ == "__main__":
    # each of these should match a function named *_matching in this file
    matching_func_names = ["many_one", "maxbpt"]

    parser = argparse.ArgumentParser(
        description="Generate a table of matches between ground truth and picker "
        "boxes. The table has the following columns: mrc, gt, picker1_x, picker1_y, "
        "picker1_w, picker1_h, picker1_conf, picker1_jac, ..."
    )
    parser.add_argument(
        "out_dir",
        help=f"Output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "file_matching_path",
        help=f"Path to file matching TSV",
    )
    parser.add_argument(
        "--methods",
        help="Matching method(s) to use. Defaults to all.",
        nargs="+",
        choices=matching_func_names,
        default=matching_func_names,
    )
    parser.add_argument(
        "--conf_range",
        help="Range of confidence values to consider during matching. "
        "Defaults to [0, 1].",
        nargs=2,
        type=float,
        default=[0, 1],
    )
    parser.add_argument(
        "--box_precision",
        help="Number of decimal places to round box coordinates and dimensions to. "
        "Defaults to 0.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--conf_precision",
        help="Number of decimal places to round confidence values to. Defaults to 3.",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--jac_precision",
        help="Number of decimal places to round Jaccard indices to. Defaults to 2.",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--gt_key",
        help=f"Key in file matches to use as the ground truth set. "
        "Defaults to 'gt'.",
        default="gt",
    )
    parser.add_argument(
        "--mrc_key",
        help=f"Key in file matches to use as the micrograph set. " "Defaults to 'mrc'.",
        default="mrc",
    )
    parser.add_argument(
        "--force",
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )

    a = parser.parse_args()

    file_matches = read_file_matching(a.file_matching_path)
    boxes = read_boxfiles(file_matches, mrc_key=a.mrc_key)

    for method in a.methods:
        matching_func = exec(method + "_matching")
        table = jac_table(
            boxes,
            matching_func,
            gt_key=a.gt_key,
            box_precision=a.box_precision,
            conf_precision=a.conf_precision,
            jac_precision=a.jac_precision,
            # matching_kwargs start here
            conf_rng=a.conf_range,
        )
        write_jac_table(
            a.out_dir,
            table,
            f"{method}_matches.tsv",
            gt_key=a.gt_key,
            mrc_key=a.mrc_key,
            force=a.force,
        )
