import argparse
import numpy as np
from collections import namedtuple
from tqdm import tqdm
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


def one_many_matching(gt_boxes, pckr_boxes, conf_range=(0, 1), **unused_kwargs):
    """Give each box in pckr_boxes its best pairing in gt_boxes, and allow one-to-many
    gt-to-pckr pairings (i.e. don't remove anything and allow reuse of boxes from the
    ground truth set). Iterating over gt_boxes instead would measure the "goodness"
    of the ground truth set (not desired here).

    Args:
        gt_boxes (list):  List of ground truth boxes
        pckr_boxes (list): List of particle picker boxes
        conf_range (tuple, optional): Tuple of length 2, representing minimum and
            maximum confidences (between 0 and 1), inclusive, within which a box
            will be returned in intersections list. Defaults to (0, 1).

    Returns:
        list: List of Intersection namedtuples. If an intersection with gt_boxes is
            not found for a particular pckr_box, an Intersection with None as box1
            will be returned.
    """

    intersections = []
    for pckr_box in pckr_boxes:
        pckr_box_max_intersection = box_intersects(
            pckr_box, gt_boxes, reverse_box_order=True
        )
        if pckr_box_max_intersection is not None:
            if conf_range[0] <= pckr_box.conf <= conf_range[1]:
                intersections.append(pckr_box_max_intersection)
        else:
            intersections.append(Intersection(None, pckr_box, jac=0))

    return intersections


def maxbpt_matching(gt_boxes, pckr_boxes, conf_range=(0, 1), **unused_kwargs):
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
            jaccard(gt_box, p) if conf_range[0] <= p.conf <= conf_range[1] else 0
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
                "picker1": [[(box1, jaccard_overlap1)], ...]
            },
            in which corresponding list indices (i.e., each "row") indicate each
            picker's match(es) for a given ground truth. If a picker list contains an
            empty list at some index, it means that the picker did not have any matched
            boxes for the ground truth at that index. For some micrograph path, any
            picker boxes corresponding to a ground-truth value of None indicate boxes
            that were unable to be paired with any ground-truth box.
    """

    # table is a dict keyed by (mrc_path, gt_box) pairs, with
    # {"picker1": box1, "picker2": box2, ...} dicts as values
    table = {}
    pickers = list(set(boxes.keys()) - {gt_key})

    try:
        mrc_paths = list(boxes[gt_key].keys())
    except KeyError:
        log("gt_key not found in boxes", lvl=2)
        return

    # helper function to round a box, jac pair to the provided precision
    def round_vals(box, jac):
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
    for mrc in tqdm(mrc_paths):
        for picker in pickers:
            gt_boxes = boxes[gt_key][mrc]
            pckr_boxes = boxes[picker][mrc]
            matches = matching_func(gt_boxes, pckr_boxes, **matching_kwargs)
            for match in matches:
                key = (mrc, match.box1)  # key by micrograph/ground-truth pair
                val = round_vals(match.box2, match.jac)  # picker box/jaccard index pair
                try:
                    # instead of storing the value by itself at this location, store
                    # it in a list of pairs; depending on the matching function, it
                    # may be possible for the same key to refer to multiple picker
                    # boxes
                    table[key][picker].append(val)
                except KeyError:
                    # set the default for this key to a dict of empty lists
                    table[key] = {p: [] for p in pickers}
                    table[key][picker] = [val]

    # rearrange matching table into the return format:
    # {"mrc": [path1, ...], "gt": [box1, ...], "picker1": [[(box1, jac1)], ...]},
    flat_table = {"mrc": [], "gt": [], **{p: [] for p in pickers}}
    for (mrc, gt_box), picker_boxes in table.items():
        flat_table["mrc"].append(mrc)
        flat_table["gt"].append(gt_box)
        for picker, box_jac_pair_list in picker_boxes.items():
            flat_table[picker].append(box_jac_pair_list)

    return flat_table


if __name__ == "__main__":
    # each of these should match a function named *_matching in this file
    matching_func_names = ["one_many", "maxbpt"]

    parser = argparse.ArgumentParser(
        description="Generate a table of matches between ground truth and picker "
        "boxes. The table has the following columns: mrc, gt, picker1_x, picker1_y, "
        "picker1_w, picker1_h, picker1_conf, picker1_jac, ..."
    )
    parser.add_argument(
        "file_matching_path",
        help=f"Path to file matching TSV",
    )
    parser.add_argument(
        "-o",
        help=f"Output directory (will be created if it doesn't exist)",
        required=True,
    )
    parser.add_argument(
        "--methods",
        help="Matching method(s) to use. Defaults to all.",
        nargs="+",
        choices=matching_func_names,
        default=matching_func_names,
    )
    parser.add_argument(
        "--conf_rng",
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
        help="Allow overwriting files in output directory",
        action="store_true",
    )

    a = parser.parse_args()

    file_matches = read_file_matching(a.file_matching_path)
    boxes = read_boxfiles(file_matches, mrc_key=a.mrc_key, norm_conf=a.conf_rng)

    for method in a.methods:
        log(f"starting calculations for method '{method}'")
        matching_func = locals()[method + "_matching"]
        table = jac_table(
            boxes,
            matching_func,
            gt_key=a.gt_key,
            box_precision=a.box_precision,
            conf_precision=a.conf_precision,
            jac_precision=a.jac_precision,
            # matching_kwargs start here
            conf_range=a.conf_rng,
        )
        write_to_pickle(
            a.o,
            table,
            f"{method}_matches.pickle",
            force=a.force,
        )
