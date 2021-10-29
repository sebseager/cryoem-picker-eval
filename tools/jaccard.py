import numpy as np
from collections import namedtuple
from scipy.optimize import linear_sum_assignment


Box = namedtuple("Box", ["x", "y", "w", "h", "conf"])
Intersection = namedtuple("Intersection", ["box1", "box2", "jac"])

# set defaults starting from rightmost positional arg (i.e. confidence)
Box.__new__.__defaults__ = (0.0,)


def jaccard(box1, box2):
    """
    Jaccard index (intersection-over-union) min-max algorithm. Finds intersection of two rectangles by combining
    the rightmost of the two left edges, bottommost of the top edges, topmost of the bottom edges, and leftmost
    of the right edges.
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
    """
    Get a box's intersections (or max intersection only, if max_only is True) with
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


def maxbpt(gt_boxes, pckr_boxes, conf_rng=(0, 1)):
    """
    Uses scipy implementation of Hungarian (Kuhn-Munkres) maximum bipartite matching
    algorithm to find a perfect matching, maximizing edge weights (Jaccard indices,
    here) and number of connections. Same as nx.minimum_weight_full_matching
    (docs: "this implementation defers the calculation of the assignment to SciPy").
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
