import os
import sys
import argparse
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm
from coord_converter import process_conversion
from sklearn.metrics import f1_score


def segmentation_f1_score(
    gt_boxes, pckr_boxes, conf_thresh=None, mrc_w=None, mrc_h=None
):
    """Calculate a score between ground truth and picker box lists by creating a
    segmentation map for each and computing the F1 score between them. Optionally
    provide micrograph width and height to fix segmentation map size.
    """

    # if micrograph width/height not set, calculate them from provided boxes
    if mrc_w is None:
        mrc_w = round(max([n.x + n.w for n in gt_boxes + pckr_boxes]))

    if mrc_h is None:
        mrc_h = round(max([n.y + n.h for n in gt_boxes + pckr_boxes]))

    # make binary arrays masking out GT/picker boxes
    gt_arr = np.zeros((mrc_h, mrc_w))
    pckr_arr = np.zeros((mrc_h, mrc_w))
    for b in gt_boxes:
        x, y, w, h = round(b.x), round(b.y), round(b.w), round(b.h)
        gt_arr[y : y + h, x : x + w] = 1
    for b in pckr_boxes:
        if conf_thresh is not None and b.conf < conf_thresh:
            continue
        x, y, w, h = round(b.x), round(b.y), round(b.w), round(b.h)
        pckr_arr[y : y + h, x : x + w] = 1

    f1 = f1_score(gt_arr.flatten(), pckr_arr.flatten())
    return f1


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Score detections between ground truth and particle picker "
        "coordinate sets, matching files by name. All coordinate files must be "
        "in the BOX file format. Use coord_converter.py to perform any necessary "
        "conversion."
    )

    parser.add_argument(
        "-g",
        help="Ground truth particle coordinate file(s)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-p",
        help="Particle picker coordinate file(s)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-c",
        help="Confidence threshold",
        type=int,
    )
    parser.add_argument(
        "--height", help="Micrograph height (pixels)", type=int, default=None
    )
    parser.add_argument(
        "--width", help="Micrograph width (pixels)", type=int, default=None
    )
    parser.add_argument(
        "--verbose", help="Print individual boxfile pair scores", action="store_true"
    )

    a = parser.parse_args()
    a.g = np.atleast_1d(a.g)
    a.p = np.atleast_1d(a.p)

    gt_names = [Path(f).stem.lower() for f in a.g]
    pckr_names = [Path(f).stem.lower() for f in a.p]

    # do startswith in case pickers append suffixes
    gt_matches = [g for g in gt_names if sum(p.startswith(g) for p in pckr_names) > 0]

    if a.verbose:
        print(f"Found {len(gt_matches)} boxfile matches\n")

    assert len(gt_matches) > 0, "No paired ground truth and picker particle sets found"

    all_scores = []
    for match in tqdm(gt_matches):
        gt_path = next(f for f in a.g if Path(f).stem.lower() == match)
        pckr_path = next(f for f in a.p if Path(f).stem.lower().startswith(match))

        # process gt and pckr box files
        gt_dfs = process_conversion([gt_path], "box", "box", out_dir=None, quiet=True)
        p_dfs = process_conversion([pckr_path], "box", "box", out_dir=None, quiet=True)

        gt_df = list(gt_dfs.values())[0]
        pckr_df = list(p_dfs.values())[0]

        for df in (gt_df, pckr_df):
            if "conf" not in df.columns:
                df["conf"] = 1

        gt_boxes = list(gt_df.itertuples(name="Box", index=False))
        pckr_boxes = list(pckr_df.itertuples(name="Box", index=False))

        try:
            score = segmentation_f1_score(
                gt_boxes,
                pckr_boxes,
                conf_thresh=a.c,
                mrc_w=a.width,
                mrc_h=a.height,
            )
            all_scores.append(score)
        except Exception as e:
            tqdm.write(f"Error scoring {gt_path} and {pckr_path} ({e})")
            continue

        if a.verbose:
            tqdm.write(f"Indiv. score ({match}): {score}")

    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    if a.verbose:
        print(f"\nAVG. SCORE: {avg_score}, STDEV: {std_score}")
    else:
        print(avg_score)
