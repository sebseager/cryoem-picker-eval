import os
import pdb
import math
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings
import numpy as np
import mrcfile
from collections import OrderedDict
from itertools import combinations_with_replacement
import pandas as pd
import random
from scipy.signal import fftconvolve
from scipy.ndimage.interpolation import rotate
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
import seaborn as sns
from coord_converter import star_to_df
from common import log, read_mrc
from consts import *

SQRT_2 = math.sqrt(2)
FFT_MODE = "same"


def read_class_avgs(mrcs_paths, n_max_classes=None):
    class_avgs = OrderedDict()
    class_idx = 0
    for m in mrcs_paths:
        mrcs = read_mrc(m, mmap=True)
        n_cls = len(mrcs) if n_max_classes is None else min(n_max_classes, len(mrcs))
        class_avgs[Path(m).stem] = mrcs[:n_cls]

    return class_avgs


def standardize_arr(arr):
    # https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    a = np.array(arr)
    return (a - np.mean(a)) / (np.std(a) * len(a))


def inscribed_square_naive(arr):
    # https://www.geeksforgeeks.org/maximum-size-sub-matrix-with-all-1s-in-a-binary-matrix/

    m = np.array(arr)

    # fix strange float precision things
    m[np.abs(m) < np.finfo(float).eps] = 0
    m[m != 0] = 1

    rows = len(m)  # no. of rows in M[][]
    cols = len(m[0])  # no. of columns in M[][]

    S = [[0 for k in range(cols)] for l in range(rows)]
    # here we have set the first row and column of S[][]

    # Construct other entries
    for i in range(1, rows):
        for j in range(1, cols):
            if m[i][j] == 1:
                S[i][j] = min(S[i][j - 1], S[i - 1][j], S[i - 1][j - 1]) + 1
            else:
                S[i][j] = 0

    # Find the maximum entry and
    # indices of maximum entry in S[][]
    max_of_s = S[0][0]
    max_i = 0
    max_j = 0
    for i in range(rows):
        for j in range(cols):
            if max_of_s < S[i][j]:
                max_of_s = S[i][j]
                max_i = i
                max_j = j

    # return top left row, top left col, and square side length
    return max_i - max_of_s, max_j - max_of_s, max_of_s


def inscribed_square_fast(img):
    """Shifts particle to center of image and extracts largest square.
    Better version of find_largest_sqaure_fast, debugged by CJC.
    This doesn't work on already-rotated arrays, since this may make mask not circular
    (will fall back to slower algo in this case).
    """

    # create mask
    mask = np.ones_like(img)
    mask[np.abs(img) < np.finfo(float).eps] = 0

    # determine shift if needed
    col_sum = np.sum(mask, axis=0)
    row_sum = np.sum(mask, axis=1)

    # check for circular mask
    if np.max(col_sum) != np.max(row_sum) or 0 not in col_sum or 0 not in row_sum:
        # if image contains no valid mask then assume mask is the inscribed circle
        h, w = [val // 2 for val in mask.shape]
        l = (np.floor(np.min(mask.shape) / SQRT_2).astype(np.int16)) // 2
        return w - l, h - l, l * 2

    mask_size = np.max(col_sum)

    # find center of current mask
    col_shift = np.rint(np.median(np.where(col_sum == np.amax(col_sum)))).astype(
        np.int32
    )
    row_shift = np.rint(np.median(np.where(row_sum == np.amax(row_sum)))).astype(
        np.int32
    )

    # calculate difference from image center
    h, w = [val // 2 for val in mask.shape]
    col_shift = w - col_shift
    row_shift = h - row_shift

    # shift particle and mask to center of image
    img_shift = np.roll(img, shift=row_shift, axis=0)
    img_shift = np.roll(img_shift, shift=col_shift, axis=1)
    mask_shift = np.roll(mask, shift=row_shift, axis=0)
    mask_shift = np.roll(mask_shift, shift=col_shift, axis=1)

    # use pythagorean theorem to get half the length of largest box inside mask
    l = (np.floor(mask_size / SQRT_2).astype(np.int16)) // 2
    # img_box = img_shift[h - l : h + l, w - l : w + l]
    # mask_box = mask_shift[h - l : h + l, w - l : w + l]

    # return top left row, top left col, and square side length
    return w - l, h - l, l * 2


def get_inscribed_square(class_mrc, angle_deg=0, func=inscribed_square_fast):
    # for inscribed_square_fast, do cropping first THEN rotation
    # if you want to rotate first (losing fewer pixels), use inscribed_square_naive
    mrc = class_mrc.copy()
    r, c, size = func(mrc)
    cropped_mrc = mrc[r : r + size, c : c + size]
    mrc = rotate(cropped_mrc, angle_deg, reshape=False, cval=0)
    return mrc


def fill_upper_triangle(arr):
    i_upper = np.triu_indices(arr.shape[0])
    arr[i_upper] = arr.T[i_upper]


def calc_scores(
    class_avgs,
    angle_step=15,
    gt_vs_rest=False,
    gt_name="GT",
    gt_classes=None,
):
    angles = np.arange(0, 360, step=angle_step)
    n_imgs = sum(len(v) for v in class_avgs.values())
    scores = np.full((n_imgs, n_imgs), np.nan, dtype="float32")

    # iterate over all mrcs files
    for name_y, mrcs_y in class_avgs.items():

        # iterate over all mrcs files again (nxn correlation)
        for name_x, mrcs_x in class_avgs.items():
            log(f"processing correlations for {name_y} vs. {name_x}")

            if gt_vs_rest:
                if name_x == gt_name and name_y == gt_name:
                    log("skipping GT vs. GT")
                    continue
                if gt_name not in (name_x, name_y):
                    log("skipping non-GT vs. non-GT")
                    continue

            for j, img_y in enumerate(tqdm(mrcs_y)):
                img_y_scaled = standardize_arr(get_inscribed_square(img_y))
                img_y_ones = np.ones(img_y_scaled.shape)
                img_y_conj = np.flipud(np.fliplr(img_y_scaled)).conj()
                img_y_shape_prod = np.prod(img_y_scaled.shape)

                for k, img_x in enumerate(mrcs_x):
                    # if past diagonal, skip
                    if k > j:
                        continue

                    # skip if we didn't ask for this class
                    if gt_classes is not None and k not in gt_classes:
                        continue

                    # rotate img_x and calculate correlation with img_y
                    for ang in angles:
                        img_x_scaled = standardize_arr(get_inscribed_square(img_x, ang))
                        corr = fftconvolve(img_x_scaled, img_y_conj, mode=FFT_MODE)

                        # https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
                        tmp = (
                            fftconvolve(
                                np.square(img_x_scaled), img_y_ones, mode=FFT_MODE
                            )
                            - np.square(
                                fftconvolve(img_x_scaled, img_y_ones, mode=FFT_MODE)
                            )
                            / img_y_shape_prod
                        )

                        # remove small machine precision errors after subtraction
                        tmp[np.where(tmp < 0)] = 0

                        corr = corr / np.sqrt(tmp * np.sum(np.square(img_y_scaled)))

                        # remove any divisions by 0 or very close to 0
                        corr[np.where(np.logical_not(np.isfinite(corr)))] = 0

                        # find correlation max
                        center_xy = np.subtract(np.divide(corr.shape, 2.0), (1, 1))
                        max_val = np.max(corr)
                        brightest_spot = np.unravel_index(np.argmax(corr), corr.shape)
                        dist = np.linalg.norm(brightest_spot - center_xy)

                        # score formula
                        h, w = corr.shape[0], corr.shape[1]
                        diag_radius = math.sqrt(w**2 + h**2) / 2
                        score = max_val - dist / diag_radius

                        # compare score and save everything if better
                        best_score = scores[j, k]
                        if np.isnan(best_score) or score > best_score:
                            scores[j, k] = score

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script saves a correlation matrix file from input RELION 2D "
        "class average files."
    )
    parser.add_argument(
        "out_dir",
        help="Output directory in which to save correlations "
        "(will be created if it does not exist)",
    )
    parser.add_argument(
        "-m",
        help="Path(s) to input *.mrcs class average image stacks  "
        "(NOTE: *.mrcs files should be renamed from default and must have unique "
        "file names; file names will be used in the figure legend)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-n",
        help="Number of class averages to use (if available) from each *.mrcs "
        "file (default is all)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--angle_step",
        help="Angle in degrees by which to rotate during correlation (default is 15)",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--gt_name",
        help="Name of ground truth image stack in files passed to -m (default is 'GT')",
        type=str,
        default="GT",
    )
    parser.add_argument(
        "--gt_classes",
        help="Indices of classes in ground truth image stack to save (default is all)",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--gt_vs_rest",
        help="Only calculate ground truth (see --gt_name) vs. the rest of the pickers.",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )

    a = parser.parse_args()

    # validation and input handling
    if len(a.m) < 2:
        log("Must have at least two class average files to run correlation", lvl=2)
    a.m = [Path(p).resolve() for p in np.atleast_1d(a.m)]
    if not all(p.is_file() for p in a.m):
        log(f"bad mrcs paths", lvl=2)
    a.out_dir = Path(a.out_dir).resolve()
    if not a.out_dir.is_dir():
        os.makedirs(a.out_dir, exist_ok=True)
    if a.gt_name not in [x.stem for x in a.m]:
        log(f"GT name '{a.gt_name}' must match an input mrcs file stem", lvl=2)

    log("reading mrcs files")
    class_avgs = read_class_avgs(a.m, n_max_classes=a.n)
    log(f"mrcs shapes", [mrc.shape for mrc in class_avgs.values()])

    scores = calc_scores(
        class_avgs,
        angle_step=a.angle_step,
        gt_vs_rest=a.gt_vs_rest,
        gt_name=a.gt_name,
        gt_classes=a.gt_classes,
    )

    suffix = "all" if a.gt_classes is None else "_".join(str(x) for x in a.gt_classes)
    out_path = a.out_dir / f"scores_{suffix}.npy"
    log(f"saving scores to {out_path}")

    # fill below diagonal and save to disk
    fill_upper_triangle(scores)
    np.save(out_path, scores)

    log("done.")
