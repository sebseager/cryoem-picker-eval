import sys
import os
import math
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings
import numpy as np
from collections import OrderedDict
from itertools import combinations, combinations_with_replacement
import pandas as pd
import random
from scipy.signal import fftconvolve
from scipy.ndimage.interpolation import rotate
from skimage.metrics import structural_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import patches
from matplotlib import patheffects
import seaborn as sns
from coord_converter import star_to_df


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


def _format_axes(ax):
    ax.spines["top"].set_visible(False)  # hide all axis spines
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().set_ticks([])  # hide ticks and labels
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])


def read_mrc(path, mmap=False):
    """
    Use the mrcfile module to read data from micrograph at given path. Returns
    numerical array containing micrograph data.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mmap:
            with mrcfile.mmap(path, mode="r", permissive=True) as f:
                mrc = f.data
        else:
            with mrcfile.open(path, mode="r", permissive=True) as f:
                mrc = f.data

    return mrc


def read_class_avgs(mrcs_paths, star_paths):
    class_avgs = OrderedDict(
        [
            (Path(m).stem, {"mrcs": read_mrc(m), "star": star_to_df(s)})
            for m, s in zip(mrcs_paths, star_paths)
        ]
    )

    # reorder class avg stacks by particle distribution if STAR available
    for stem, data in class_avgs.items():
        if "star" not in data:
            continue

        distr = data["star"]["_rlnClassDistribution"].to_numpy()
        sorted_idx = np.argsort(distr)[::-1]
        class_avgs[stem]["sorted_idx"] = sorted_idx
        class_avgs[stem]["sorted_distr"] = np.take_along_axis(
            distr, sorted_idx, axis=None
        )
        # we use [:, None, None] to broadcast index to 3D like mrcs stack
        class_avgs[stem]["mrcs"] = np.take_along_axis(
            data["mrcs"], sorted_idx[:, None, None], axis=0
        )

    class_names = [Path(m).stem for m in mrcs_paths]

    return class_avgs, class_names


def load_np_files(out_dir, do_recalc_all=False):
    """
    Loads numpy arrays in the given output directory.
    """

    names = ["max_scores", "best_angles", "best_corrs", "global_max_points"]
    paths = {n: out_dir / f"corr_{n}.npy" for n in names}
    arrs = {n: None for n in names}

    if do_recalc_all:
        _log("Recalculating all correlations")
        return arrs

    for n, p in paths.items():
        try:
            arrs[p] = np.load(p, allow_pickle=True)
            _log(f"Found existing data for: {n}")
        except FileNotFoundError:
            pass

    return arrs


def fill_below_diag_inplace(arr):
    for idx, item in np.ndenumerate(arr):
        if idx[0] > idx[1]:
            arr[idx[0], idx[1]] = arr[idx[1], idx[0]]


def standardize_arr(arr):
    # https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    a = np.array(arr)
    return (a - np.mean(a)) / (np.std(a) * len(a))


def find_largest_square(arr):
    # https://www.geeksforgeeks.org/maximum-size-sub-matrix-with-all-1s-in-a-binary-matrix/

    m = np.array(arr)

    # fix strange float precision things
    m[np.abs(m) < np.finfo(np.float).eps] = 0
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


def find_largest_square_fast(arr):
    # NOTE: DOES NOT WORK ON ALREADY-ROTATED ARRAYS
    # (WILL FALL BACK TO SLOWER find_largest_square)

    # copy arr and account for floating point error
    img = arr.copy()
    img[np.abs(img) < np.finfo(np.float).eps] = 0

    # calculate mask
    h, w = arr.shape
    mask = np.zeros((h, w))
    mask[np.where(img != 0)] = 1

    # determine shift if needed
    # assumes rows/columns present of only mask create col/row zero counts
    col_sum, row_sum = np.sum(mask, axis=0), np.sum(mask, axis=1)

    # determine minimum col/row
    col_shift = np.argmin(col_sum)

    # determine shift to center
    col_shift = 0 if not col_sum[col_shift + 1] == 0 else w - (col_shift + 1)
    row_shift = np.argmin(row_sum)
    row_shift = 0 if not row_sum[row_shift + 1] == 0 else h - (row_shift + 1)

    # shift class and mask images
    img_shift = np.roll(img, shift=row_shift, axis=0)
    img_shift = np.roll(img_shift, shift=col_shift, axis=1)
    mask_shift = np.roll(mask, shift=row_shift, axis=0)
    mask_shift = np.roll(mask_shift, shift=col_shift, axis=1)

    # calculate center of shifted mask
    idx = np.where(mask_shift != 0)
    row_min, row_max = np.min(idx[0]), np.max(idx[0])
    col_min, col_max = np.min(idx[1]), np.max(idx[1])

    if not (row_min == col_min and row_max == col_max):
        _log("mask not centered - falling back to slower method", 1)
        return find_largest_square(arr)

    # circle diameter
    d = (row_max - row_min) + 1

    # pythagorean theorem to get side length
    l = np.floor(np.sqrt((d ** 2) / 2)).astype(np.int16)

    # crop inscribed square
    i, j = np.argmax(np.sum(mask_shift, axis=1) - l >= 0), np.argmax(
        np.sum(mask_shift, axis=0) - l >= 0
    )

    # return top left row, top left col, and square side length
    return i, j, l


def inscribed_square_from_mask(class_mrc, angle_deg=0, func=find_largest_square_fast):
    # for find_largest_square_fast, do cropping first THEN rotation
    # if you want to rotate first (losing fewer pixels), use find_largest_square
    mrc = class_mrc.copy()
    r, c, size = func(mrc)
    cropped_mrc = mrc[r : r + size, c : c + size]
    mrc = rotate(cropped_mrc, angle_deg, reshape=False, cval=0)
    return mrc


def build_corrs(all_imgs, angle_step, do_noise_zeros=False):
    angles = np.arange(0, 360, step=angle_step)

    # make all-vs-all lists, capping mrcs lengths at num_avgs
    num_imgs = len(all_imgs)

    # all-vs-all matrices to hold various correlation results
    max_scores = np.full((num_imgs, num_imgs), -1, dtype="float")
    best_angles = np.full((num_imgs, num_imgs), 0, dtype="float")
    best_corrs = np.full((num_imgs, num_imgs), None, dtype="object")
    global_max_points = np.full((num_imgs, num_imgs, 2), -1, dtype="float")

    # j is rows, k is cols
    for j, img_y in enumerate(tqdm(all_imgs)):
        # normalize img_y
        img_y_scaled = standardize_arr(inscribed_square_from_mask(img_y))
        img_y_ones = np.ones(img_y_scaled.shape)
        img_y_conj = np.flipud(np.fliplr(img_y_scaled)).conj()

        for k, img_x in enumerate(all_imgs):
            # skip if we're below the matrix diagonal
            if j > k:
                continue

            # seed with coordinates to get reproducible results
            # replace mask (zeros) with uniform random values between
            # image min and max
            if do_noise_zeros:
                np.random.seed(int("%s%s0" % (j, k)))
                img_x[img_x == 0.0] = np.random.uniform(
                    img_x.min(), img_x.max(), np.count_nonzero(img_x == 0)
                )
                np.random.seed(int("%s%s1" % (j, k)))
                img_y[img_y == 0.0] = np.random.uniform(
                    img_y.min(), img_y.max(), np.count_nonzero(img_y == 0)
                )
                np.random.seed(None)  # re-seed from /dev/urandom

            # build 2d correlations for each angle offset
            corrs = []
            for theta in angles:
                img_x_scaled = standardize_arr(inscribed_square_from_mask(img_x, theta))
                corr = fftconvolve(img_x_scaled, img_y_conj, mode="full")

                # https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
                tmp = fftconvolve(
                    np.square(img_x_scaled), img_y_ones, mode="full"
                ) - np.square(fftconvolve(img_x_scaled, img_y_ones, mode="full")) / (
                    np.prod(img_y_scaled.shape)
                )

                # remove small machine precision errors after subtraction
                tmp[np.where(tmp < 0)] = 0

                corr = corr / np.sqrt(tmp * np.sum(np.square(img_y_scaled)))

                # remove any divisions by 0 or very close to 0
                corr[np.where(np.logical_not(np.isfinite(corr)))] = 0
                corrs.append(corr)

            # scoring
            scores = []
            max_points = []
            for corr in corrs:
                # subtract to get index
                center_xy = np.subtract(np.divide(corr.shape, 2.0), (1, 1))
                max_corr = np.max(corr)
                max_xys = list(zip(*np.where(corr == max_corr)))
                if len(max_xys) > 1:
                    print("found multiple maxima - keeping first")
                brightest_spot = np.unravel_index(np.argmax(corr), corr.shape)
                max_points.append(brightest_spot)
                dist = np.linalg.norm(brightest_spot - center_xy)

                # score formula
                h, w = corr.shape[0], corr.shape[1]
                diag_radius = math.sqrt(w ** 2 + h ** 2) / 2
                score = max_corr - dist / diag_radius
                scores.append(score)

            try:
                scores = [-1 if s is None else s for s in scores]
                score = np.max(scores)
                i_score = scores.index(score)

                max_scores[j, k] = score
                best_angles[j, k] = angles[i_score]
                best_corrs[j, k] = corrs[i_score]
                global_max_points[j, k] = max_points[i_score]

            except (NameError, IndexError) as e:
                _log("couldn't assign max corr. score\n" + str(e), 1)

    return max_scores, best_angles, best_corrs, global_max_points


def plot_corr_previews(corr_arrs, class_names, num_avgs):
    # get all combinations of two pickers (without matching a picker to itself)
    class_name_combos = [
        x
        for x in list(map(dict, combinations_with_replacement(class_names, 2)))
        if x[0] != x[1]
    ]

    # create figure
    corr_fig = plt.figure(figsize=(5.35 * len(class_name_combos), 5), dpi=500)
    corr_outer_grid = gs.GridSpec(ncols=len(class_name_combos), nrows=1)
    extra_artists = []

    # iterate over each picker combo
    for combo_i, combo in enumerate(tqdm(class_name_combos)):
        pckr_x_i = all_class_names.index(combo[1])
        pckr_y_i = all_class_names.index(combo[0])
        imgs_x_slice = [pckr_x_i * num_avgs, pckr_x_i * num_avgs + num_avgs]
        imgs_y_slice = [pckr_y_i * num_avgs, pckr_y_i * num_avgs + num_avgs]

        imgs_x = all_imgs[slice(*imgs_x_slice)]
        imgs_y = all_imgs[slice(*imgs_y_slice)]

        # skip if slice was empty for either x or y axis
        if not imgs_x or not imgs_y:
            continue

        # make grid for this combo
        corr_inner_top = corr_outer_grid[0, combo_i].subgridspec(
            len(imgs_x) + 1, len(imgs_y) + 1, wspace=0.05, hspace=0.05
        )

        # label positions
        topgrid_pos = corr_inner_top.get_grid_positions(corr_fig)
        topgrid_top = max(topgrid_pos[1])  # [1] is row top positions
        topgrid_midy = (
            topgrid_top + min(topgrid_pos[0])
        ) / 2  # [0] is row bottom positions
        topgrid_left = min(topgrid_pos[2])  # [2] is col left positions
        topgrid_midx = (
            topgrid_left + max(topgrid_pos[3])
        ) / 2  # [3] is col right positions

        # top axis title
        extra_artists.append(
            corr_fig.text(
                topgrid_midx,
                topgrid_top + 0.01,
                combo[1],
                ha="center",
                va="bottom",
                size="medium",
            )
        )

        # left axis title
        extra_artists.append(
            corr_fig.text(
                topgrid_left - 0.008,
                topgrid_midy,
                combo[0],
                ha="right",
                va="center",
                rotation=90,
                size="medium",
            )
        )

        # place preview images
        for j, y in enumerate([None] + list(range(*imgs_y_slice))):
            for k, x in enumerate([None] + list(range(*imgs_x_slice))):

                try:
                    # create and format current cell
                    ax_top = corr_fig.add_subplot(corr_inner_top[j, k])
                    format_axes(ax_top)

                    # place corr images on axes
                    if y is None and x is not None:
                        ax_top.imshow(
                            inscribed_square_from_mask(imgs_x[k - 1]), cmap=GRAY_CMAP
                        )
                    elif x is None and y is not None:
                        ax_top.imshow(
                            inscribed_square_from_mask(imgs_y[j - 1]), cmap=GRAY_CMAP
                        )

                    # format cells
                    elif x is not None and y is not None:
                        corr, score, theta = (
                            corr_arrs["best_corrs"][y, x],
                            corr_arrs["max_scores"][y, x],
                            corr_arrs["best_angles"][y, x],
                        )

                        if corr is not None:
                            ax_top.imshow(corr, cmap=GRAY_CMAP)

                            # plot circles around brightest spot
                            ax_top.scatter(
                                *zip(corr_arrs["global_max_points"][y, x][::-1]),
                                s=18,
                                facecolors="None",
                                edgecolors="r",
                                linewidths=0.6,
                            )
                            ax_top.scatter(
                                corr.shape[0] / 2 - 1,
                                corr.shape[1] / 2 - 1,
                                s=18,
                                facecolors="None",
                                edgecolors="g",
                                linewidths=0.6,
                            )

                except IndexError:
                    # in case of fewer than num_avgs classes
                    continue

                # add axes
                corr_fig.add_subplot(ax_top)

    plt.show()


def plot_heatmap(all_imgs, class_avgs):
    num_imgs = len(all_imgs)

    # create figure
    corr_fig = plt.figure(figsize=(10, 10), dpi=300)
    corr_outer_grid = gs.GridSpec(ncols=1, nrows=1)
    extra_artists = []

    # +2 is for colorbar and preview axis
    total_axis_len = num_imgs + 2

    # make grid
    corr_inner_top = corr_outer_grid[0, 0].subgridspec(
        total_axis_len, total_axis_len, wspace=0.05, hspace=0.05
    )

    # label positions
    topgrid_pos = corr_inner_top.get_grid_positions(corr_fig)
    topgrid_top = max(topgrid_pos[1])  # [1] is row top positions
    topgrid_midy = (
        topgrid_top + min(topgrid_pos[0])
    ) / 2  # [0] is row bottom positions
    topgrid_left = min(topgrid_pos[2])  # [2] is col left positions
    topgrid_midx = (
        topgrid_left + max(topgrid_pos[3])
    ) / 2  # [3] is col right positions

    # plot ith class avg image on both axes
    # since this analysis is all-vs-all
    for i in tqdm(range(num_imgs)):

        # plot images in reverse order
        img_i = num_imgs - i - 1
        img = all_imgs[img_i].copy()

        # class image label
        class_label_num = (img_i % num_avgs) + 1

        # current picker name
        pckr_name = ""
        tmp_i = i
        for d in reversed(class_avgs.values()):
            if tmp_i < len(d["mrcs"]):
                if tmp_i == math.ceil(num_avgs / 2.0):
                    pckr_name = d["picker"]
                break
            else:
                tmp_i -= len(d["mrcs"])

        # plot class avg images on both axes
        for j, ax in enumerate(
            (
                corr_fig.add_subplot(corr_inner_top[num_imgs, i + 1]),
                corr_fig.add_subplot(corr_inner_top[i, 0]),
            )
        ):
            ax.imshow(it.inscribed_square_from_mask(img), cmap=GRAY_CMAP)

            corr_fig.add_subplot(ax)
            format_axes(ax)

            rot = 180 if j == 0 else 90

            # j == 0 is horizontal axis
            # coords are (x, y)
            xytext = (6, -6) if j == 0 else (-8, 6)

            # class number
            ax.annotate(
                class_label_num,
                xy=(0, 0),
                xycoords="axes fraction",
                xytext=xytext,
                textcoords="offset points",
                ha="center",
                va="center",
                rotation=rot,
                fontsize=6,
            )

            xytext = (6, -20) if j == 0 else (-20, 8)

            # picker name
            if pckr_name != "":
                ax.annotate(
                    pckr_name,
                    xy=(0, 0),
                    xycoords="axes fraction",
                    xytext=xytext,
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    rotation=rot,
                    fontsize=8,
                )

    # format heatmap
    ax_heatmap = corr_fig.add_subplot(corr_inner_top[0:-2, 1:-1])
    ax_colorbar = corr_fig.add_subplot(
        corr_inner_top[: -(math.floor(total_axis_len * 0.8)), -1].subgridspec(1, 4)[1:6]
    )

    format_axes(ax_heatmap)
    format_axes(ax_colorbar)

    # heatmap mask
    mask = np.ones_like(max_scores, dtype=bool)
    mask[np.tril_indices_from(mask)] = False

    # heatmap proper
    # np.flip flips both axes by default
    lo = math.floor(max_scores.min() * 10) / 10
    hi = math.ceil(max_scores.max() * 10) / 10
    cmap = plt.get_cmap("coolwarm").copy()
    heatmap = sns.heatmap(
        np.flip(max_scores),
        mask=mask,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        linewidths=0.35,
        ax=ax_heatmap,
        cbar_ax=ax_colorbar,
        cbar_kws={"ticks": [lo, hi]},
    )

    # formatting
    heatmap.set(xticklabels=[], yticklabels=[], facecolor="white")
    ax_heatmap.tick_params(left=False, bottom=False)
    ax_colorbar.tick_params(width=0.8, labelsize=12, rotation=90)

    corr_fig.add_subplot(ax_heatmap)
    corr_fig.add_subplot(ax_colorbar)

    plt.show()


def plot_class_distributions(class_avgs):
    n_cols = len(filtered_class_avg_dicts)
    dist_fig = plt.figure(figsize=(5.35 * n_cols, 3), dpi=200)
    dist_grid = gs.GridSpec(ncols=n_cols, nrows=1)

    for i, class_dict in enumerate(class_avgs.values()):
        ax = dist_fig.add_subplot(dist_grid[0, i])
        xs = list(range(1, len(class_dict["sorted_distr"]) + 1))
        ys = class_dict["sorted_distr"]
        ax.bar(x=xs, height=ys, color="gray")
        ax.set_title(class_dict["picker"])
        ax.set_xticks(xs)

    plt.show(dist_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script builds a correlation matrix from input RELION 2D "
        "class average files."
    )
    parser.add_argument(
        "-m",
        help="Path(s) to input *.mrcs class average image stacks  "
        "(NOTE: model.star files should be renamed and must have unique file names; "
        "file names will be used in the figure legend)",
        nargs="+",
    )
    parser.add_argument(
        "-s",
        help="Path(s) to input class average metadata *.star files "
        "(NOTE: this argument can be left empty, but if files are provided they must "
        "correspond in order and number with files passed to -m)",
    )
    parser.add_argument(
        "out_dir",
        help="Output directory in which to store generated temporary data and final "
        "heatmap (will be created if it does not exist)",
        nargs=1,
    )
    parser.add_argument(
        "-n",
        help="Number of class averages to use (if available) from each model.star "
        "file (default is 10)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--angle_step",
        help="Angle in degrees by which to rotate during correlation (default is 15)",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--noise",
        help="Replace RELION zero mask with random noise",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence info-level output",
    )

    a = parser.parse_args()

    # validation
    if a.s is not None and len(a.m) != len(a.s):
        _log("Number of class average files and STAR files must match", 2)
    if len(a.m) < 2:
        _log("Must have at least two class average files to run correlation", 2)

    # normalize paths
    a.m = [Path(p).resolve() for p in np.atleast_1d(a.m)]
    if not all(p.is_file() for p in a.m):
        _log(f"bad mrcs paths", 2)
    a.s = [Path(p).resolve() for p in np.atleast_1d(a.s)]
    if not all(p.is_file() for p in a.s):
        _log(f"bad mrcs paths", 2)
    a.out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        os.makedirs(out_dir, exist_ok=True)

    class_avgs, class_names = read_class_avgs(a.m, a.s)

    corr_arrs = load_np_files(a.out_dir, do_recalc_all=a.force)

    all_imgs = [avg for v in class_avgs.values() for avg in v["mrcs"][: a.n]]

    do_recalc = any(v is None for v in corr_arrs.values())

    if do_recalc:
        _log("calculating correlations")

        (
            corr_arrs["max_scores"],
            orr_arrs["best_angles"],
            corr_arrs["best_corrs"],
            corr_arrs["global_max_points"],
        ) = calc_corrs(all_imgs, a.noise)

        # fill missing values below main diagonal
        for arr in corr_arrs.values():
            fill_below_diag_inplace(arr)

        # save to disk
        for n, arr in corr_arrs.items():
            np.save(out_dir / f"corr_{n}.npy", arr)

    plot_corr_previews(corr_arrs, class_names, a.n)
    plot_heatmap(all_imgs, class_avgs)
    plot_class_distributions(class_avgs)

    _log("done.", 0, quiet=a.quiet)
