import sys
import os
import math
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings
import numpy as np
import mrcfile
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
from common import log
from consts import *

sqrt_2 = math.sqrt(2)


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
    if not star_paths:
        star_paths = [None for _ in mrcs_paths]

    class_avgs = OrderedDict(
        [
            (
                Path(m).stem,
                {
                    "mrcs": read_mrc(m),
                    "star": None if not s else star_to_df(s),
                    "name": Path(m).stem,
                },
            )
            for m, s in zip(mrcs_paths, star_paths)
        ]
    )

    # reorder class avg stacks by particle distribution if STAR available
    for stem, data in class_avgs.items():
        if "star" not in data or data["star"] is None:
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
        log("recalculating all correlations")
        return arrs

    for n, p in paths.items():
        try:
            arrs[n] = np.load(p, allow_pickle=True)
            log(f"found existing data for: {n}")
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


def find_largest_square_fast(arr):
    # NOTE: does not work on already-rotated arrays
    #       since this may make mask not circular anymore
    #       (will fall back to slower algo in this case)

    # copy arr and account for floating point error
    img = arr.copy()
    img[np.abs(img) < np.finfo(float).eps] = 0

    # calculate mask
    h, w = arr.shape
    mask = np.zeros((h, w))
    mask[np.where(img != 0)] = 1

    # determine shift if needed
    # assumes rows/columns present of only mask create col/row zero counts
    col_sum, row_sum = np.sum(mask, axis=0), np.sum(mask, axis=1)

    # determine shift to center
    col_shift = np.argmin(col_sum)
    if col_shift + 1 < len(col_sum) and col_sum[col_shift + 1] != 0:
        col_shift = 0
    else:
        col_shift = w - (col_shift + 1)

    row_shift = np.argmin(row_sum)
    if row_shift + 1 < len(row_sum) and row_sum[row_shift + 1] != 0:
        row_shift = 0
    else:
        row_shift = h - (row_shift + 1)

    # ORIG: WE SHOULD CHECK FOR OUT OF BOUNDS
    # col_shift = 0 if not col_sum[col_shift + 1] == 0 else w - (col_shift + 1)
    # row_shift = 0 if not row_sum[row_shift + 1] == 0 else h - (row_shift + 1)

    # shift class and mask images
    img_shift = np.roll(img, shift=row_shift, axis=0)
    img_shift = np.roll(img_shift, shift=col_shift, axis=1)
    mask_shift = np.roll(mask, shift=row_shift, axis=0)
    mask_shift = np.roll(mask_shift, shift=col_shift, axis=1)

    # calculate center of shifted mask
    idx = np.where(mask_shift != 0)
    try:
        row_min, row_max = np.min(idx[0]), np.max(idx[0])
        col_min, col_max = np.min(idx[1]), np.max(idx[1])
    except ValueError:
        log("array is empty - no mask found")
        return 0, 0, 0

    if not (row_min == col_min and row_max == col_max):
        log("mask not centered - falling back to slower method", lvl=1)
        return find_largest_square(arr)

    # circle diameter
    d = (row_max - row_min) + 1

    # pythagorean theorem to get side length
    l = np.floor(d / sqrt_2).astype(np.int16)

    # crop inscribed square
    # i = np.argmax(np.sum(mask_shift, axis=1) - l >= 0)
    # j = np.argmax(np.sum(mask_shift, axis=0) - l >= 0)

    # instead of above, we know the square will be centered
    # so we can just crop half the side length from the center
    half_w, half_h, half_l = w // 2, h // 2, l // 2
    i, j = half_h - half_l, half_w - half_l
    i = 0 if i < 0 else i
    j = 0 if j < 0 else j

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
                log("couldn't assign max corr. score\n" + str(e), lvl=1)

    return max_scores, best_angles, best_corrs, global_max_points


def id_pckr_by_idx(idx, class_avgs, num_max_avgs=None):
    """Given an index into all_imgs and class average data, return
    the picker name, number of averages, and start index. We want to figure
    out where each picker starts in all_imgs.
    """

    tmp_i = 0
    for pckr_name, pckr in class_avgs.items():
        n = len(pckr["mrcs"])
        if num_max_avgs is not None and n > num_max_avgs:
            n = num_max_avgs
        if tmp_i + n > idx:
            return (pckr_name, n, tmp_i)
        tmp_i += n

    log("couldn't find picker for idx %s" % idx, lvl=1)


def get_pckr_idx_range(name, class_avgs, num_max_avgs=None):
    """Given a picker name and class average data, return the
    start and end indices of the picker in all_imgs.
    """

    tmp_i = 0
    for pckr_name, pckr in class_avgs.items():
        n = len(pckr["mrcs"])
        if num_max_avgs is not None and n > num_max_avgs:
            n = num_max_avgs
        if pckr_name == name:
            return (tmp_i, tmp_i + n)
        tmp_i += n

    log("picker %s not found in class averages" % name, lvl=2)


def plot_corr_previews(out_dir, corr_arrs, class_names, num_avgs):
    # get all combinations of two pickers (without matching a picker to itself)
    class_name_combos = [
        x for x in list(combinations_with_replacement(class_names, 2)) if x[0] != x[1]
    ]

    # create figure
    corr_fig = plt.figure(figsize=(5.35 * len(class_name_combos), 5), dpi=600)
    corr_outer_grid = gs.GridSpec(ncols=len(class_name_combos), nrows=1)
    extra_artists = []

    # iterate over each picker combo
    for combo_i, combo in enumerate(tqdm(class_name_combos)):
        pckr_x_name = combo[1]
        pckr_y_name = combo[0]

        imgs_x_slice = get_pckr_idx_range(pckr_x_name, class_avgs, num_avgs)
        imgs_y_slice = get_pckr_idx_range(pckr_y_name, class_avgs, num_avgs)

        # pckr_x_i = class_names.index(combo[1])
        # pckr_y_i = class_names.index(combo[0])
        # imgs_x_slice = [pckr_x_i * num_avgs, pckr_x_i * num_avgs + num_avgs]
        # imgs_y_slice = [pckr_y_i * num_avgs, pckr_y_i * num_avgs + num_avgs]

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
                    _format_axes(ax_top)

                    # place corr images on axes
                    if y is None and x is not None:
                        ax_top.imshow(
                            inscribed_square_from_mask(imgs_x[k - 1]), cmap=plt.cm.gray
                        )
                    elif x is None and y is not None:
                        ax_top.imshow(
                            inscribed_square_from_mask(imgs_y[j - 1]), cmap=plt.cm.gray
                        )

                    # format cells
                    elif x is not None and y is not None:
                        corr, score, theta = (
                            corr_arrs["best_corrs"][y, x],
                            corr_arrs["max_scores"][y, x],
                            corr_arrs["best_angles"][y, x],
                        )

                        if corr is not None:
                            ax_top.imshow(corr, cmap=plt.cm.gray)

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

    plt.savefig(out_dir / "corr_previews.png", bbox_extra_artists=extra_artists)


def plot_heatmap(
    out_dir,
    all_imgs,
    class_avgs,
    num_avgs,
    max_scores,
    use_ax_nums=True,
    clip_to=(0, 1),
    specify_classes=None,
):
    """
    Plot heatmap of correlation scores. Set use_ax_nums to False to skip plotting
    class indices on the axes (use for large heatmaps). Use clip_to to manually
    set the min and max of the colorbar (default is 0-1). Set specify_classes to
    a dict of the form {"picker_name": [class_indices]} to plot only the specified
    classes for each picker.
    """

    num_imgs = len(all_imgs)

    # create figure
    corr_fig = plt.figure(figsize=(10, 10), dpi=800)
    corr_outer_grid = gs.GridSpec(ncols=1, nrows=1)
    extra_artists = []

    # +2 is for colorbar and preview axis
    total_axis_len = num_imgs + 2

    # make grid
    corr_inner_top = corr_outer_grid[0, 0].subgridspec(
        total_axis_len, total_axis_len, wspace=0.05, hspace=0.05
    )

    # reverse class_avgs OrderedDict since we plot in reverse order
    class_avgs_rev = OrderedDict(reversed(list(class_avgs.items())))

    # label positions
    topgrid_pos = corr_inner_top.get_grid_positions(corr_fig)
    topgrid_top = max(topgrid_pos[1])  # [1] is row top pos
    topgrid_midy = (topgrid_top + min(topgrid_pos[0])) / 2  # [0] is row bottom pos
    topgrid_left = min(topgrid_pos[2])  # [2] is col left pos
    topgrid_midx = (topgrid_left + max(topgrid_pos[3])) / 2  # [3] is col right pos

    # plot ith class avg image on both axes since this analysis is all-vs-all
    for i in tqdm(range(num_imgs)):
        # figure out which picker we're plotting
        this_pckr_name, this_pckr_num_avgs, this_pckr_start_idx = id_pckr_by_idx(
            i, class_avgs_rev, num_avgs
        )
        # class image label
        class_label_num = this_pckr_num_avgs - (i - this_pckr_start_idx)

        if specify_classes is not None and this_pckr_name in specify_classes:
            if class_label_num - 1 not in specify_classes[this_pckr_name]:
                continue

        # plot images in reverse order
        img_i = num_imgs - i - 1
        img = all_imgs[img_i].copy()

        # plot class avg images on both axes
        for j, ax in enumerate(
            (
                corr_fig.add_subplot(corr_inner_top[num_imgs, i + 1]),
                corr_fig.add_subplot(corr_inner_top[i, 0]),
            )
        ):
            ax.imshow(inscribed_square_from_mask(img), cmap=plt.cm.gray)

            corr_fig.add_subplot(ax)
            _format_axes(ax)

            rot = 180 if j == 0 else 90

            # j == 0 is horizontal axis
            # coords are (x, y)

            # class number
            if use_ax_nums:
                ax.annotate(
                    class_label_num,
                    xy=(0, 0),
                    xycoords="axes fraction",
                    xytext=(6, -6) if j == 0 else (-8, 6),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    rotation=rot,
                    fontsize=6,
                )

            # picker name
            if this_pckr_name is not None and class_label_num == 1:
                ax.annotate(
                    this_pckr_name,
                    xy=(0, 0),
                    xycoords="axes fraction",
                    xytext=(8, -25) if j == 0 else (-25, 4),
                    textcoords="offset points",
                    ha="right" if j == 0 else "left",
                    va="bottom",
                    rotation=rot,
                    fontsize=8,
                )

    # format heatmap
    ax_heatmap = corr_fig.add_subplot(corr_inner_top[0:-2, 1:-1])
    ax_colorbar = corr_fig.add_subplot(
        corr_inner_top[: -(math.floor(total_axis_len * 0.8)), -1].subgridspec(1, 4)[1:6]
    )

    _format_axes(ax_heatmap)
    _format_axes(ax_colorbar)

    # any correlations < 0 are set to 0, any correlations > 1 are set to 1
    max_scores = np.clip(max_scores, clip_to[0], clip_to[1])

    # remove any rows/cols not included in specify_classes
    if specify_classes is not None:
        all_incl_idxs = []
        for pckr_name, incl_idxs in specify_classes.items():
            rng = get_pckr_idx_range(pckr_name, class_avgs, num_avgs)
            all_incl_idxs.extend([rng[0] + i for i in incl_idxs])

        mask = np.ones(max_scores.shape[0], dtype=bool)
        mask[all_incl_idxs] = False
        max_scores = np.delete(max_scores, mask, axis=0)

        mask = np.ones(max_scores.shape[1], dtype=bool)
        mask[all_incl_idxs] = False
        max_scores = np.delete(max_scores, mask, axis=1)

    # heatmap mask
    mask = np.ones_like(max_scores, dtype=bool)
    mask[np.tril_indices_from(mask)] = False

    # heatmap proper
    # np.flip flips both axes by default
    to_plot = np.flip(max_scores)
    lo = math.floor(max_scores.min() * 10) / 10
    hi = math.ceil(max_scores.max() * 10) / 10
    cmap = plt.get_cmap("coolwarm").copy()
    heatmap = sns.heatmap(
        to_plot,
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

    plt.savefig(out_dir / "heatmap.png", bbox_extra_artists=extra_artists)


def plot_class_distributions(out_dir, class_avgs):
    n_cols = len(class_avgs)
    dist_fig = plt.figure(figsize=(5.35 * n_cols, 3), dpi=800)
    dist_grid = gs.GridSpec(ncols=n_cols, nrows=1)

    if all(v["star"] is None for v in class_avgs.values()):
        log("skipping class average distribution - no STAR files provided")
        return

    for i, class_dict in enumerate(class_avgs.values()):
        if "sorted_distr" not in class_dict:
            continue
        ax = dist_fig.add_subplot(dist_grid[0, i])
        xs = list(range(1, len(class_dict["sorted_distr"]) + 1))
        ys = class_dict["sorted_distr"]
        ax.bar(x=xs, height=ys, color="gray")
        ax.set_title(class_dict["name"])
        ax.set_xticks(xs)

    plt.savefig(out_dir / "class_avg_dists.png")


def plot_max_score_hist(
    out_dir, max_scores, class_avgs, gt_name="GT", num_avgs=None, clip_to=(0, 1)
):
    # find each non-ground-truth class avg's best score against ground truth

    # any correlations < 0 are set to 0, any correlations > 1 are set to 1
    max_scores = np.clip(max_scores, clip_to[0], clip_to[1])

    # select GT-vs-all scores
    gt_start, gt_end = get_pckr_idx_range(gt_name, class_avgs, num_avgs)
    like_heatmap = np.flip(max_scores)
    l = like_heatmap.shape[0]

    mask = np.ones_like(like_heatmap, dtype=bool)  # make all-True mask
    mask[l - gt_end : l - gt_start, :] = False  # make GT-vs-all scores False
    mask[:, l - gt_end : l - gt_start] = True  # make GT-vs-GT scores True again

    # apply mask to scores
    like_heatmap[mask] = np.nan

    # get max score for each class avg against GT
    # ignore RuntimeWarning
    with warnings.catch_warnings():
        # nanmax throws RuntimeWarning if all scores are nan
        warnings.simplefilter("ignore")
        try:
            # find max with ground truth for each class average (down columns)
            maxes = np.nanmax(like_heatmap, axis=0)
        except ValueError:
            log("internal error (scores array empty) - skipping max score histogram")
            return

    # plot histogram
    hist_fig, hist_ax = plt.subplots(figsize=(10, 6), dpi=800)

    # reverse class_avgs OrderedDict since we plot in reverse order
    class_avgs_rev = OrderedDict(reversed(list(class_avgs.items())))

    for i, pckr_name in enumerate(class_avgs_rev.keys()):
        pckr_slice = get_pckr_idx_range(pckr_name, class_avgs, num_avgs)
        try:
            slice_start, slice_end = pckr_slice
            inv_pckr_slice = slice(l - slice_end, l - slice_start)
            y, edges = np.histogram(maxes[inv_pckr_slice], bins=40)
        except ValueError:
            continue  # skip all-nan slices
        centers = 0.5 * (edges[1:] + edges[:-1])
        plt.plot(centers, y, color=PICKER_COLORS[i], label=pckr_name)

    hist_ax.set_xlabel("Max Correlation Score")
    hist_ax.set_ylabel("Frequency")
    hist_ax.legend(loc="best", frameon=False)
    plt.savefig(out_dir / "max_score_hist.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script builds a correlation matrix from input RELION 2D "
        "class average files."
    )
    parser.add_argument(
        "out_dir",
        help="Output directory in which to store generated temporary data and final "
        "heatmap (will be created if it does not exist)",
    )
    parser.add_argument(
        "-m",
        help="Path(s) to input *.mrcs class average image stacks  "
        "(NOTE: *.mrcs files should be renamed from default and must have unique "
        "file names; file names will be used in the figure legend)",
        nargs="+",
    )
    parser.add_argument(
        "-s",
        help="Path(s) to input class average metadata *.star files "
        "(NOTE: this argument can be left empty, but if files are provided they must "
        "correspond in order and number with files passed to -m)",
        nargs="*",
        default=(),
    )
    parser.add_argument(
        "-n",
        help="Number of class averages to use (if available) from each *.mrcs "
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
        "--gt_name",
        help="Name of ground truth image stack in files passed to -m (default is 'GT')",
        type=str,
        default="GT",
    )
    parser.add_argument(
        "--noise",
        help="Replace RELION zero mask with random noise",
        action="store_true",
    )
    parser.add_argument(
        "--score_clip",
        help="Specify two decimal values between which to clip correlation scores "
        "(default is '0 1')",
        nargs=2,
        default=(0, 1),
        type=float,
    )
    parser.add_argument(
        "--hm_nums_off",
        help="Turn of image numbering on heatmap (for large plots)",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )

    a = parser.parse_args()

    # validation
    if a.s and len(a.m) != len(a.s):
        log("Number of class average files and STAR files must match", lvl=2)
    if len(a.m) < 2:
        log("Must have at least two class average files to run correlation", lvl=2)

    a.score_clip = tuple(a.score_clip)

    # normalize paths
    a.m = [Path(p).resolve() for p in np.atleast_1d(a.m)]
    if not all(p.is_file() for p in a.m):
        log(f"bad mrcs paths", lvl=2)
    a.s = [Path(p).resolve() for p in np.atleast_1d(a.s)]
    if not all(p.is_file() for p in a.s):
        log(f"bad mrcs paths", lvl=2)
    a.out_dir = Path(a.out_dir).resolve()
    if not a.out_dir.is_dir():
        os.makedirs(a.out_dir, exist_ok=True)

    # more validation
    if a.gt_name not in [x.stem for x in a.m]:
        log("Ground truth name must match one of the file stems passed in -m", lvl=2)

    class_avgs, class_names = read_class_avgs(a.m, a.s)

    log(f"*.mrcs shapes", [x["mrcs"].shape for x in class_avgs.values()])

    corr_arrs = load_np_files(a.out_dir, do_recalc_all=a.force)
    all_imgs = [avg for v in class_avgs.values() for avg in v["mrcs"][: a.n]]
    do_recalc = any(v is None for v in corr_arrs.values())

    if do_recalc:
        log("calculating correlations")

        (
            corr_arrs["max_scores"],
            corr_arrs["best_angles"],
            corr_arrs["best_corrs"],
            corr_arrs["global_max_points"],
        ) = build_corrs(all_imgs, a.angle_step, do_noise_zeros=a.noise)

        # fill missing values below main diagonal
        for arr in corr_arrs.values():
            fill_below_diag_inplace(arr)

        # save to disk
        for n, arr in corr_arrs.items():
            np.save(a.out_dir / f"corr_{n}.npy", arr)

    log("plotting correlation previews")
    plot_corr_previews(a.out_dir, corr_arrs, class_names, a.n)

    log("plotting heatmap")
    plot_heatmap(
        a.out_dir,
        all_imgs,
        class_avgs,
        a.n,
        corr_arrs["max_scores"],
        use_ax_nums=not a.hm_nums_off,
        clip_to=a.score_clip,
        specify_classes={"GT": [0, 1, 2, 3], "APPLEpicker": [1, 3, 4, 5]},
    )

    log("plotting class average distributions")
    plot_class_distributions(a.out_dir, class_avgs)

    log("plotting correlation max score histogram")
    plot_max_score_hist(
        a.out_dir,
        corr_arrs["max_scores"],
        class_avgs,
        a.gt_name,
        a.n,
        clip_to=a.score_clip,
    )

    log("done.")
