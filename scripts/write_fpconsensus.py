import numpy as np
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from itertools import takewhile, combinations
from write_jaccard import jaccard
from common import *


def overlap_graph(boxes, conf_range=(0.0, 1.0), **unused_kwargs):
    """Construct a graph from picker box lists, with edges between boxes that overlap
    each other within tp_agreement_range or more.

    Args:
        boxes (dict): Pass in as many box lists as should be included in the graph,
            keyed by the picker name. Should be of length >= 2.
        conf_range (tuple): Only boxes with confidence values within this range are
            considered when constructing the graph.

    Returns:
        nx.Graph: A k-partite graph, where k = len(box_lists).
    """

    graph = nx.Graph()

    def accept_pair(box1, box2):
        # check if both boxes satisfy conf_range
        for b in (box1, box2):
            if b.conf < conf_range[0] or b.conf > conf_range[1]:
                return False

        return True

    for i, (name1, lst1) in enumerate(boxes.items()):
        for j, (name2, lst2) in enumerate(boxes.items()):
            if j <= i:
                continue  # avoid duplicate edges or edges within the same box list

            pairs = [
                (box1, box2, jaccard(box1, box2))
                for box1 in lst1
                for box2 in lst2
                if accept_pair(box1, box2)
            ]

            # add all acceptable pairs to graph
            for box1, box2, jac in pairs:
                graph.add_node(box1, picker=name1)
                graph.add_node(box2, picker=name2)
                graph.add_edge(box1, box2, weight=jac)  # weight attribute used by nx

    return graph


def kpartite_cliques(graph, clique_size=3, **unused_kwargs):
    """Algorithm for finding non-duplicate cliques in graph using networkx's
    enumerate_all_cliques. A clique is a subset of graph vertices such that any two
    vertices in the subset are adjacent.

    Args:
        graph (nx.Graph): Multi-partite graph, as returned by graph_from_box_lists
        clique_size (int): Number of different box sets in graph g (i.e. k-partite)

    Returns:
        filter: Filter object of cliques (lists) of Box namedtuples. This acts like
            a generator; we don't cast to list immediately to avoid overhead.
    """

    def accept_clique(clique):
        # clique must be of length k
        if len(clique) != clique_size:
            return False

        # all k nodes in the clique must be from different pickers
        if len(set([graph.nodes[v]["picker"] for v in clique])) != clique_size:
            return False

        # overlap between any two boxes in clique must satisfy min_clique_agreement
        min_overlap = float("inf")
        for u, v in combinations(clique, 2):
            for edge in graph.get_edge_data(u, v):
                try:
                    if edge["weight"] < min_overlap:
                        min_overlap = edge["weight"]
                except TypeError:  # in some cases edge may be None
                    pass
                if min_overlap < min_clique_agreement:
                    return False

        return True

    cliques = filter(accept_clique, nx.enumerate_all_cliques(graph))

    return cliques


def fpconsensus_table(
    one_many_table,
    mrc_key="mrc",
    gt_key="gt",
    min_jac_key="min_jac",
    gt_overlap_range=(0.0, 0.6),
    min_clique_agreement=0.6,
    **graph_kwargs,
):

    one_many_table = deepcopy(one_many_table)
    one_many_table.pop(gt_key)  # won't need GTs
    unique_mrcs = list(set(one_many_table[mrc_key]))
    pickers = list(set(one_many_table.keys()) - {mrc_key})

    flat_table = {mrc_key: [], min_jac_key: [], **{p: [] for p in pickers}}

    # convert lists to numpy
    for k, v in one_many_table.items():
        one_many_table[k] = np.array(v, dtype=object)

    for mrc in tqdm(unique_mrcs):
        # build a dict of box lists, keyed by picker name:
        # {picker1: [box1, box2, ...], picker2: [box1, box2, ...], ...}
        box_lists = {}
        mrc_indices = np.where(one_many_table[mrc_key] == mrc)[0]
        for picker in pickers:
            boxes = list(flatten(list(one_many_table[picker][mrc_indices])))

            # remove boxes with ground-truth overlap outside gt_overlap_range
            box_lists[picker] = [
                box
                for box, jac in boxes
                if jac >= gt_overlap_range[0] and jac <= gt_overlap_range[1]
            ]

        # create cliques
        graph = overlap_graph(box_lists, **graph_kwargs)
        cliques = kpartite_cliques(graph, **graph_kwargs)

        # add cliques to table
        for clique in cliques:
            min_jac = min(
                [
                    edge["weight"]
                    for u, v in combinations(clique, 2)
                    for edge in graph.get_edge_data(u, v)
                ]
            )

            # only add clique if the lowest overlap between any two of its boxes
            # satisfies min_clique_agreement
            if min_jac >= min_clique_agreement:
                flat_table[mrc_key].append(mrc)
                flat_table[min_jac_key].append(min_jac)
                for box in clique:
                    flat_table[graph.nodes[box]["picker"]].append(box)

    return flat_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
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
        help="Overwrite (recalculate) any temporary data files in output directory",
        action="store_true",
    )

    a = parser.parse_args()

    # tmp
    matches = read_from_pickle(
        "/Users/seb/code/py/cryo-docs/scripts/TEST_OUT/one_many_matches.pickle"
    )
    fpconsensus_table(matches)


# def BAD_write_fpunion_table(
#     out_dir,
#     picker_lists,
#     conf_thresh=0.3,
#     tpt=0.6,
#     tpa_range=(0.6, 1.0),
#     jac_prefix="jac",
#     out_prefix="fpunions",
#     force=False,
# ):
#     """Write cliques generated via graph_from_boxes and nx_cliques to TSV table:
#     (1) where all boxes have jac score < tpt
#     (2) considering cliques in which all boxes overlap within tpa_range
#     (3) including only boxes with conf scores greater than conf_thresh

#     Args:
#         out_dir (str): Full path to output directory
#         picker_lists (list): List of lists of picker names, as provided by user in
#             config file. Each list of picker names represents the pickers included
#             in one output file. Output TSVs are numbered in the order of picker_lists.
#         conf_thresh (float, optional): Confidence threshold. Defaults to 0.3.
#         tpt (float, optional): True-positive threshold (below which a detection is FP).
#             Defaults to 0.6.
#         tpa_range (tuple, optional): Range of Jaccard overlaps which constitute a valid
#         false-positive "agreement" or "union" grouping. Defaults to (0.6, 1.0).
#         force (bool, optional): If True, redo calculations even if output TSV already
#             exists at destination. Defaults to False.
#     """

#     # get output subdirectory
#     subdir = _get_calc_subdir(out_dir)

#     # we can number the fpunions tables by index, since we assume the order
#     # of picker_names and fpunions_picker_lists won't change in the config file
#     # (if it does, grapher behavior upon reading fpunion tables is undefined)
#     for p_i, pckrs in enumerate(tqdm(picker_lists)):
#         # get table path
#         table_path = os.path.join(out_dir, subdir, "%s_%s.tsv" % (out_prefix, p_i))

#         # skip if table already exists
#         if not force and os.path.isfile(table_path):
#             logger.info("Using existing FP unions table at %s" % table_path)
#             continue

#         table_cols = ["mrc", "tpa"] + pckrs
#         fpunion_df = pd.DataFrame(columns=table_cols)

#         file_matches = im.read_file_matches(out_dir, orient="records")
#         for row in file_matches:
#             cur_mrc = row["mrc"]

#             jac_dfs = {p: im.read_tsv_df(out_dir, "jac", p) for p in pckrs}

#             # filter by mrc, confidence, and jaccard
#             jac_dfs = {
#                 p: df[
#                     (df["mrc"] == cur_mrc)
#                     & (df["pckr_conf"] >= conf_thresh)
#                     & (df["jac"] < tpt)
#                 ]
#                 for p, df in jac_dfs.items()
#             }

#             cols = Box._fields

#             # list of lists of boxes (one per picker)
#             boxes_by_pckr = [
#                 list(
#                     im.df_filter_cols(jac_dfs[p], "pckr", col_names=cols).itertuples(
#                         name="Box", index=False
#                     )
#                 )
#                 for p in pckrs
#             ]

#             # jac index between all combinations of boxes in a clique will be >= tpa
#             graph = sp.graph_from_boxes(tpa_range, *boxes_by_pckr)
#             cliques, min_tpas = sp.get_cliques_nx(graph, len(pckrs))

#             # empty clique list still needs to be 2d or we can't iterate over it
#             if not cliques:
#                 cliques = [[]]

#             null_clique = [None] * len(pckrs)
#             cliques_rows = [
#                 [cur_mrc, t, *c] if c else [cur_mrc, t, *null_clique]
#                 for c, t in zip(cliques, min_tpas)
#             ]

#             fpunion_df = fpunion_df.append(
#                 pd.DataFrame.from_records(cliques_rows, columns=table_cols)
#             )

#         fpunion_df.to_csv(table_path, sep=TSV_SEP, index=False, na_rep="nan")
#         logger.info("Wrote FP unions table to %s" % table_path)
