import argparse
import numpy as np
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from itertools import combinations
from datetime import datetime
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
        # clique must be of length clique_size
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
        # build a dict of box lists, keyed by picker name
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
            # ignore clique if the lowest overlap between any two of its boxes
            # is less than min_clique_agreement
            do_skip = any(
                [
                    edge["weight"] < min_clique_agreement
                    for u, v in combinations(clique, 2)
                    for edge in graph.get_edge_data(u, v)
                ]
            )

            if do_skip:
                continue

            # we're ok to add this clique
            flat_table[mrc_key].append(mrc)
            flat_table[min_jac_key].append(min_jac)
            remaining_pickers = set(pickers)
            for box in clique:
                picker = graph.nodes[box]["picker"]
                flat_table[picker].append(box)
                remaining_pickers.remove(picker)
            for picker in remaining_pickers:
                flat_table[picker].append(None)  # ensure lists stay the same length

    return flat_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a table of picker cliques such that all picker boxes are"
        "within a given confidence range and true-positive overlap range with the "
        "ground-truth. Additionally, ensure that each clique meets the minimum clique "
        "agreement overlap threshold."
    )
    parser.add_argument(
        "out_dir",
        help=f"Output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "one_many_matches_path",
        help=f"Path to one-to-many Jaccard matching table (generated by "
        "write_jaccard.py)",
    )
    parser.add_argument(
        "-k",
        help=f"Clique sizes for which to create tables",
        required=True,
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--gt_overlap_range",
        help=f"True-positive Jaccard score range (range of scores with ground-truth) "
        "outside which boxes are not added to cliques.",
        nargs=2,
        default=(0.0, 0.6),
        type=float,
    )
    parser.add_argument(
        "--min_clique_agreement",
        help=f"Minimum overlap required between any two boxes in a clique for the "
        "clique to be added to the table.",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--conf_range",
        help=f"Range of box confidences to include in cliques.",
        default=(0.0, 1.0),
        nargs=2,
        type=float,
    )
    parser.add_argument(
        "--mrc_key",
        help=f"Key in file matches to use as the micrograph set. Defaults to 'mrc'.",
        default="mrc",
    )
    parser.add_argument(
        "--gt_key",
        help=f"Key in file matches to read as the ground truth set. Defaults to 'gt'.",
        default="gt",
    )
    parser.add_argument(
        "--min_jac_key",
        help=f"Key in output table that represents the minimum Jaccard index between "
        "any two boxes in a given clique. Defaults to 'jac'.",
        default="jac",
    )

    a = parser.parse_args()

    matches = read_from_pickle(a.one_many_matches_path)

    for clique_size in a.k:
        table = fpconsensus_table(
            matches,
            mrc_key=a.mrc_key,
            gt_key=a.gt_key,
            min_jac_key=a.min_jac_key,
            gt_overlap_range=a.gt_overlap_range,
            min_clique_agreement=a.min_clique_agreement,
            # graph_kwargs start here
            clique_size=a.k,
            conf_range=a.conf_range,
        )

        # since this takes a while to process, make sure we write to disk
        a.out_dir = norm_path(a.out_dir)
        filename, ext = f"fpcliques_{clique_size}", ".pickle"
        if (a.out_dir / (filename + ext)).is_file():
            print(f"filename {filename} already exists; appending current time string")
            filename += datetime.now().strftime("_%y%m%d%H%M%S")

        write_to_pickle(a.out_dir, table, filename + ext)
