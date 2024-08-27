import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import time

from graphillion import GraphSet as gs
from multiprocessing import Pool, cpu_count

import problem
from greedy import build_greedy_universe
from bct import remove_redundant_blocks
from search import local_search


def remove_degree_one_vertices(prob):
    while True:
        removed = False
        for v in prob.G.nodes:
            if v != prob.s and v != prob.t and prob.G.degree[v] == 1:
                prob.G.remove_node(v)
                removed = True
                break
        if not removed:
            break


def frontier_size(
    edge,
    processed_nodes: set[int],
    current_frontier: set[int],
    current_neighbors: dict[int, set[int]],
):
    current_frontier |= set(edge)
    for i, v in enumerate(edge):
        current_neighbors[v].remove(edge[1 - i])
        if len(current_neighbors[v]) == 0:
            processed_nodes.add(v)
    current_frontier -= processed_nodes
    return len(current_frontier)


def all_frontier_size(edge_order, all_neighbors):
    score = 0
    current_neighbors: dict[int, set[int]] = copy.deepcopy(all_neighbors)
    processed_nodes = set()
    current_frontier = set()
    for edge in edge_order:
        score += (
            frontier_size(edge, processed_nodes, current_frontier, current_neighbors)
            ** 2
        )
    return score


# def zdd_size(graphset: gs):
#     return len(graphset.dumps().split("\n"))


def remove_far_vertices(G, s, t, l):
    dict_s = nx.shortest_path_length(G, source=s)
    dict_t = nx.shortest_path_length(G, source=t)
    far_vertices = [v for v in G.nodes if dict_s[v] + dict_t[v] > l]
    G.remove_nodes_from(far_vertices)


def path_count(edge_order, s, t, l, score):
    if score > 30000:
        return None
    gs.set_universe(edge_order, traversal="as-is")
    paths = gs.paths(s, t).smaller(l + 1)
    return paths.len()


def count_all_simple_paths(args):
    G, source, target, cutoff = args
    return len(list(nx.all_simple_paths(G, source, target, cutoff=cutoff)))


def parallel_count_all_simple_paths(G, sources, target, cutoff):
    with Pool(processes=cpu_count()) as pool:
        args = [(G, source, target, cutoff) for source in sources]
        path_counts = pool.map(count_all_simple_paths, args)
    total_paths = sum(path_counts)
    return total_paths


def get_all_neighbors(G: nx.Graph):
    all_neighbors = {}
    for i in G.nodes:
        all_neighbors[i] = set(G.neighbors(i))
    return all_neighbors


def get_path_distance(G, v, s, t):
    distance_s = nx.shortest_path_length(G, source=v, target=s)
    distance_t = nx.shortest_path_length(G, source=v, target=t)
    sum_distance = distance_s + distance_t
    diff_distance = abs(distance_s - distance_t)
    return (sum_distance, diff_distance)


def get_edge_order(G, s, t):
    all_neighbors = get_all_neighbors(G)
    scores = []
    edge_orders = []
    i_to_v = {}

    for i, v in enumerate(G.nodes):
        edge_order = build_greedy_universe(G.edges, source=v, neighbors=all_neighbors)
        score = all_frontier_size(
            edge_order,
            all_neighbors,
        )
        scores.append(score)
        edge_orders.append(edge_order)
        i_to_v[i] = v

    indices = np.argsort(scores)[: len(scores) // 4]
    if len(indices) == 0:
        return edge_orders[0]
    sorted_score = sorted(scores)[: len(scores) // 4]
    path_distances = [get_path_distance(G, i_to_v[i], s, t) for i in indices]
    distance_indices = [
        i for i, _ in sorted(enumerate(path_distances), key=lambda x: x[1])
    ]

    sorted_edge_order = edge_orders[indices[distance_indices[0]]]
    return sorted_edge_order


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    success = {}
    for f_num, file in enumerate(dir):
        if f_num != 7:  # -38,-39,+37,-20,-19,-17,-16,+10,9,-7
            continue
        start_time = time.time()
        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

            remove_far_vertices(prob.G, prob.s, prob.t, prob.l)
            remove_redundant_blocks(prob.G, prob.s, prob.t)

            edge_order = get_edge_order(prob.G, prob.s, prob.t)
            all_neighbors = get_all_neighbors(prob.G)
            score = all_frontier_size(edge_order, all_neighbors)
            print(f"pre_score = {score}")
            edge_order = local_search(prob.G, edge_order, 60)
            score = all_frontier_size(edge_order, all_neighbors)
            print(f"post_score = {score}")

            number_of_passes = path_count(edge_order, prob.s, prob.t, prob.l, score)

            print(f"paths = {number_of_passes}")
            print(f"time = {time.time() - start_time}")
            if number_of_passes is not None:
                success[f_num] = time.time() - start_time
            print()

            # nx.draw(
            #     prob.G,
            #     with_labels=True,
            #     node_color="white",
            # )
            # plt.show()
    print(f"solved:{len(success)}")
    for case in success:
        print(f"{case}  {success[case]:.6f}s")
