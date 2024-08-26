import itertools
import copy
import networkx as nx
import time
import os
import numpy as np


from bct import remove_redundant_blocks
from greedy import build_greedy_universe
import problem


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

    # sorted_edge_order = edge_orders[indices[0]]
    sorted_edge_order = edge_orders[indices[distance_indices[0]]]
    return sorted_edge_order


def get_all_neighbors(G: nx.Graph):
    all_neighbors = {}
    for i in G.nodes:
        all_neighbors[i] = set(G.neighbors(i))
    return all_neighbors


def frontier_size(
    edge,
    processed_nodes: set[int],
    current_frontier: set[int],
    current_neighbors: dict[int, set[int]],
):
    current_frontier |= set(edge)
    for i, v in enumerate(edge):
        rm_node = edge[1 - i]
        current_neighbors[v].remove(rm_node)
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


def swap_edges(edge_order, i, perm):
    n = len(edge_order)
    if n < 3:
        return edge_order
    new_order = copy.deepcopy(edge_order)

    indices = [i, (i + 1) % n, (i + 2) % n]
    permuted_edges = [edge_order[indices[p]] for p in perm]

    for idx, new_edge in zip(indices, permuted_edges):
        new_order[idx] = new_edge

    return new_order


def local_search(G, edge_order, seconds):
    start_time = time.time()
    all_neighbors = get_all_neighbors(G)
    best_order = copy.deepcopy(edge_order)
    best_score = all_frontier_size(best_order, all_neighbors)

    improved = True
    while improved and time.time() - start_time < seconds:
        improved = False
        for i in range(len(edge_order)):
            for perm in itertools.permutations([0, 1, 2]):
                new_order = swap_edges(best_order, i, perm)
                new_score = all_frontier_size(new_order, all_neighbors)
                if new_score < best_score:
                    best_order = new_order
                    best_score = new_score
                    improved = True
    return best_order


def remove_far_vertices(G, s, t, l):
    dict_s = nx.shortest_path_length(G, source=s)
    dict_t = nx.shortest_path_length(G, source=t)
    far_vertices = [v for v in G.nodes if dict_s[v] + dict_t[v] > l]
    G.remove_nodes_from(far_vertices)


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    success = {}
    for f_num, file in enumerate(dir):
        # if f_num != 43: # -38,-39,+37,-20,-19,-17,-16,+10,9,-7
        #     continue
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

            print(f"time = {time.time() - start_time}")
            print()

            # nx.draw(
            #     prob.G,
            #     with_labels=True,
            #     node_color="white",
            # )
            # plt.show()
