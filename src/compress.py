import networkx as nx
import matplotlib.pyplot as plt
import os
import time

from graphillion import GraphSet as gs

import problem
from bct import remove_redundant_blocks, remove_far_vertices


def remove_2degree_nodes(G: nx.MultiGraph, s, t):
    removed = True
    while removed:
        removed = False
        for v in list(G.nodes):
            if G.degree[v] == 2 and v != s and v != t:
                neighbors = list(G.neighbors(v))
                u, w = neighbors
                if G.has_edge(u, w):
                    continue
                weight = G[v][u][0]["weight"] + G[v][w][0]["weight"]
                G.add_edge(neighbors[0], neighbors[1], weight=weight)
                G.remove_node(v)
                removed = True


def sample():
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 5),
        (5, 1),
        (1, 6),
        (6, 7),
        (7, 8),
        (4, 8),
        (5, 8),
    ]
    G = nx.Graph(edges)
    s = 1
    t = 4
    l = 3
    return G, s, t, l


def plot(G: nx.MultiGraph):
    pos = nx.spring_layout(G_m)
    edge_labels = nx.get_edge_attributes(G_m, "weight")
    nx.draw_networkx_nodes(G_m, pos, node_size=700)
    nx.draw_networkx_edge_labels(G_m, pos, edge_labels=edge_labels)
    for edge in G_m.edges(data=True):
        nx.draw_networkx_edges(
            G_m, pos, edgelist=[(edge[0], edge[1])], width=1.0, alpha=0.5
        )
    nx.draw_networkx_labels(G_m, pos, font_size=20, font_family="sans-serif")
    plt.show()


def to_gs_edge_list(G):
    edge_list = [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]
    return edge_list


def path_count(edge_order, s, t, l):
    gs.set_universe(edge_order, traversal="as-is")
    print(gs.graphs())
    print(gs._weights)
    paths = gs.paths(s, t)
    return paths.len()


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    success = {}
    for f_num, file in enumerate(dir):
        if f_num != 37:  # -38,-39,+37,-20,-19,-17,-16,10,9,-7
            continue
        start_time = time.time()
        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

        # G, s, t, l = sample()

        G = prob.G
        s = prob.s
        t = prob.t
        l = prob.l

        remove_far_vertices(G, s, t, l)
        remove_redundant_blocks(G, s, t)

        G_m = nx.MultiGraph(G)
        for edge in G_m.edges(data=True):
            edge[2]["weight"] = 1

        remove_2degree_nodes(G_m, s, t)

        # plot(G_m)

        edge_list = to_gs_edge_list(G_m)
        print(edge_list)
        number_of_paths = path_count(edge_list, s, t, l)
        print(f"paths = {number_of_paths}")
