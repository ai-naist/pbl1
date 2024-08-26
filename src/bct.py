import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import collections
import time

import problem

# from graphillion import GraphSet as gs


def remove_far_vertices(G, s, t, l):
    dict_s = nx.shortest_path_length(G, source=s)
    dict_t = nx.shortest_path_length(G, source=t)
    far_vertices = [v for v in G.nodes if dict_s[v] + dict_t[v] > l]
    G.remove_nodes_from(far_vertices)


def remove_redundant_blocks(G: nx.Graph, s, t):
    biconn_comps = list(nx.biconnected_components(G))

    # aps = set(nx.articulation_points(G)) より高速
    aps = set(
        [
            ap
            for ap, count in collections.Counter(
                itertools.chain.from_iterable(biconn_comps)
            ).items()
            if count > 1
        ]
    )

    bct = []
    for b_i, block in enumerate(biconn_comps):
        for ap in aps.intersection(block):
            bct.append((b_i, -ap))
        if s in block:
            s_bct = b_i
        if t in block:
            t_bct = b_i

    if s_bct == t_bct:
        remove_nodes = set(G.nodes()) - biconn_comps[s_bct]
        G.remove_nodes_from(remove_nodes)
        return

    G_bct = nx.Graph(bct)
    # print(s_bct, t_bct)
    # nx.draw(G_bct, with_labels=True)
    # plt.show()

    inpath_bct_nodes = nx.bidirectional_shortest_path(G_bct, s_bct, t_bct)
    inpath_vertices = set()
    for node in inpath_bct_nodes:
        if node < 0:
            inpath_vertices.add(-node)
        else:
            inpath_vertices |= biconn_comps[node]
    remove_nodes = set(G.nodes()) - inpath_vertices
    G.remove_nodes_from(remove_nodes)


def max_bicc_size(G):
    return max(len(bicc) for bicc in nx.biconnected_components(G))


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    for f_num, file in enumerate(dir):
        # if f_num != 37:
        #     continue

        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

            # nx.draw(
            #     prob.G,
            #     with_labels=True,
            #     node_color="white",
            # )
            # plt.show()

            remove_far_vertices(prob.G, prob.s, prob.t, prob.l)
            remove_redundant_blocks(prob.G, prob.s, prob.t)

            print(f"max_bicc_size = {max_bicc_size(prob.G)}")

            # nx.draw(
            #     prob.G,
            #     with_labels=True,
            #     node_color="white",
            # )
            # plt.show()

            print(f"n = {len(prob.G.nodes())}, m = {len(prob.G.edges())}")
