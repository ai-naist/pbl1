import networkx as nx
import os
import time
from graphillion import GraphSet as gs

import problem


def remove_far_vertices(prob):
    dist_s = nx.shortest_path_length(prob.G, source=prob.s)
    dist_t = nx.shortest_path_length(prob.G, source=prob.t)
    far_vertices = []
    for v in prob.G.nodes:
        if dist_s[v] + dist_t[v] > prob.l:
            far_vertices.append(v)
    prob.G.remove_nodes_from(far_vertices)


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


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    success = {}
    for f_num, file in enumerate(dir):
        if f_num != 8:  # -38,-39,+37,-20,-19,-17,-16,+10,9,-7
            continue
        start_time = time.time()
        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

            remove_far_vertices(prob)
            remove_degree_one_vertices(prob)

            gs.set_universe(prob.G.edges)
            paths = gs.paths(prob.s, prob.t).smaller(prob.l + 1)
            print("zdd:", len(paths.dumps().split("\n")))
