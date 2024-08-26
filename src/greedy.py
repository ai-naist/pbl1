import heapq
import networkx as nx

# from graphillion import GraphSet as gs


def build_greedy_universe(universe, source, neighbors):
    sorted_edges = []
    indexed_edges = set()
    for e in universe:
        if e[:2] in indexed_edges or (e[1], e[0]) in indexed_edges:
            raise KeyError(e)
        sorted_edges.append(e[:2])
        indexed_edges.add(e[:2])

    sorted_edges = greedy_sort(indexed_edges, source, neighbors)
    # print(indexed_edges) # set of edges
    return sorted_edges


def greedy_sort(indexed_edges, source, neighbors):
    vertices = set(neighbors.keys())

    sorted_edges = []
    visited_vertices = set()
    u = source

    degree = dict()
    for v in vertices:
        degree[v] = len(neighbors[v])

    heap = []
    while True:
        visited_vertices.add(u)
        for v in sorted(neighbors[u]):
            degree[v] -= 1
            if v in visited_vertices:
                degree[u] -= 1
                e = (u, v) if (u, v) in indexed_edges else (v, u)
                sorted_edges.append(e)
                if degree[v]:
                    for w in sorted(neighbors[v]):
                        if w not in visited_vertices:
                            heapq.heappush(heap, (degree[v], degree[w], w))
        for v in sorted(neighbors[u]):
            if v not in visited_vertices:
                heapq.heappush(heap, (degree[u], degree[v], v))
        if visited_vertices == vertices:
            break
        while u in visited_vertices:
            if not heap:
                u = min(vertices - visited_vertices)
            else:
                u = heapq.heappop(heap)[2]
    assert set(indexed_edges) == set(sorted_edges)
    return sorted_edges


def get_all_neighbors(G: nx.Graph):
    all_neighbors = {}
    for i in G.nodes:
        all_neighbors[i] = set(G.neighbors(i))
    return all_neighbors


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from problem import Problem
    import networkx as nx

    files = sorted(os.listdir("public"))
    for f_num, file in enumerate(files):
        if f_num != 11:
            continue
        if file.endswith(".col"):
            prob = Problem(file)
            print(f_num, end=" ")
            prob.show()

            all_neighbors = get_all_neighbors(prob.G)
            print(all_neighbors)

            edge_order = build_greedy_universe(prob.G.edges(), prob.s, all_neighbors)
            # print(edge_order)

            nx.draw(prob.G, with_labels=True, node_color="white")
            plt.show()
