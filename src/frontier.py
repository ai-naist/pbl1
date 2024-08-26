import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
import itertools
import copy

import problem


# メンバ
# nx.Graph: G
# int: n, m, l, s, t
class Problem:

    # 注意：コンストラクタを呼んだ後，read_file, init_grid_graph等で初期化の必要あり．
    def __init__(
        self,
        path="public/rocketfuel-k_o-10.col",
    ):
        self.G = nx.Graph()
        self.read_file(path)

    # ファイルから入力を読み込む
    def read_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                data = line.split()
                if data[0] == "c":
                    continue
                elif data[0] == "p":
                    self.n, self.m = int(data[2]), int(data[3])
                elif data[0] == "e":
                    self.G.add_edge(int(data[1]), int(data[2]))
                elif data[0] == "l":
                    self.l = int(data[1])
                else:  # data[0] == 't'
                    self.s, self.t = int(data[1]), int(data[2])

    # k x k グリッドグラフを作成
    def init_grid_graph(self, k):
        self.n, self.m = k**2, 2 * k * (k - 1)
        self.s, self.t = 1, self.n
        self.l = 2 * (k - 1) + 3  # 最短パス長 = 2 * (k - 1). 1度まで迂回を許す
        for i in range(k):
            for j in range(k):
                v = k * i + j + 1  # v は 1-indexed
                if j < k - 1:
                    self.G.add_edge(v, v + 1)  # 横の辺
                if i < k - 1:
                    self.G.add_edge(v, v + k)  # 縦の辺

    # メンバを表示する関数
    def show(self):
        print(
            "n = {}, m = {}, l = {}, s = {}, t = {}".format(
                self.n, self.m, self.l, self.s, self.t
            )
        )
        print(self.G.edges)


dir_path = "public"
files = [
    "/rocketfuel-k_o-10.col",  # n, m = 10, 11
    "/topologyzoo-k_o-132.col",  # n, m = 145, 186
    "/matpower-case_ACTIVSg200-140.col",  # 200, 245
]

file_path = dir_path + files[1]  # 番号でファイルを指定
prob = Problem()
prob.read_file(file_path)
# prob.init_grid_graph(3)

# prob.show()
# nx.draw(prob.G, with_labels=True, node_color="white")
# nx.draw_spectral(prob.G, with_labels=True, node_color='white') # グリッドグラフの場合はこっちがよい

# from graphillion import GraphSet

# GraphSet.set_universe(prob.G.edges)


def zdd_size(graph_set):
    return len(graph_set.dumps().split("\n"))


def time_measure(loops: int):
    def decorator(func):
        import time

        def wrapper(*args, **kwargs):
            start = time.time()
            for i in range(loops):
                func(*args, **kwargs)
            print(time.time() - start)

        return wrapper

    return decorator


# def score(edge_order):
#     s = 0
#     for i in range(len(edge_order)):
#         v1 = set(e[0] for e in edge_order[:i]) | set(e[1] for e in edge_order[:i])
#         v2 = set(e[0] for e in edge_order[i:]) | set(e[1] for e in edge_order[i:])
#         frontier = v1 & v2
#         s += len(frontier) ** 2
#     return s


# def score_func(edge_order):
#     s = 0
#     for i in range(len(edge_order) - 1):
#         if i:
#             v1 |= set(edge_order[i])
#             queue.popleft()
#         else:
#             v1 = set(edge_order[i])
#             queue = collections.deque(edge_order[1:])

#         v2 = set(itertools.chain.from_iterable(queue))
#         frontier = v1 & v2
#         s += len(frontier) ** 2
#     return s


edge_order = list(prob.G.edges)

# 時間計測
# import time

# start = time.time()
# for i in range(100):
#     score(edge_order)
#     # score_func(edge_order)
# print(time.time() - start)

# print(score(edge_order))
# print(score_func(edge_order))

# 隣接ノード
all_neighbors = {}
for i in range(1, prob.n + 1):
    all_neighbors[i] = list(prob.G.neighbors(i))
# print(all_neighbors)


def get_all_neighbors(G: nx.Graph):
    all_neighbors = {}
    for i in G.nodes:
        all_neighbors[i] = set(G.neighbors(i))
    return all_neighbors


# print(get_all_neighbors(prob.G))


def set_frontier(
    graph: nx.Graph,
    edge: tuple[int, int],
    current_frontier: set[int],
    processed_nodes: set[int],
    processed_edges: set[frozenset[int, int]],
):
    processed_edges.add(frozenset(edge))
    for node in edge:
        focused_edges = set(
            frozenset((node, neighbor)) for neighbor in graph.neighbors(node)
        )
        if not len(focused_edges - processed_edges):
            processed_nodes.add(node)
            current_frontier.discard(node)
        else:
            current_frontier.add(node)
    return current_frontier, processed_nodes, processed_edges


def get_next_edge_candidates(
    current_frontier: set[int],
    processed_edges: set[frozenset[int, int]],
):
    next_edge_candidates = set()
    for node in current_frontier:
        for neighbor in all_neighbors[node]:
            edge = frozenset((node, neighbor))
            next_edge_candidates.add(edge)
    next_edge_candidates -= processed_edges
    return next_edge_candidates


def calc_next_frontiers_size(
    graph: nx.Graph,
    next_edge_candidates: set[frozenset[int, int]],
    current_frontier: set[int],
    processed_nodes: set[int],
    processed_edges: set[frozenset[int, int]],
):
    next_frontiers_size = {}
    for edge in next_edge_candidates:
        next_frontier = set_frontier(
            graph,
            edge,
            copy.deepcopy(current_frontier),
            copy.deepcopy(processed_nodes),
            copy.deepcopy(processed_edges),
        )[0]
        next_frontiers_size[frozenset(edge)] = len(next_frontier)
    next_frontiers_size = dict(sorted(next_frontiers_size.items(), key=lambda x: x[1]))
    return next_frontiers_size


# current_frontier, processed_nodes, processed_edges = get_frontier(
#     prob.G, (6, 4), set(), set(), set(frozenset())
# )
# next_edge_candidates = get_next_edge_candidates(
#     current_frontier, processed_nodes, processed_edges
# )
# next_frontiers_size = calc_next_frontiers_size(
#     prob.G, next_edge_candidates, current_frontier, processed_nodes, processed_edges
# )
# next_edge_stack = collections.deque(next_frontiers_size.keys())
# next_edge = tuple(next_edge_stack.popleft())
# print(current_frontier, processed_nodes, processed_edges)
# print(next_edge_candidates)
# print(next_frontiers_size)
# print(next_edge_stack)
# print(next_edge)


# @time_measure(loops=100)
# def test():
#     get_frontier((6, 4), set(), set(), set(frozenset()))

# test()
# print(get_frontier((6, 4), set(), set(), set(frozenset())))


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


# @time_measure(loops=100)
# def test():
#     all_frontier_size(edge_order, all_neighbors)


edge_order = [
    (1, 2),
    (2, 9),
    (2, 7),
    (7, 8),
    (7, 5),
    (5, 10),
    (5, 11),
    (5, 1),
    (1, 3),
    (3, 4),
    (4, 6),
    (4, 1),
]
# test()

# 次数の少ないノードを抽出
degrees = dict(nx.degree(prob.G))
del degrees[prob.s], degrees[prob.t]
# print(degrees)
sorted_v = sorted(degrees, key=lambda x: degrees[x])
v_list_size = max(int(2 * (len(edge_order) ** 0.5)), list(degrees.values()).count(1))  # fmt: skip
candidates_source_node = sorted_v[:v_list_size]
# 始点と終点を先頭に追加
candidates_source_node.insert(0, prob.t)
candidates_source_node.insert(0, prob.s)
print(candidates_source_node)


edge_order_candidates = []
for source_node in candidates_source_node:
    edge_order = []
    current_frontier = set()
    processed_nodes = set()
    processed_edges = set()
    processed_nodes.add(source_node)
    init_edges = get_next_edge_candidates(processed_nodes, processed_edges)
    next_frontiers_size = calc_next_frontiers_size(
        prob.G,
        init_edges,
        current_frontier,
        processed_nodes,
        processed_edges,
    )
    next_edge_stack = collections.deque(next_frontiers_size.keys())
    edge = tuple(next_edge_stack.popleft())
    remain_edge_set = set([frozenset(edge) for edge in prob.G.edges])
    edge_order.append(edge)
    remain_edge_set.remove(set(edge))
    current_frontier, processed_nodes, processed_edges = set_frontier(
        prob.G, edge, current_frontier, processed_nodes, processed_edges
    )
    while remain_edge_set:
        next_edge_candidates = get_next_edge_candidates(
            current_frontier, processed_edges
        )
        next_frontiers_size = calc_next_frontiers_size(
            prob.G,
            next_edge_candidates,
            current_frontier,
            processed_nodes,
            processed_edges,
        )
        next_edge_stack = collections.deque(next_frontiers_size.keys())
        if not next_edge_stack:
            next_frontiers_size = calc_next_frontiers_size(
                prob.G,
                remain_edge_set,
                current_frontier,
                processed_nodes,
                processed_edges,
            )
            next_edge_stack = collections.deque(next_frontiers_size.keys())
        edge = tuple(next_edge_stack.popleft())
        edge_order.append(edge)
        remain_edge_set.remove(set(edge))
        current_frontier, processed_nodes, processed_edges = set_frontier(
            prob.G, edge, current_frontier, processed_nodes, processed_edges
        )
    edge_order_candidates.append(edge_order)


for edge_order in edge_order_candidates:
    GraphSet.set_universe(edge_order, traversal="as-is")
    paths = GraphSet.paths(prob.s, prob.t).smaller(prob.l + 1)
    print(edge_order)
    print(all_frontier_size(edge_order))
    print(zdd_size(paths))
