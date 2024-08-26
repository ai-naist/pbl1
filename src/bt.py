import networkx as nx
import matplotlib.pyplot as plt
import os
import time

import problem
from bct import remove_redundant_blocks, remove_far_vertices


class BT:
    def __init__(self, prob):
        self.prob = prob
        self.shortest_path_lengths = {}
        self.calc_shortest_path_length()

    def calc_shortest_path_length(self):
        for node in self.prob.G.nodes:
            self.shortest_path_lengths[node] = nx.shortest_path_length(
                self.prob.G, source=node, target=self.prob.t
            )

    def dfs(self, v):
        self.call_count += 1  # 呼び出し回数をカウント
        self.used[v] = True
        self.length += 1

        if v == self.prob.t and self.length <= self.prob.l:
            self.num_path += 1

        else:  # v != t
            for u in self.prob.G.neighbors(v):  # u は v と隣接する頂点
                if not self.used[u]:
                    if self.length + self.shortest_path_lengths[u] > self.prob.l:
                        self.used[u] = False
                        continue

                    self.dfs(u)  # uが探索中のパスに含まれない頂点なら探索

        self.used[v] = False
        self.length -= 1

    def run(self):
        self.call_count = 0
        self.used = {v: False for v in self.prob.G.nodes}
        self.length = 0
        self.num_path = 0

        self.dfs(self.prob.s)

    def count_nx(self):
        self.num_path = len(
            list(
                nx.all_simple_paths(
                    self.prob.G, self.prob.s, self.prob.t, cutoff=self.prob.l
                )
            )
        )


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    success = {}
    for f_num, file in enumerate(dir):
        if f_num != 21:  # -38,-39,+37,-20,-19,-17,-16,+10,9,-7
            continue
        start_time = time.time()
        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

            remove_far_vertices(prob.G, prob.s, prob.t, prob.l)
            remove_redundant_blocks(prob.G, prob.s, prob.t)

            bt = BT(prob)
            # bt.run()
            bt.count_nx()

            print(f"paths = {bt.num_path}")
            # print(f"call_count = {bt.call_count}")
            print(f"time = {time.time() - start_time}")
            print()
