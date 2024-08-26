import networkx as nx
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

import problem
from bct import remove_redundant_blocks


def path_count(G, s, t, l):
    return len(list(nx.all_simple_paths(G, s, t, cutoff=l)))


# def count_all_simple_paths(args):
#     G, source, target, cutoff = args
#     return len(list(nx.all_simple_paths(G, source, target, cutoff=cutoff)))


# def parallel_count_all_simple_paths(G, sources, target, cutoff):
#     with Pool(processes=cpu_count()) as pool:
#         args = [(G, source, target, cutoff) for source in sources]
#         path_counts = pool.map(count_all_simple_paths, args)
#     total_paths = sum(path_counts)
#     return total_paths


if __name__ == "__main__":
    dir = sorted(os.listdir("public"))
    for f_num, file in enumerate(dir):
        if f_num != 7:
            continue

        if file.endswith(".col"):
            print(f_num, end=" ")
            prob = problem.Problem(file)
            prob.show()

            remove_redundant_blocks(prob.G, prob.s, prob.t)

            number_of_passes = path_count(prob.G, prob.s, prob.t, prob.l)
            # number_of_passes = parallel_count_all_simple_paths(
            #     prob.G, [prob.s], prob.t, prob.l
            # )

            print(f"paths = {number_of_passes}")

            # nx.draw(
            #     prob.G,
            #     with_labels=True,
            #     node_color="white",
            # )
            # plt.show()
