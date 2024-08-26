import networkx as nx
import os
import matplotlib.pyplot as plt


class Problem:
    def __init__(self, file_path):
        self.G = nx.Graph()
        self.name = file_path

        with open("public/" + file_path) as f:
            for line in f:
                data = line.split()
                if data[0] == "c":
                    continue
                elif data[0] == "p":
                    self.n, self.m = int(data[2]), int(data[3])
                elif data[0] == "e":
                    if data[1] == data[2]:
                        continue
                    self.G.add_edge(int(data[1]), int(data[2]))
                elif data[0] == "l":
                    self.l = int(data[1])
                else:  # data[0] == 't'
                    self.s, self.t = int(data[1]), int(data[2])

    def show(self):
        print(f"{self.name}")
        print(
            "n = {}, m = {}, l = {}, s = {}, t = {}".format(
                self.n, self.m, self.l, self.s, self.t
            )
        )

    def plot(self):
        nx.draw(self.G, with_labels=True, node_color="white")


if __name__ == "__main__":
    files = sorted(os.listdir("public"))
    for f_num, file in enumerate(files):
        if f_num != 43:
            continue
        if file.endswith(".col"):
            prob = Problem(file)
            print(f_num, end=" ")
            prob.show()

            nx.draw(prob.G, with_labels=True, node_color="white")
            plt.show()
