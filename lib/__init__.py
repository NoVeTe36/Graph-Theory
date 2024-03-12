# package

from .basic_graph_matrix import Graph_Matrix


def load_karate(bidirectional=True) -> Graph_Matrix:
    "Load karate graph. If not orinted, symetrize"
    FILENAME = "D:/Code/Visual Studio Code/USTH_project/Graph Theory/lib/soc-karate.mtx"
    n = 34  # we know the number of nodes = 34
    g = Graph_Matrix(34)
    with open(FILENAME, encoding="utf-8") as f:
        for line in f:
            if line and line[0] != "%":
                fields = line.split()
                if len(fields) == 2:  # select only line with edge definition
                    node1, node2 = int(fields[0]) - 1, int(fields[1]) - 1
                    g.add_edge(node1, node2, bidirectional=bidirectional)
    return g
