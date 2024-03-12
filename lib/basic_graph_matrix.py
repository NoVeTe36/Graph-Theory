"""A naive pure python implementation of a Graph class

Based on ADJACENCY MATRIX

This implementation represents a graph where nodes are numbered, from 0 to N-1
where N is the order of the graph, known in advance.
Edges may be weighted by a float.

This implementation is for an DIRECTED GRAPH.

E. Viennet @ USTH, March 2024
"""

from collections import defaultdict
import numpy as np

class Graph_Matrix:
    def __init__(self, n: int = 0):
        # use a list of list as a matrix (inefficient ! consider using numpy)
        self.adjacency_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        self.ROW = len(self.adjacency_matrix)

    def _check_node_index(self, node):
        # checks if n is a valid node index in this graph
        if node < 0 or node >= len(self.adjacency_matrix):
            raise ValueError(f"invalid node index ({node})")

    def add_node(self, node: int):
        self._check_node_index(node)
        # nothing to do...

    def add_edge(self, node1, node2, weight=1.0, bidirectional=False):
        "Add edge from node1 to node2"
        self._check_node_index(node1)
        self._check_node_index(node2)
        self.adjacency_matrix[node1][node2] = weight
        if bidirectional:
            self.adjacency_matrix[node2][node1] = weight

    def get_nodes(self):
        return range(len(self.adjacency_matrix))

    def get_edges(self) -> list[tuple[int, int]]:
        """Returns the list of edges (from, to) with weight != 0.
        Weights are not returned.
        """
        edges = []
        n = len(self.adjacency_matrix)
        for node1 in range(n):
            for node2 in range(n):
                if self.adjacency_matrix[node1][node2] != 0.0:
                    edges.append((node1, node2))
        return edges

    def get_neighbors(self, node) -> list[int]:
        "List of node's neighbors (directed: node is the starting node)"
        self._check_node_index(node)
        row = self.adjacency_matrix[node]
        return [i for i, n in enumerate(row) if n != 0.0]

    def __repr__(self):
        return "\n".join(str(l) for l in self.adjacency_matrix)
    
    def get_adjacency_matrix(self):
        return self.adjacency_matrix
    
    def BFS(self,s, t, parent):
        visited =[False]*(self.ROW)     
        queue=[]      
        queue.append(s)
        visited[s] = True
        while queue: 
            u = queue.pop(0)
            for ind, val in enumerate(np.array(self.get_adjacency_matrix()).astype(int)[u]):
                if visited[ind] == False and val > 0 :
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u                   
        return True if visited[t] else False

    def EdKa(self, graph, source, sink):
        parent = {}
        max_flow = 0
        optimal_path = []
        while self.BFS(source, sink, parent):    
            path_flow = float("inf")
            
            while sink != source:
                path_flow = min(path_flow, np.array(self.get_adjacency_matrix()).astype(int)[parent[sink]][sink])
                sink = parent[sink]

            max_flow += path_flow

            while (sink != source):
                u = parent[sink]
                graph[u][sink] -= path_flow
                graph[sink][u] += path_flow
                sink = parent[sink]
                optimal_path.append(sink)
            return max_flow, optimal_path
        
    def get_edge_weight(self, node1, node2):
        return np.array(self.get_adjacency_matrix()).astype(int)[node1][node2]
    
    
# if __name__ == "__main__":
#     # Example usage
#     g = Graph_Matrix(3)
#     g.add_node(0)
#     g.add_node(1)
#     g.add_node(2)
#     g.add_edge(0, 1)
#     g.add_edge(1, 2)

#     print("Nodes:", g.get_nodes())
#     print("Edges:", g.get_edges())
#     print("Neighbors of 1:", g.get_neighbors(1))

#     print(g)

# """
# 1. Compare the two implementations
#     - Memory footprint
#     - add_edge()
#     - get_neighbors()
# """
