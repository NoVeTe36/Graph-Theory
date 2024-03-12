"""A naive pure python implementation of a Graph class

Based on ADJACENCY LIST

This implementation represents a graph where nodes are stored as keys in a
dictionary (`adjacency_list`), and the values are lists containing the neighbors
of each node. This approach is efficient for sparse graphs and provides quick
access to neighbors, which is a common operation in graph algorithms.

Note that this is a basic implementation. Depending on your needs, you might
want to add error handling (e.g., when adding edges between non-existent nodes)
or additional features like methods for removing nodes or edges, checking if a
path exists between two nodes, etc.

This implementation is for an UNDIRECTED GRAPH; it can be modified
for directed graphs by adjusting how edges are added.


E. Viennet @ USTH, March 2024
"""


class Graph_AdjList:
    def __init__(self):
        self.adjacency_lists = {}  # { node : list of neghbors }

    def add_node(self, node):
        if node not in self.adjacency_lists:
            self.adjacency_lists[node] = []

    def add_edge(self, node1, node2):
        if node1 in self.adjacency_lists and node2 in self.adjacency_lists:
            if node2 not in self.adjacency_lists[node1]:  # avoid duplicate links
                self.adjacency_lists[node1].append(node2)
                # undirected graph, add reciprocal edge:
                self.adjacency_lists[node2].append(node1)
        else:
            raise ValueError("add_edge: unknown node")

    def get_nodes(self):
        return list(self.adjacency_lists.keys())

    def get_edges(self):
        edges = []
        for node, neighbors in self.adjacency_lists.items():
            for neighbor in neighbors:
                if (
                    neighbor,
                    node,
                ) not in edges:  # Needed to avoid duplicates in undirected graph
                    edges.append((node, neighbor))
        return edges

    def get_neighbors(self, node):
        return self.adjacency_lists[node] if node in self.adjacency_lists else None

    def __str__(self):
        return str(self.adjacency_lists)


# Example usage
g = Graph_AdjList()
g.add_node("A")
g.add_node("B")
g.add_node("C")
g.add_edge("A", "B")
g.add_edge("B", "C")
g.add_edge("A", "B")  # Trying to add a duplicate edge

print("Nodes:", g.get_nodes())
print("Edges:", g.get_edges())
print("Neighbors of B:", g.get_neighbors("B"))

"""
1. Does the `add_node` method prevent the addition of duplicate nodes to the graph?

2. Is the `add_edge` method designed to add an edge between two nodes that do not already exist in the graph?

3. Can the `get_neighbors` method return a list of all nodes connected to a given node?

4. Does the current graph implementation support directed edges?

5. In the `get_edges` method, is a check performed to prevent the same edge from being listed twice for an undirected graph?

6. Is the adjacency list representation more space-efficient than an adjacency matrix for sparse graphs?

7. Does the current implementation of the graph allow for the storage of edge weights?

8. Can the `get_nodes` method return a list of all nodes currently in the graph?

9. Can this graph implementation be used to represent a network of interconnected computers in a simulation?

10. What is the requirement on `node` type ?
    i.e;: can a node be
    - an integer or a float ?
    - a string ?
    - a tuple ?
    - A list ?

11. What is the time complexity of checking for the existence of a specific edge in the grah ?

12. What is the time complexity of checking for the existence of a specific node in the grah ?

"""
