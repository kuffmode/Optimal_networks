from typing import Union
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog

class NavigableNetwork:
    """
    A class representing a navigable network with Nash equilibrium properties.

    This class implements a network where nodes are positioned in coordinate space
    and can establish connections to enable efficient navigation between any pair
    of nodes using greedy routing.

    Attributes:
        coordinates (np.ndarray): Array of node coordinates in n-dimensional space.
            Shape: (n_nodes, n_dimensions)
        n_nodes (int): Number of nodes in the network
        distances (np.ndarray): Matrix of pairwise distances between all nodes.
            Shape: (n_nodes, n_nodes)
    """

    def __init__(self, coordinates: np.ndarray):
        """
        Initialize the NavigableNetwork with node coordinates.

        Args:
            coordinates (np.ndarray): Array of node coordinates in n-dimensional space.
                Shape: (n_nodes, n_dimensions)
        """
        if len(coordinates) == 0:
            raise ValueError("Coordinates array cannot be empty.")
        if coordinates.ndim != 2:
            raise ValueError("Coordinates must be a 2D array with shape (n_nodes, dimension)")

        self.coordinates = coordinates
        self.n_nodes = len(coordinates)
        # Calculate pairwise distances between all nodes
        self.distances = self._compute_distances()

    def _compute_distances(self) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all nodes in the network.

        Returns:
            np.ndarray: Matrix of pairwise distances with shape (n_nodes, n_nodes)
                where distances[i,j] represents the Euclidean distance between
                nodes i and j.
        """
        return squareform(pdist(self.coordinates, metric='euclidean'))

    def _compute_frame_edges(self) -> np.ndarray:
        """
        Compute the frame edges required for the frame topology.

        Frame edges are defined as edges that must exist because no other node
        can replace them as the greedy next-hop for a target.

        Returns:
            np.ndarray: Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i, j] = True indicates that the edge i -> j is a frame edge.
        """
        frame_adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        for target in range(self.n_nodes):
            for source in range(self.n_nodes):
                if source == target:
                    continue

                # Check if any other node can replace source as a greedy next-hop to target
                is_frame_edge = True
                for other in range(self.n_nodes):
                    if other == source or other == target:
                        continue
                    if self.distances[other, target] < self.distances[source, target]:
                        is_frame_edge = False
                        break

                if is_frame_edge:
                    frame_adjacency[source, target] = True

        return frame_adjacency

    def build_nash_equilibrium(self) -> np.ndarray:
        """
        Build the Nash equilibrium network configuration using linear programming.

        Constructs an adjacency matrix representing the network configuration
        where no node can improve its routing capability by changing its
        connections unilaterally.

        Returns:
            np.ndarray: Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j
        """
        # Start with frame edges
        adjacency = self._compute_frame_edges()

        for source_node in range(self.n_nodes):
            n_vars = self.n_nodes
            c = np.ones(n_vars)

            A = []
            b = []
            for target_node in range(self.n_nodes):
                if target_node == source_node:
                    continue

                constraint = np.zeros(n_vars)
                for neighbor in range(self.n_nodes):
                    if neighbor == source_node:
                        continue
                    if self.distances[neighbor, target_node] <= self.distances[source_node, target_node]:
                        constraint[neighbor] = 1

                A.append(constraint)
                b.append(1)

            bounds = [(0, 1) for _ in range(n_vars)]
            result = linprog(c, A_ub=-np.array(A), b_ub=-np.array(b), bounds=bounds, method='highs')

            if result.success:
                selected_neighbors = np.where(result.x > 0.5)[0]
                adjacency[source_node, selected_neighbors] = True
            else:
                raise ValueError(f"Linear programming failed for source node {source_node}.")

        return adjacency

    def verify_navigability(self, adjacency: np.ndarray,using_communicability:bool=False) -> Union[bool, float]:
        """
        Verify that the network enables successful greedy routing between all node pairs.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j
            using_communicability (bool): If True, just uses communicability matrix to verify overall navigability

        Returns:
            either bool: True if greedy routing succeeds between all node pairs, False otherwise
            or float: how much of the network is navigable
        """
        if using_communicability:
            return (scipy.linalg.expm(adjacency).astype(bool).sum())/(adjacency.shape[0]**2)
        else:
            def can_route(start: int, target: int) -> bool:
                """
                Check if greedy routing can reach from start to target node.

                Args:
                    start (int): Starting node index
                    target (int): Target node index

                Returns:
                    bool: True if route exists, False otherwise
                """
                if start == target:
                    return True

                current_node = start
                visited_nodes = {current_node}

                while True:
                    # Find unvisited neighbors of current node
                    unvisited_neighbors = [
                        node for node in range(self.n_nodes)
                        if adjacency[current_node, node] and node not in visited_nodes
                    ]

                    # If no unvisited neighbors, routing fails
                    if not unvisited_neighbors:
                        return False

                    # Select neighbor closest to target
                    next_node = min(
                        unvisited_neighbors,
                        key=lambda node: self.distances[node, target]
                    )

                    # Check if we've reached the target
                    if next_node == target:
                        return True

                    # Check if we're getting closer to target
                    if self.distances[next_node, target] >= self.distances[current_node, target]:
                        return False  # Routing fails if not getting closer

                    current_node = next_node
                    visited_nodes.add(current_node)

                    # Prevent infinite loops
                    if len(visited_nodes) == self.n_nodes:
                        return False

            # Verify routing between all node pairs
            for source in range(self.n_nodes):
                for destination in range(self.n_nodes):
                    if source != destination and not can_route(source, destination):
                        return False
            return True