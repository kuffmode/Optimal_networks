import numpy as np
from typing import Set, Optional
from scipy.spatial.distance import pdist, squareform

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

    def _get_minimal_covers(self, source_node: int) -> Set[int]:
        """
        Compute the minimal set of neighbors needed for coverage.
        Args:
            source_node (int): Index of the source node.
        Returns:
            Set[int]: Indices of nodes forming the minimal cover set.
        """
        required_neighbors: Set[int] = set()
        uncovered_targets: Set[int] = set(range(self.n_nodes)) - {source_node}

        while uncovered_targets:
            best_neighbor = None
            best_covered_targets = set()

            for potential_neighbor in range(self.n_nodes):
                if potential_neighbor == source_node or potential_neighbor in required_neighbors:
                    continue

                # Calculate targets this neighbor can cover
                covered_targets = {
                    target for target in uncovered_targets
                    if self.distances[potential_neighbor, target] <= self.distances[source_node, target]
                }

                # Prioritize the closest neighbor that covers targets
                if (
                    len(covered_targets) > len(best_covered_targets)
                    or (len(covered_targets) == len(best_covered_targets)
                        and (best_neighbor is None or self.distances[source_node, potential_neighbor] < self.distances[source_node, best_neighbor]))
                ):
                    best_neighbor = potential_neighbor
                    best_covered_targets = covered_targets

            if not best_covered_targets:
                raise ValueError(f"Node {source_node} cannot cover all targets. Check input data or logic.")

            # Add the best neighbor to the required set and update uncovered targets
            required_neighbors.add(best_neighbor)
            uncovered_targets -= best_covered_targets

        return required_neighbors



    def build_nash_equilibrium(self, symmetry: Optional[str] = None) -> np.ndarray:
        """
        Build the Nash equilibrium network configuration.

        Constructs an adjacency matrix representing the network configuration
        where no node can improve its routing capability by changing its
        connections unilaterally. Optionally enforces symmetry.

        Args:
            symmetry (Optional[str]): Determines how symmetry is enforced. Can be:
                - None: No symmetry enforcement (default).
                - 'forced': Explicitly enforce symmetry by adding mutual edges.

        Returns:
            np.ndarray: Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j
        """
        # Initialize adjacency matrix
        adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        # For each node, establish connections to its required neighbors
        for current_node in range(self.n_nodes):
            optimal_neighbors = self._get_minimal_covers(current_node)
            for neighbor in optimal_neighbors:
                adjacency[current_node, neighbor] = True

        if symmetry == 'forced':
            # Enforce symmetry by making the matrix symmetric
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if adjacency[i, j] or adjacency[j, i]:
                        adjacency[i, j] = adjacency[j, i] = True



        return adjacency

    def verify_navigability(self, adjacency: np.ndarray) -> bool:
        """
        Verify that the network enables successful greedy routing between all node pairs.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j

        Returns:
            bool: True if greedy routing succeeds between all node pairs, False otherwise
        """
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