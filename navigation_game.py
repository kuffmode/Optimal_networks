from typing import Union
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog

class NavigableNetwork:
    """
    A class representing a navigable network with Nash equilibrium properties.

    This class implements a network where nodes are positioned in coordinate space (e.g., 2D or 3D Euclidean space).
    They then establish connections to reach total navigation between with minimal wiring cost.
    The resulting network is directed, fully reachable from any node to any other node, and can have cycles.
    See Gulyás, A., Bíró, J. J., Kőrösi, A., Rétvári, G., & Krioukov, D. (2015). Navigable networks as Nash equilibria of navigation games. Nat. Commun., 6(1), 7651. https://doi.org/10.1038/ncomms8651

    Attributes:
        coordinates (np.ndarray): Array of node coordinates in n-dimensional space.
            Shape: (n_nodes, n_dimensions)
        n_nodes (int): Number of nodes in the network
        distances (np.ndarray): Matrix of pairwise Euclidean distances between all nodes.
            Shape: (n_nodes, n_nodes)
    """

    def __init__(self, coordinates: np.ndarray):
        """
        Initialize the NavigableNetwork with node coordinates.

        Args:
            coordinates (np.ndarray): Array of node coordinates in n-dimensional Euclidean space.
                Shape: (n_nodes, n_dimensions)
        """
        if len(coordinates) == 0:
            raise ValueError("Coordinates array cannot be empty. U playin homie?")
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

        Frame edges are defined as edges that must exist because no other node can replace them.

        Returns:
            np.ndarray: Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i, j] = True indicates that the edge i -> j is a frame edge.
        """
        
        # Start with an empty adjacency matrix
        frame_adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)

        # Go through all node pairs and skip the diagonal
        for target in range(self.n_nodes):
            for source in range(self.n_nodes):
                if source == target:
                    continue

                # Check if any other node can replace source as a greedy next-hop to target.
                # Here we're assuming that the source node is the best choice for the target node unless proven otherwise.
                is_frame_edge = True
                for other in range(self.n_nodes):
                    if other == source or other == target:
                        continue
                    
                    # Here's the proof. Some other node is closer to the target than the source.
                    if self.distances[other, target] < self.distances[source, target]:
                        is_frame_edge = False
                        break
                    
                # If no other node is closer to the target than the source, the edge was indeed a frame edge.
                if is_frame_edge:
                    frame_adjacency[source, target] = True

        return frame_adjacency

    def build_nash_equilibrium(self) -> np.ndarray:
        """
        Build the Nash equilibrium network of the navigation game using linear programming.

        Constructs an adjacency matrix representing the network configuration
        where no node can improve its routing capability by changing its
        connections unilaterally. The local constraints are wiring cost (distance) and navigability.
        See Gulyás, A., Bíró, J. J., Kőrösi, A., Rétvári, G., & Krioukov, D. (2015). Navigable networks as Nash equilibria of navigation games. Nat. Commun., 6(1), 7651. https://doi.org/10.1038/ncomms8651


        Returns:
            np.ndarray: Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j
        """
        # Start with frame edges, meaning that the network is navigable at least in the frame topology.
        adjacency = self._compute_frame_edges()

        # Solve the Nash equilibrium using linear programming (minimizing the cost of wiring)
        for source_node in range(self.n_nodes):
            n_vars = self.n_nodes
            # Objective function: minimize the number of edges
            c = np.ones(n_vars)

            # Constraints: each target node must be reachable from the source node, so, full navigability.
            A = []
            b = []
            for target_node in range(self.n_nodes):
                if target_node == source_node:
                    continue
                
                # Constraint: target is covered by at least one neighbor
                constraint = np.zeros(n_vars)
                for neighbor in range(self.n_nodes):
                    if neighbor == source_node:
                        continue
                    if self.distances[neighbor, target_node] <= self.distances[source_node, target_node]:
                        constraint[neighbor] = 1

                A.append(constraint)
                b.append(1)

            # Bounds: each edge is either present or not.
            bounds = [(0, 1) for _ in range(n_vars)]
            result = linprog(c, A_ub=-np.array(A), b_ub=-np.array(b), bounds=bounds, method='highs')

            # Hopefully, the linear programming solver found a solution (it's a convex problem).
            if result.success:
                selected_neighbors = np.where(result.x > 0.5)[0]
                adjacency[source_node, selected_neighbors] = True
            else:
                raise ValueError(f"Linear programming failed for source node {source_node}. Or I did with the code.")

        return adjacency

    def verify_navigability(self, adjacency: np.ndarray, using_communicability:bool=False) -> Union[bool, float]:
        """
        Verify that the network enables successful greedy routing between all node pairs or just checks overall navigability.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i,j] = True indicates an edge from node i to node j
            using_communicability (bool): If True, just uses communicability matrix to verify overall navigability

        Returns:
            either bool: True if greedy routing succeeds between all node pairs, False otherwise
            or float: how much of the network is navigable
        """
        # The trick here is that we can use the communicability matrix to check overall navigability.
        # If there's a zero in the communicability matrix, the network is not fully navigable.
        if using_communicability:
            return (scipy.linalg.expm(adjacency).astype(bool).sum())/(adjacency.shape[0]**2)
        
        # Otherwise, we'll check greedy routing between all node pairs. Sure.
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
                    return True # Duh

                current_node = start
                visited_nodes = {current_node}

                while True:
                    # Find unvisited neighbors of the current node
                    unvisited_neighbors = [
                        node for node in range(self.n_nodes)
                        if adjacency[current_node, node] and node not in visited_nodes
                    ]

                    # If there's no unvisited neighbors, routing fails :(
                    if not unvisited_neighbors:
                        return False

                    # Select the neighbor closest to the target
                    next_node = min(
                        unvisited_neighbors,
                        key=lambda node: self.distances[node, target]
                    )

                    # Check if we've reached the target
                    if next_node == target:
                        return True

                    # Check if we're getting closer to target, if not, BOOM, routing fails
                    if self.distances[next_node, target] >= self.distances[current_node, target]:
                        return False

                    # Move to the next node
                    current_node = next_node
                    visited_nodes.add(current_node)

                    # Prevent infinite loops. Nobody likes infinite loops.
                    if len(visited_nodes) == self.n_nodes:
                        return False

            # Verify routing between all node pairs
            for source in range(self.n_nodes):
                for destination in range(self.n_nodes):
                    if source != destination and not can_route(source, destination):
                        return False
            return True
    
    def verify_equilibrium(self, adjacency: np.ndarray) -> bool:
        """
        Verify that a given adjacency matrix represents a fully navigable but minimally wired network.

        The method first checks if the network is navigable. Then, it iteratively removes each link
        and checks whether the network remains navigable. If the network remains navigable after any removal,
        it is not minimally wired and thus, we're not in Nash equilibrium.

        Args:
            adjacency (np.ndarray): Boolean/Binary adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i, j] = True indicates an edge from node i to node j.

        Returns:
            bool: True if the network is minimally wired (removing any link breaks navigability),
                  False otherwise.
        """
        # First, verify that the initial network is fully navigable
        if not self.verify_navigability(adjacency):
            print("Initial network is not fully navigable. Go tell your jokes elsewhere.")
            return False

        # Iterate over all edges in the adjacency matrix
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if adjacency[i, j]:
                    # Create a copy of the adjacency matrix and remove the edge
                    modified_adjacency = adjacency.copy()
                    modified_adjacency[i, j] = False

                    # Check if the modified network is still navigable
                    if self.verify_navigability(modified_adjacency):
                        print(f"Network remained navigable after removing edge ({i}, {j}).")
                        return False

        # If removing any edge breaks navigability, the network is minimally wired
        print("The network is minimally wired: removing any link breaks navigability.")
        return True