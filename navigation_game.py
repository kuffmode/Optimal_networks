import numpy as np
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
