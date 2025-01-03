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

        for source in range(self.n_nodes):
            for target in range(self.n_nodes):
                if source == target:
                    continue
                    
                # A node covers another node if it's closer to the target
                # Check if any other node could serve as a better next hop
                has_better_hop = False
                for via_node in range(self.n_nodes):
                    if via_node == source or via_node == target:
                        continue
                        
                    # If we find a node that's closer to the target than the source
                    # and is also closer to the target than to the source
                    if (self.distances[via_node, target] < self.distances[source, target] and
                        self.distances[source, via_node] < self.distances[source, target]):
                        has_better_hop = True
                        break
                
                if not has_better_hop:
                    frame_adjacency[source, target] = True
        
        return frame_adjacency

    def _can_reach_greedily(self, source: int, target: int, current_adjacency: np.ndarray) -> bool:
        """Check if source can reach target through greedy routing."""
        current = source
        visited = {current}
        
        while current != target:
            # Find unvisited neighbors closer to target
            neighbors = np.where(current_adjacency[current])[0]
            valid_next = [n for n in neighbors 
                        if n not in visited and 
                        self.distances[n, target] < self.distances[current, target]]
            
            if not valid_next:
                return False
                
            current = min(valid_next, key=lambda x: self.distances[x, target])
            visited.add(current)
            
            if len(visited) > self.n_nodes:  # Prevent infinite loops
                return False
                
        return True

    def _get_coverage_sets(self, u: int) -> dict:
        """
        For node u, compute S_u_v for each possible next hop v.
        S_u_v = {w|d(v,w) < d(u,w)} means:
        nodes w that v would be a good next hop for when u wants to reach them
        (because v is closer to w than u is)
        
        Example from paper for node B:
        S_B_A = {A,D} means A is a good next hop for B to reach A and D
        because d(A,w) < d(B,w) for w in {A,D}
        """
        coverage_sets = {}
        
        if u == 0:  # Debug for node A
            print("\nDistance comparisons for node A:")
        
        for v in range(self.n_nodes):  # v is potential next hop
            if v == u:
                continue
                
            # S_u_v contains nodes that v helps u reach
            S_u_v = set()
            if u == 0:
                print(f"\nComparing for potential next hop v={v}:")
                
            for w in range(self.n_nodes):  # w is destination
                # Check if v is a good next hop from u to reach w
                if self.distances[v, w] < self.distances[u, w]:
                    S_u_v.add(w)
                    if u == 0:
                        print(f"  w={w}: d({v},w)={self.distances[v,w]:.2f} < d(A,w)={self.distances[u,w]:.2f}")
                        print(f"  -> {v} helps A reach {w}")
                        
            coverage_sets[v] = S_u_v
            
        return coverage_sets

    def build_nash_equilibrium(self) -> np.ndarray:
        """
        Build Nash equilibrium using minimum set cover.
        Now using properly oriented coverage sets: S_v_u contains nodes closer to u than to v
        """
        adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        
        for u in range(self.n_nodes):  # u is our source node
            coverage_sets = self._get_coverage_sets(u)
            nodes_to_cover = set(range(self.n_nodes)) - {u}
            
            # Build coverage matrix for minimum set cover
            # For each potential connection v, we consider what it can help us reach
            cover_matrix = np.zeros((len(nodes_to_cover), self.n_nodes-1))
            
            # Convert coverage sets to minimum set cover problem
            # If w is in S_v_u, it means u is a better choice than v for reaching w
            # Therefore, v is NOT a good choice for reaching w
            for i, target in enumerate(sorted(nodes_to_cover)):
                col_idx = 0
                for v in sorted(nodes_to_cover):  # potential connections
                    # If target is IN S_u_v, then v helps reach target
                    if target in coverage_sets[v]:
                        cover_matrix[i, col_idx] = 1
                    col_idx += 1
            
            # Debug output for node A
            if u == 0:
                print("\nCoverage matrix for A:")
                print(cover_matrix)
            
            # Solve minimum set cover
            c = np.ones(self.n_nodes-1)
            bounds = [(0, 1) for _ in range(self.n_nodes-1)]
            
            # Ensure each target is covered by at least one connection
            result = linprog(c, A_ub=-cover_matrix, b_ub=-np.ones(len(nodes_to_cover)), 
                            bounds=bounds, method='highs')
            
            if result.success:
                # Convert solution back to node indices
                selected = np.zeros(self.n_nodes, dtype=bool)
                node_list = [v for v in range(self.n_nodes) if v != u]
                selected[node_list] = result.x > 0.5
                adjacency[u] = selected
            else:
                raise ValueError(f"LP failed for source {u}. Coverage matrix:\n{cover_matrix}")
                
        return adjacency

    def verify_navigability(self, adjacency: np.ndarray, using_communicability:bool=False, verbose:bool = False) -> bool:
        """
        Verify that the network is fully navigable using either greedy routing (local information) or its communicability matrix (global information).
        If there is no holes in the communicability matrix, it means all nodes eventually can reach each other so the network is fully navigable BOOM.
        However, note that in Nash equilibria, the network is expected to be fully navigable using local information,
        so the communicability matrix is not used to verify equilibrium conditions.
        See: Zamora-López, G., & Gilson, M. (2024). An integrative dynamical perspective for graph theory and the analysis of complex networks. Chaos, 34(4). https://doi.org/10.1063/5.0202241

        Args:
            adjacency (np.ndarray): Boolean/Binary adjacency matrix with shape (n_nodes, n_nodes)
                where adjacency[i, j] = True (or 1) indicates an edge from node i to node j.
            using_communicability (bool): If True, use the communicability matrix to verify navigability.
            verbose (bool): If True, print additional information for debugging.

        Returns:
            bool: True if the network is fully navigable, False otherwise.
        """
        if using_communicability:
            propagation_matrix = scipy.linalg.expm(adjacency).astype(bool)
            total_possible_interactions = self.n_nodes**2
            total_reachable = propagation_matrix.sum()
            proportion_reachable = total_reachable / total_possible_interactions
            return proportion_reachable == 1.0
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
                        if verbose:
                            print(f"Routing failed from node {start} to {target}: No unvisited neighbors from node {current_node}.")
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
                    if self.distances[next_node, target] > self.distances[current_node, target]:
                        if verbose:
                            print(f"Routing failed from node {start} to {target}: Next node {next_node} is not closer to target.")
                        return False  # Routing fails if not getting closer

                    current_node = next_node
                    visited_nodes.add(current_node)

                    # Prevent infinite loops
                    if len(visited_nodes) == self.n_nodes:
                        if verbose:
                            print(f"Routing failed from node {start} to {target}: Visited all nodes without reaching target.")
                        return False

        # Verify routing between all node pairs
        for source in range(self.n_nodes):
            for destination in range(self.n_nodes):
                if source != destination and not can_route(source, destination):
                    if verbose:
                        print(f"Failed to route from {source} to {destination}.")
                    return False
        return True
        

    
    def verify_cost_optimization(self, adjacency: np.ndarray) -> bool:
        """
        Check that removing any edge from a node increases its individual cost.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix representing the network.

        Returns:
            bool: True if the network satisfies cost optimization for all nodes, False otherwise.
        """
        for source in range(self.n_nodes):
            for target in range(self.n_nodes):
                if adjacency[source, target]:
                    modified_adjacency = adjacency.copy()
                    modified_adjacency[source, target] = False
                    if self.verify_navigability(modified_adjacency):
                        # Removing the edge didn't break navigability, which means cost is not optimized
                        return False
        return True

    def verify_unilateral_improvement(self, adjacency: np.ndarray) -> bool:
        """
        Check that no node can reduce its number of edges while maintaining full navigability.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix representing the network.

        Returns:
            bool: True if no node can improve its situation unilaterally, False otherwise.
        """
        for source in range(self.n_nodes):
            for target in range(self.n_nodes):
                if adjacency[source, target]:
                    modified_adjacency = adjacency.copy()
                    modified_adjacency[source, target] = False
                    if self.verify_navigability(modified_adjacency):
                        # If navigability is maintained after removing an edge, it means the node could unilaterally improve
                        return False
        return True

    def verify_nash_equilibrium(self, adjacency: np.ndarray) -> bool:
        """
        Verify that the given network configuration satisfies all conditions of a Nash equilibrium.

        Args:
            adjacency (np.ndarray): Boolean adjacency matrix representing the network.

        Returns:
            bool: True if the network is a valid Nash equilibrium, False otherwise.
        """
        return (
            self.verify_navigability(adjacency)
            and self.verify_cost_optimization(adjacency)
            and self.verify_unilateral_improvement(adjacency)
        )
