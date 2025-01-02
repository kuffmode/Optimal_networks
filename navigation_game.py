"""
Optimally Navigable Networks

This module is a rough* implementation of the work done by [1]. Given some coordinates in the Euclidean space,
it provides the "backbone" network that is minimally wired while being 100% navigable, thus, being a Nash equilibrium of
the navigation game. 

The network is represented as:
- Nodes with coordinates in n-dimensional space
- Edges between nodes that enable navigation
- Distance metrics based on Euclidean distance
- Nash equilibrium properties for optimal connectivity

Classes:
    NavigableNetwork: Main class implementing the navigable network functionality


[1] Gulyás, A., Bíró, J. J., Kőrösi, A., Rétvári, G., & Krioukov, D. (2015). Navigable networks as Nash equilibria of navigation games. Nat. Commun., 6(1), 7651. https://doi.org/10.1038/ncomms8651
* I'm calling it rough implementation because I read the paper, used Claude Sonnet, and implemented the relevant parts for brain networks.
* The paper itself has a lot more to offer.
"""

from typing import Set
import numpy as np
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
            raise ValueError("Coordinates array cannot be empty, who are you fooling?")
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
        Get minimal set of nodes needed by the source node to enable navigation to all targets.

        For each target node, we need at least one neighbor node where:
        distance(neighbor, target) <= distance(source, target)

        Args:
            source_node (int): Index of the source node

        Returns:
            Set[int]: Set of node indices that form the minimal cover set
        """
        required_neighbors: Set[int] = set()
        
        # Examine each potential target node
        for target_node in range(self.n_nodes):
            # Skip if target is the source node itself
            if target_node == source_node:
                continue
                
            # Find all nodes that can serve as intermediate points to reach the target
            potential_covers = [
                neighbor_node for neighbor_node in range(self.n_nodes) 
                if (neighbor_node != source_node and  # not the source node itself
                    # can provide shorter or equal path to target
                    self.distances[neighbor_node, target_node] <= self.distances[source_node, target_node])
            ]
            
            # Skip if no nodes can provide better coverage
            if not potential_covers:
                continue
                
            # Select the node closest to the target as the optimal cover
            best_cover = min(
                potential_covers,
                key=lambda node: self.distances[node, target_node]
            )
            required_neighbors.add(best_cover)
            
        return required_neighbors

    def build_nash_equilibrium(self) -> np.ndarray:
        """
        Build the Nash equilibrium network configuration.

        Constructs an adjacency matrix representing the network configuration
        where no node can improve its routing capability by changing its
        connections unilaterally.

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
