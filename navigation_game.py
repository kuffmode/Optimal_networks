import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
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



@dataclass
class NetworkParameters:
    """Parameters for the network development simulation.
    
    Attributes:
        alpha (float): Weight of resistance distance in payoff
        beta_infinity (float): Maximum value of beta (wiring cost weight)
        tau_beta (float): Timescale for beta growth
        t0 (float): Initial temperature
        tau_t (float): Timescale for temperature decay
        connectivity_penalty (float): Penalty for disconnected components (M)
        n_flips_per_iteration (int): Number of edge flips attempted per iteration
        seed (Optional[int]): Random seed for reproducibility
    """
    alpha: float = 1.0
    beta_infinity: float = 2.0
    tau_beta: float = 50.0
    t0: float = 1.0
    tau_t: float = 50.0
    connectivity_penalty: float = 100.0
    n_flips_per_iteration: int = 10
    seed: Optional[int] = 11

class DevelopingNetwork:
    """A class implementing network development through simulated annealing.
    
    The network evolves to minimize a combination of resistance distance,
    wiring cost, and connectivity penalties through stochastic optimization.
    """
    
    def __init__(self, coordinates: NDArray[np.float64], params: NetworkParameters):
        """Initialize the network with node coordinates.
        
        Args:
            coordinates: Array of shape (n_nodes, dimension) containing 
                       spatial coordinates of nodes
        """
        if coordinates.ndim != 2:
            raise ValueError("Coordinates must be a 2D array with shape (n_nodes, dimension)")
        self.params = params
        self.coordinates = coordinates
        self.n_nodes = len(coordinates)
        # Pre-compute normalized distance matrix
        self.distances = squareform(pdist(coordinates)) / np.sqrt(len(coordinates[0]))
        # Pre-compute normalization factor for resistance distances
        self.omega0 = float(self.n_nodes)
        
    def _initialize_adjacency(self) -> NDArray[np.int8]:
        """Initialize adjacency matrix with a ring structure plus random edges.
        
        Returns:
            Binary adjacency matrix of shape (n_nodes, n_nodes)
        """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int8)
        # Create ring structure
        idx = np.arange(self.n_nodes)
        adj[idx, (idx + 1) % self.n_nodes] = 1
        adj[(idx + 1) % self.n_nodes, idx] = 1
        return adj
    
    def _compute_laplacian(self, adj: NDArray[np.int8]) -> NDArray[np.float64]:
        """Compute the weighted Laplacian matrix.
        
        Args:
            adj: Binary adjacency matrix
            
        Returns:
            Weighted Laplacian matrix
        """
        # Compute weight matrix W = A_{ij} / D_{ij}
        W = np.divide(adj, self.distances + 1e-12, where=adj.astype(bool))
        # Compute degree matrix and Laplacian
        d_deg = np.diag(W.sum(axis=1))
        return d_deg - W
    
    def _compute_resistance_distances(self, L: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute resistance distances from Laplacian.
        
        Args:
            L: Weighted Laplacian matrix
            
        Returns:
            Matrix of resistance distances between all pairs of nodes
        """
        L_plus = np.linalg.pinv(L)
        diag = np.diag(L_plus)[:, np.newaxis]
        return diag + diag.T - 2 * L_plus
    
    def _compute_component_sizes(self, adj: NDArray[np.int8]) -> NDArray[np.int64]:
        """Compute size of connected component for each node.
        
        Args:
            adj: Binary adjacency matrix
            
        Returns:
            Array containing component size for each node
        """
        # Find connected components using matrix powers
        connected = adj.copy()
        prev_connected = np.zeros_like(adj)
        
        while not np.array_equal(connected, prev_connected):
            prev_connected = connected.copy()
            connected = (connected @ adj > 0) | connected
            
        # Get component sizes
        return connected.sum(axis=1)
    
    def _compute_node_payoff(self, 
                           i: int, 
                           adj: NDArray[np.int8], 
                           beta: float, 
                           temp: float) -> float:
        """Compute payoff for a single node.
        
        Args:
            i: Node index
            adj: Current adjacency matrix
            beta: Current beta value (wiring cost weight)
            temp: Current temperature
            
        Returns:
            Total payoff value for node i
        """
        # Compute resistance distances
        L = self._compute_laplacian(adj)
        Omega = self._compute_resistance_distances(L)
        
        # Resistance cost
        resistance_cost = Omega[i].sum() / self.omega0
        
        # Wiring cost 
        wiring_cost = (adj[i] * self.distances[i]).sum()
        
        # Connectivity penalty
        comp_sizes = self._compute_component_sizes(adj)
        connectivity_cost = (self.n_nodes - comp_sizes[i]) * self.params.connectivity_penalty
        
        # Add noise
        noise = temp * np.random.randn()
        
        return -self.params.alpha * resistance_cost - beta * wiring_cost - connectivity_cost + noise

    def simulate(self, 
                n_iterations: int) -> NDArray[np.int8]:
        """Run network development simulation.
        
        Args:
            params: Simulation parameters
            n_iterations: Number of iterations to simulate
            
        Returns:
            Array of shape (n_nodes, n_nodes, n_iterations) containing
            adjacency matrix at each iteration
        """
        # Set random seed if specified
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
            
        # Initialize storage for adjacency matrices
        adjacency_history = np.zeros((self.n_nodes, self.n_nodes, n_iterations), 
                                   dtype=np.int8)
        
        # Initialize first adjacency matrix
        adj = self._initialize_adjacency()
        adjacency_history[:, :, 0] = adj
        
        # Main simulation loop
        for t in range(1, n_iterations):
            # Update temperature and beta
            beta_t = self.params.beta_infinity * (1 - np.exp(-t / self.params.tau_beta))
            temp_t = self.params.t0 * np.exp(-t / self.params.tau_t)
            
            # Attempt edge flips
            for _ in range(self.params.n_flips_per_iteration):
                # Select random node pair
                i, j = np.random.randint(0, self.n_nodes, size=2)
                if i == j:
                    continue
                    
                # Store current payoff
                current_payoff = self._compute_node_payoff(i, adj, beta_t, temp_t)
                
                # Flip edge
                adj[i, j] = 1 - adj[i, j]
                adj[j, i] = adj[i, j]
                
                # Compute new payoff
                new_payoff = self._compute_node_payoff(i, adj, beta_t, temp_t)
                
                # Accept or reject flip
                delta_p = new_payoff - current_payoff
                if delta_p < 0 and np.random.random() > np.exp(delta_p / (temp_t + 1e-12)):
                    # Reject: revert flip
                    adj[i, j] = 1 - adj[i, j]
                    adj[j, i] = adj[i, j]
            
            # Store current adjacency
            adjacency_history[:, :, t] = adj
            
        return adjacency_history