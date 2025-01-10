import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
from skopt.space import Real
from skopt import gp_minimize
import networkx as nx
from scipy.stats import ks_2samp
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

class SimulationMode(Enum):
    """Defines the simulation mode for parameter evolution."""
    FIXED = "fixed"          # Parameters remain constant
    DEVELOPMENTAL = "dev"    # Parameters change over time

@dataclass
class NetworkParameters:
    """Parameters for the network simulation.
    
    Attributes:
        mode: SimulationMode determining if parameters are fixed or developmental
        alpha: Weight of resistance distance in payoff
        beta: Weight of wiring cost (used in FIXED mode)
        beta_infinity: Maximum beta value (used in DEVELOPMENTAL mode)
        tau_beta: Timescale for beta growth (used in DEVELOPMENTAL mode)
        temperature: Fixed temperature value (used in FIXED mode)
        t0: Initial temperature (used in DEVELOPMENTAL mode)
        tau_t: Timescale for temperature decay (used in DEVELOPMENTAL mode)
        connectivity_penalty: Penalty for disconnected components (M)
        n_flips_per_iteration: Number of edge flips attempted per iteration
        seed: Random seed for reproducibility
    """
    mode: SimulationMode = SimulationMode.DEVELOPMENTAL
    alpha: float = 1.0
    # Fixed mode parameters
    beta: float = 2.0
    temperature: float = 0.1
    # Developmental mode parameters
    beta_infinity: float = 2.0
    tau_beta: float = 50.0
    t0: float = 1.0
    tau_t: float = 50.0
    # Common parameters
    connectivity_penalty: float = 100.0
    n_flips_per_iteration: int = 10
    seed: Optional[int] = 42

    def get_beta(self, t: int) -> float:
        """Get beta value for given timestep."""
        if self.mode == SimulationMode.FIXED:
            return self.beta
        return self.beta_infinity * (1 - np.exp(-t / self.tau_beta))
    
    def get_temperature(self, t: int) -> float:
        """Get temperature value for given timestep."""
        if self.mode == SimulationMode.FIXED:
            return self.temperature
        return self.t0 * np.exp(-t / self.tau_t)

@njit()
def _compute_component_size_numba(i: int, adj: np.ndarray, n_nodes: int) -> int:
    """Compute size of connected component containing node i."""
    # Initialize visited array
    visited = np.zeros(n_nodes, dtype=np.bool_)
    queue = np.zeros(n_nodes, dtype=np.int64)
    
    # Start BFS from node i
    queue_start = 0
    queue_end = 1
    queue[0] = i
    visited[i] = True
    component_size = 1
    
    # BFS loop
    while queue_start < queue_end:
        current = queue[queue_start]
        queue_start += 1
        
        # Check all possible neighbors
        for neighbor in range(n_nodes):
            if adj[current, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                queue[queue_end] = neighbor
                queue_end += 1
                component_size += 1
                
    return component_size

@njit()
def _compute_node_payoff_numba(i: int,
                             adj: np.ndarray,
                             distances: np.ndarray,
                             alpha: float,
                             beta: float,
                             temp: float,
                             connectivity_penalty: float,
                             n_nodes: int,
                             omega0: float) -> float:
    """Numba-optimized version of node payoff computation."""
    # Compute weighted Laplacian
    W = np.zeros_like(distances)
    for i1 in range(n_nodes):
        for j1 in range(n_nodes):
            if i1 != j1 and adj[i1, j1] == 1:
                W[i1, j1] = 1.0 / (distances[i1, j1] + 1e-12)
    
    d_deg = np.diag(W.sum(axis=1))
    L = d_deg - W
    
    # Compute pseudoinverse using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    mask = eigvals > 1e-10
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[mask] = 1.0 / eigvals[mask]
    L_plus = (eigvecs * eigvals_inv.reshape(1, -1)) @ eigvecs.T
    
    # Compute resistance distances for node i
    Omega_i = L_plus[i, i] + np.diag(L_plus) - 2 * L_plus[i, :]
    
    # Resistance cost
    resistance_cost = np.sum(Omega_i) / omega0
    
    # Wiring cost
    wiring_cost = np.sum(adj[i] * distances[i])
    
    # Connectivity penalty
    comp_size = _compute_component_size_numba(i, adj, n_nodes)
    connectivity_cost = (n_nodes - comp_size) * connectivity_penalty
    
    # Add noise
    noise = temp * np.random.normal()
    
    return -alpha * resistance_cost - beta * wiring_cost - connectivity_cost + noise

def _process_edge_flips_parallel_batched(adj: np.ndarray,
                                    n_flips: int,
                                    n_nodes: int,
                                    params,
                                    beta_t: float,
                                    temp_t: float,
                                    distances: np.ndarray,
                                    omega0: float,
                                    batch_size: int = 4) -> np.ndarray:
    """Process edge flips in parallel batches."""
    adj_copy = adj.copy()
    n_batches = (n_flips + batch_size - 1) // batch_size
    
    for batch in range(n_batches):
        current_batch_size = min(batch_size, n_flips - batch * batch_size)
        
        def process_flip(_):
            i, j = np.random.randint(0, n_nodes, size=2)
            if i == j:
                return None
                
            flip_result = {
                'i': i, 'j': j,
                'accepted': False
            }
            
            # Compute current payoff
            current_payoff = _compute_node_payoff_numba(
                i, adj_copy, distances, params.alpha, beta_t, temp_t,
                params.connectivity_penalty, n_nodes, omega0
            )
            
            # Test flip
            adj_test = adj_copy.copy()
            adj_test[i, j] = 1 - adj_test[i, j]
            adj_test[j, i] = adj_test[i, j]
            
            # Compute new payoff
            new_payoff = _compute_node_payoff_numba(
                i, adj_test, distances, params.alpha, beta_t, temp_t,
                params.connectivity_penalty, n_nodes, omega0
            )
            
            # Decide acceptance
            delta_p = new_payoff - current_payoff
            if delta_p >= 0 or np.random.random() <= np.exp(delta_p / (temp_t + 1e-12)):
                flip_result['accepted'] = True
                
            return flip_result
        
        # Process batch in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_flip)(i) for i in range(current_batch_size)
        )
        
        # Apply accepted flips sequentially
        for result in results:
            if result is not None and result['accepted']:
                i, j = result['i'], result['j']
                adj_copy[i, j] = 1 - adj_copy[i, j]
                adj_copy[j, i] = adj_copy[i, j]
    
    return adj_copy

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
    
    def simulate(self, n_iterations: int) -> np.ndarray:
            """Run optimized network simulation."""
            adjacency_history = np.zeros((self.n_nodes, self.n_nodes, n_iterations), 
                                    dtype=np.int8)
            
            adj = self._initialize_adjacency()
            adjacency_history[:, :, 0] = adj
            
            # Main simulation loop with progress bar
            for t in tqdm(range(1, n_iterations), desc="Simulating network evolution"):
                beta_t = self.params.get_beta(t)
                temp_t = self.params.get_temperature(t)
                
                # Process multiple edge flips in parallel
                adj = _process_edge_flips_parallel_batched(
                    adj, self.params.n_flips_per_iteration,
                    self.n_nodes, self.params, beta_t, temp_t,
                    self.distances, self.omega0
                )
                
                adjacency_history[:, :, t] = adj
                
            return adjacency_history
        
        
@dataclass
class FittingResults:
    """Results from parameter fitting.
    
    Attributes:
        best_params: Best found parameters
        all_params: History of all tried parameters
        all_scores: History of all scores
        best_score: Best achieved score
        convergence_data: Data about optimization convergence
    """
    best_params: dict
    all_params: list
    all_scores: list
    best_score: float
    convergence_data: dict
@dataclass
class ParameterSpace:
    """Definition of parameter ranges for fitting.
    
    Attributes:
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        fixed_params: Dictionary of parameter values to keep fixed (not fit)
    """
    param_ranges: Dict[str, Tuple[float, float]]
    fixed_params: Dict[str, float] = None
    
    def __post_init__(self):
        self.fixed_params = self.fixed_params or {}
        
        # Validate parameters based on mode
        valid_fixed = set(self.param_ranges.keys()) | set(self.fixed_params.keys())
        
        # Fixed mode parameters
        fixed_params = {'alpha', 'beta', 'temperature', 'connectivity_penalty'}
        # Developmental mode parameters
        dev_params = {'alpha', 'beta_infinity', 'tau_beta', 't0', 'tau_t', 'connectivity_penalty'}
        
        if not (valid_fixed <= fixed_params or valid_fixed <= dev_params):
            raise ValueError(
                f"Invalid parameter names. Must be subset of {fixed_params} "
                f"for FIXED mode or {dev_params} for DEVELOPMENTAL mode."
            )

class NetworkFitter:
    """Fits network parameters to match empirical data."""
    
    def __init__(self, 
                 coordinates: np.ndarray,
                 empirical_adj: np.ndarray,
                 parameter_space: ParameterSpace,
                 n_iterations: int = 300,
                 mode: SimulationMode = SimulationMode.FIXED,
                 n_trials: int = 50,
                 random_state: Optional[int] = None):
        """
        Args:
            coordinates: Node coordinates
            empirical_adj: Empirical adjacency matrix to match
            parameter_space: Definition of parameters to fit and their ranges
            n_iterations: Number of iterations for each simulation
            mode: Simulation mode (FIXED or DEVELOPMENTAL)
            n_trials: Number of optimization trials
            random_state: Random seed
        """
        self.coordinates = coordinates
        self.empirical_adj = empirical_adj
        self.parameter_space = parameter_space
        self.n_iterations = n_iterations
        self.mode = mode
        self.n_trials = n_trials
        self.random_state = random_state
        self.euclidean_distance = squareform(pdist(coordinates))
        
        # Validate parameter space against mode
        self._validate_parameter_space()
        
        # Pre-compute empirical metrics
        self._compute_empirical_metrics()
    
    def _validate_parameter_space(self):
        """Ensure parameter space is valid for chosen mode."""
        if self.mode == SimulationMode.FIXED:
            valid_params = {'alpha', 'beta', 'temperature', 'connectivity_penalty'}
        else:
            valid_params = {'alpha', 'beta_infinity', 'tau_beta', 't0', 
                          'tau_t', 'connectivity_penalty'}
            
        all_params = (set(self.parameter_space.param_ranges.keys()) | 
                     set(self.parameter_space.fixed_params.keys()))
                     
        if not all_params <= valid_params:
            raise ValueError(f"Invalid parameters for mode {self.mode}. "
                           f"Valid parameters are: {valid_params}")
    
    def _compute_empirical_metrics(self):
        """Pre-compute metrics for empirical network."""
        # Degrees
        self.empirical_degrees = np.sum(self.empirical_adj, axis=0)
        
        # Clustering coefficient
        self.empirical_clustering = list(nx.clustering(
            nx.from_numpy_array(self.empirical_adj)
        ).values())
        
        # Betweenness centrality
        self.empirical_betweenness = list(nx.betweenness_centrality(
            nx.from_numpy_array(self.empirical_adj)
        ).values())
        
        # Edge distances
        self.empirical_distances = self.euclidean_distance[
            np.triu(self.empirical_adj, 1) > 0
        ]
    
    def _evaluate_network(self, synthetic_adj: np.ndarray) -> float:
        """Compute similarity score between synthetic and empirical networks."""
        # Get synthetic metrics
        degrees_synthetic = np.sum(synthetic_adj, axis=0)
        
        clustering_synthetic = nx.clustering(nx.from_numpy_array(synthetic_adj))
        
        betweenness_synthetic = nx.betweenness_centrality(nx.from_numpy_array(synthetic_adj))
        
        distance_synthetic = self.euclidean_distance[np.triu(synthetic_adj, 1) > 0]
        
        # Compute KS statistics
        ks_scores = [
            ks_2samp(degrees_synthetic, self.empirical_degrees)[0],
            ks_2samp(list(clustering_synthetic.values()), self.empirical_clustering)[0],
            ks_2samp(list(betweenness_synthetic.values()), self.empirical_betweenness)[0],
            ks_2samp(distance_synthetic, self.empirical_distances)[0]
        ]
        
        return np.max(ks_scores)
    
    def _objective(self, params: list) -> float:
        """Objective function for optimization."""
        # Combine fitted and fixed parameters
        param_dict = dict(zip(self.parameter_space.param_ranges.keys(), params))
        param_dict.update(self.parameter_space.fixed_params)
        
        # Create network parameters
        if self.mode == SimulationMode.FIXED:
            network_params = NetworkParameters(
                mode=SimulationMode.FIXED,
                alpha=param_dict.get('alpha', 1.0),
                beta=param_dict.get('beta', 2.0),
                temperature=param_dict.get('temperature', 0.1),
                connectivity_penalty=param_dict.get('connectivity_penalty', 100.0)
            )
        else:
            network_params = NetworkParameters(
                mode=SimulationMode.DEVELOPMENTAL,
                alpha=param_dict.get('alpha', 1.0),
                beta_infinity=param_dict.get('beta_infinity', 2.0),
                tau_beta=param_dict.get('tau_beta', 50.0),
                t0=param_dict.get('t0', 1.0),
                tau_t=param_dict.get('tau_t', 50.0),
                connectivity_penalty=param_dict.get('connectivity_penalty', 100.0)
            )
        
        # Run simulation and evaluate
        network = DevelopingNetwork(self.coordinates, params=network_params)
        adj_history = network.simulate(self.n_iterations)
        return self._evaluate_network(adj_history[:, :, -1])
    
    def fit(self) -> FittingResults:
        """Run parameter fitting procedure."""
        # Create optimization space from parameter ranges
        space = [
            Real(low, high, name=name)
            for name, (low, high) in self.parameter_space.param_ranges.items()
        ]
        
        # Run optimization
        with tqdm(total=self.n_trials, desc="Fitting parameters") as pbar:
            def callback(res):
                pbar.update(1)
                
            result = gp_minimize(
                self._objective,
                space,
                n_calls=self.n_trials,
                random_state=self.random_state,
                callback=callback
            )
        
        # Package results
        param_names = list(self.parameter_space.param_ranges.keys())
        best_params = dict(zip(param_names, result.x))
        best_params.update(self.parameter_space.fixed_params)
        
        all_params = [dict(zip(param_names, x)) for x in result.x_iters]
        for params in all_params:
            params.update(self.parameter_space.fixed_params)
        
        return FittingResults(
            best_params=best_params,
            all_params=all_params,
            all_scores=result.func_vals,
            best_score=result.fun,
            convergence_data={
                'models': result.models,
                'space': result.space,
                'specs': result.specs
            }
        )