
from typing import Callable, Union, Literal, TypeAlias, Optional
import numpy as np
import numpy.typing as npt
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# Type Definitions
FloatArray = npt.NDArray[np.float64]
Trajectory: TypeAlias = Union[float, FloatArray]
DistanceMetric = Callable[[FloatArray, FloatArray], FloatArray]
NoiseType = Union[Literal[0], FloatArray]

def validate_parameters(
    sim_length: int,
    *trajectories: Trajectory,
    names: tuple[str, ...],
    allow_float: tuple[bool, ...],
    allow_zero: tuple[bool, ...]
) -> None:
    """
    Validate simulation parameters.
    
    Args:
        sim_length: Length of simulation
        *trajectories: Parameter trajectories to validate
        names: Names of parameters for error messages
        allow_float: Whether each parameter can be float
        allow_zero: Whether each parameter can be zero
    
    Raises:
        ValueError: If any parameter is invalid
        
    Example:
        >>> validate_parameters(
        ...     1000,
        ...     1.0, np.zeros(1000),
        ...     names=('alpha', 'beta'),
        ...     allow_float=(True, False),
        ...     allow_zero=(False, True)
        ... )
    """
    for traj, name, float_ok, zero_ok in zip(
        trajectories, names, allow_float, allow_zero
    ):
        if isinstance(traj, (float, int)):
            if not float_ok:
                raise ValueError(
                    f"{name} must be an array, got {type(traj)}"
                )
            if not zero_ok and traj == 0:
                raise ValueError(f"{name} cannot be zero")
        elif isinstance(traj, np.ndarray):
            if traj.shape != (sim_length,):
                raise ValueError(
                    f"{name} trajectory length {len(traj)} doesn't match "
                    f"simulation length {sim_length}"
                )
            if not zero_ok and np.any(traj == 0):
                raise ValueError(f"{name} cannot contain zeros")
        else:
            raise ValueError(
                f"{name} must be float or array, got {type(traj)}"
            )

@njit
def get_param_value(param: Trajectory, t: int) -> float:
    """Get parameter value at time t efficiently."""
    if isinstance(param, float):
        return param
    return param[t]

def normalize_distances(
    coordinates: FloatArray,
    normalization: Literal["sqrt_dim", "max", "mean"] = "sqrt_dim"
) -> FloatArray:
    """
    Compute and normalize pairwise distances.
    
    Args:
        coordinates: Node coordinates (n_nodes, n_dimensions)
        normalization: Normalization method
            - "sqrt_dim": Divide by sqrt(dimensionality)
            - "max": Divide by maximum distance
            - "mean": Divide by mean distance
            
    Returns:
        Normalized distance matrix (n_nodes, n_nodes)
        
    Raises:
        ValueError: If normalization factor is zero or method unknown
    """
    if coordinates.ndim != 2:
        raise ValueError("Coordinates must be 2D array (n_nodes, n_dimensions)")
        
    distances = squareform(pdist(coordinates, metric='euclidean'))
    
    if normalization == "sqrt_dim":
        norm_factor = np.sqrt(coordinates.shape[1])
    elif normalization == "max":
        norm_factor = np.max(distances)
    elif normalization == "mean":
        norm_factor = np.mean(distances)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
        
    if norm_factor == 0:
        raise ValueError("Normalization factor is zero")
        
    return distances / norm_factor

@njit
def compute_component_sizes(adjacency: FloatArray) -> FloatArray:
    """
    Compute size of connected component for each node efficiently.
    Uses numba-optimized BFS to find connected components.
    
    Args:
        adjacency: Adjacency matrix (n_nodes, n_nodes)
    
    Returns:
        Array of component sizes (n_nodes,)
    """
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    sizes = np.zeros(n_nodes, dtype=np.float64)
    
    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
            
        # BFS from start_node
        component = np.zeros(n_nodes, dtype=np.bool_)
        queue = np.zeros(n_nodes, dtype=np.int64)
        queue_size = 1
        queue[0] = start_node
        component[start_node] = True
        
        idx = 0
        while idx < queue_size:
            node = queue[idx]
            idx += 1
            
            for neighbor in range(n_nodes):
                if (adjacency[node, neighbor] and 
                    not component[neighbor]):
                    component[neighbor] = True
                    queue[queue_size] = neighbor
                    queue_size += 1
        
        # Update all nodes in component
        comp_size = float(queue_size)
        for i in range(queue_size):
            node = queue[i]
            visited[node] = True
            sizes[node] = comp_size
            
    return sizes

@njit
def resistance_distance(adjacency: FloatArray, coordinates: FloatArray) -> FloatArray:
    """
    Compute resistance distances between all pairs of nodes.
    
    Args:
        adjacency: Binary adjacency matrix (n_nodes, n_nodes)
        coordinates: Node coordinates (n_nodes, n_dimensions)
            Note: coordinates are used only for weighting edges
            
    Returns:
        Matrix of resistance distances (n_nodes, n_nodes)
    """
    n_nodes = len(adjacency)
    distances = np.zeros((n_nodes, n_nodes))
    
    # Compute weight matrix (1/euclidean_distance for connected nodes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency[i, j]:
                dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                weight = 1.0 / (dist + 1e-12)  # Avoid division by zero
                distances[i, j] = weight
                distances[j, i] = weight
                
    # Compute weighted Laplacian
    diag = np.sum(distances, axis=1)
    laplacian = np.diag(diag) - distances
    
    # Compute pseudoinverse using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    mask = eigvals > 1e-10  # Filter out numerical zeros
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[mask] = 1.0 / eigvals[mask]
    
    # Compute resistance distances
    L_plus = (eigvecs * eigvals_inv.reshape(1, -1)) @ eigvecs.T
    resistance = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            r = L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j]
            resistance[i,j] = r
            resistance[j,i] = r
            
    return resistance

@njit
def compute_node_payoff(
    node: int,
    adjacency: FloatArray,
    coordinates: FloatArray,
    distance_fn: Callable[[FloatArray, FloatArray], FloatArray],
    alpha: float,
    beta: float,
    noise: float,
    connectivity_penalty: float,
) -> float:
    """
    Compute payoff for a single node.
    
    Args:
        node: Index of node
        adjacency: Current adjacency matrix
        coordinates: Node coordinates
        distance_fn: Function computing distance metric
        alpha: Weight of distance term
        beta: Weight of wiring cost
        noise: Noise value (0 for deterministic)
        connectivity_penalty: Penalty for disconnected components
        skip_connectivity: Whether to skip connectivity computation
        
    Returns:
        Total payoff value
    """
    n_nodes = len(adjacency)
    payoff = 0.0
    
    # Distance term
    if alpha != 0:
        distances = distance_fn(adjacency, coordinates)
        payoff -= alpha * np.sum(distances[node])
        
    # Wiring cost
    if beta != 0:
        euclidean = np.sqrt(np.sum((coordinates[node] - coordinates)**2, axis=1))
        payoff -= beta * np.sum(adjacency[node] * euclidean)
        
    # Connectivity penalty
    if connectivity_penalty != 0:
        comp_sizes = compute_component_sizes(adjacency)
        payoff -= connectivity_penalty * (n_nodes - comp_sizes[node])
        
    # Add noise
    if noise != 0:
        payoff += noise
        
    return payoff

def simulate_network_evolution(
    coordinates: FloatArray,
    n_iterations: int,
    distance_fn: DistanceMetric,
    alpha: Trajectory,
    beta: Trajectory,
    noise: NoiseType,
    connectivity_penalty: Trajectory,
    initial_adjacency: Optional[FloatArray] = None,
    n_jobs: int = -1,
    batch_size: int = 4,
    random_seed: Optional[int] = None
) -> FloatArray:
    """
    Simulate network evolution through payoff optimization.
    
    Args:
        coordinates: Node coordinates (n_nodes, n_dimensions)
        n_iterations: Number of simulation steps
        distance_fn: Function computing distance metric
        alpha: Weight of distance term (float or array)
        beta: Weight of wiring cost (float or array)
        noise: Noise values (0 or array)
        connectivity_penalty: Penalty for disconnected components
        initial_adjacency: Starting adjacency matrix (optional)
        n_jobs: Number of parallel jobs (-1 for all cores)
        batch_size: Number of edge flips to process in parallel
        random_seed: Random seed for reproducibility
        
    Returns:
        History of adjacency matrices (n_nodes, n_nodes, n_iterations)
    """
    # Parameter validation
    validate_parameters(
        n_iterations,
        alpha, beta, noise, connectivity_penalty,
        names=('alpha', 'beta', 'noise', 'connectivity_penalty'),
        allow_float=(True, True, False, True),
        allow_zero=(True, True, True, True)
    )
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_nodes = len(coordinates)
    
    # Initialize adjacency if not provided
    if initial_adjacency is None:
        # Start with ring structure
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        idx = np.arange(n_nodes)
        adjacency[idx, (idx + 1) % n_nodes] = 1
        adjacency[(idx + 1) % n_nodes, idx] = 1
    else:
        adjacency = initial_adjacency.copy()
        
    # Pre-allocate history
    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = adjacency
    
    # Simulation loop with progress bar
    with tqdm(total=n_iterations-1, desc="Simulating network evolution") as pbar:
        for t in range(1, n_iterations):
            # Get current parameters
            alpha_t = get_param_value(alpha, t)
            beta_t = get_param_value(beta, t)
            noise_t = get_param_value(noise, t) if isinstance(noise, np.ndarray) else 0
            penalty_t = get_param_value(connectivity_penalty, t)
            
            # Process edge flips in parallel batches
            def process_flip(_):
                i, j = np.random.randint(0, n_nodes, size=2)
                if i == j:
                    return None
                    
                # Compute current payoff
                current = compute_node_payoff(
                    i, adjacency, coordinates, distance_fn,
                    alpha_t, beta_t, noise_t, penalty_t,
                )
                
                # Test flip
                adj_test = adjacency.copy()
                adj_test[i, j] = 1 - adj_test[i, j]
                adj_test[j, i] = adj_test[i, j]
                
                # Compute new payoff
                new = compute_node_payoff(
                    i, adj_test, coordinates, distance_fn,
                    alpha_t, beta_t, noise_t, penalty_t,
                )
                
                return {
                    'i': i, 'j': j,
                    'accepted': new > current
                }
                
            # Process batch in parallel
            n_flips = batch_size
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_flip)(i) for i in range(n_flips)
            )
            
            # Apply accepted flips
            for result in results:
                if result is not None and result['accepted']:
                    i, j = result['i'], result['j']
                    adjacency[i, j] = 1 - adjacency[i, j]
                    adjacency[j, i] = adjacency[i, j]
                    
            history[:, :, t] = adjacency
            pbar.update(1)
            
    return history