"""Network evolution simulation with Dask-based parallel processing."""
from typing import Any, Callable, Dict, Union, Literal, TypeAlias, Optional, Tuple, List
import numpy as np
import numba
import numpy.typing as npt
from numba.core.errors import TypingError
from numba import njit, jit, NumbaError
import dask.array as da
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from dask_config import DaskConfig, get_or_create_dask_client

numba.config.DISABLE_JIT_WARNINGS = 1

def jit_safe(nopython=True, **jit_kwargs):
    """Safe JIT wrapper with fallback to Python."""
    def decorator(func):
        if not nopython:
            return func
        
        try:
            jitted = jit(nopython=True, **jit_kwargs)(func)
            return jitted
        except (TypingError, NumbaError) as e:
            print(f"Numba could not compile {func.__name__}: {e}")
            return func
    return decorator

@njit
def _diag_indices(n):
    """Get diagonal indices."""
    rows = np.arange(n)
    cols = np.arange(n)
    return rows, cols

@njit
def _set_diagonal(matrix: np.ndarray, value: float = 0.0):
    """Set diagonal elements."""
    n = matrix.shape[0]
    rows, cols = _diag_indices(n)
    for i in range(n):
        matrix[rows[i], cols[i]] = value
    return matrix

@jit_safe()
def process_matrix(matrix):
    """Process matrix by handling special values."""
    n, m = matrix.shape
    result = np.empty_like(matrix)
    for i in range(n):
        for j in range(m):
            val = matrix[i, j]
            if np.isnan(val) or val in (np.inf, -np.inf):
                result[i, j] = 0
            else:
                result[i, j] = val
    return result

@njit
def sample_nodes_spatially(
    n_nodes: int,
    coordinates: npt.NDArray[np.float64],
    center_node: Optional[int],
    sampling_rate: float,
    sigma: float = 1.0
) -> np.ndarray:
    """Sample nodes based on spatial proximity to a center node.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes
    coordinates : npt.NDArray[np.float64]
        Node coordinates array
    center_node : Optional[int]
        Center node for spatial sampling (None for uniform)
    sampling_rate : float
        Fraction of nodes to sample
    sigma : float
        Standard deviation for Gaussian sampling
        
    Returns
    -------
    np.ndarray
        Indices of sampled nodes
    """
    n_samples = max(1, int(n_nodes * sampling_rate))
    
    if center_node is None or center_node < 0:  # Handle None case in numba
        # Uniform sampling
        indices = np.arange(n_nodes)
        np.random.shuffle(indices)
        return indices[:n_samples]
    
    # Compute distances from center node
    distances = np.zeros(n_nodes)
    for i in range(n_nodes):
        distances[i] = np.sqrt(
            np.sum((coordinates[i] - coordinates[center_node])**2)
        )
    
    # Compute Gaussian probabilities
    probs = np.exp(-0.5 * (distances / sigma)**2)
    probs[center_node] = 0  # Exclude center node
    probs = probs / np.sum(probs)  # Normalize
    
    # Sample using probabilities
    sampled = np.zeros(n_samples, dtype=np.int64)
    available = np.ones(n_nodes, dtype=np.bool_)
    
    for i in range(n_samples):
        # Normalize remaining probabilities
        remaining_probs = probs * available
        if np.sum(remaining_probs) == 0:
            # If no valid probabilities left, sample uniformly from remaining nodes
            remaining_indices = np.where(available)[0]
            idx = remaining_indices[
                np.random.randint(0, len(remaining_indices))
            ]
        else:
            remaining_probs = remaining_probs / np.sum(remaining_probs)
            # Sample one index
            cumsum = np.cumsum(remaining_probs)
            idx = np.searchsorted(cumsum, np.random.random())
            
        sampled[i] = idx
        available[idx] = False
        
    return sampled

FloatArray = npt.NDArray[np.float64]
Trajectory: TypeAlias = Union[float, FloatArray, da.Array]
DistanceMetric = Callable[[FloatArray, FloatArray], FloatArray]
NoiseType = Union[Literal[0], FloatArray, da.Array]

def validate_parameters(
    sim_length: int,
    trajectories: tuple[Trajectory, ...],
    names: tuple[str, ...],
    allow_float: tuple[bool, ...],
    allow_zero: tuple[bool, ...]
) -> None:
    """Validate simulation parameters."""
    for traj, name, float_ok, zero_ok in zip(trajectories, names, allow_float, allow_zero):
        if isinstance(traj, (float, int)):
            if not float_ok:
                raise ValueError(f"{name} must be an array, got {type(traj)}")
            if not zero_ok and traj == 0:
                raise ValueError(f"{name} cannot be zero")
        elif isinstance(traj, (np.ndarray, da.Array)):
            if isinstance(traj, da.Array):
                shape = traj.shape[0]
            else:
                shape = traj.shape[0]
            if shape != sim_length:
                raise ValueError(
                    f"{name} trajectory length {shape} doesn't match "
                    f"simulation length {sim_length}"
                )
            if not zero_ok:
                if isinstance(traj, da.Array):
                    if da.any(traj == 0).compute():
                        raise ValueError(f"{name} cannot contain zeros")
                else:
                    if np.any(traj == 0):
                        raise ValueError(f"{name} cannot contain zeros")
        else:
            raise ValueError(f"{name} must be float or array, got {type(traj)}")

def get_param_value(param: Trajectory, t: int) -> float:
    """Get parameter value at time t efficiently."""
    if isinstance(param, (float, int)):
        return float(param)
    elif isinstance(param, da.Array):
        return float(param[t].compute())
    return float(param[t])

def normalize_distances(
    coordinates: FloatArray,
    normalization: Literal["sqrt_dim", "max", "mean"] = "sqrt_dim"
) -> FloatArray:
    """Compute and normalize pairwise distances."""
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
    """Compute size of connected component for each node efficiently."""
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    sizes = np.zeros(n_nodes, dtype=np.float64)
    
    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
            
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
        
        comp_size = float(queue_size)
        for i in range(queue_size):
            node = queue[i]
            visited[node] = True
            sizes[node] = comp_size
            
    return sizes

@jit_safe()
def propagation_distance(adjacency_matrix, coordinates):
    """Compute propagation distance matrix."""
    N = adjacency_matrix.shape[0]
    adjacency_matrix = adjacency_matrix.astype(np.float64)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    adjacency_matrix /= spectral_radius
    
    I = np.eye(N, dtype=np.float64)
    M = I - 0.5 * adjacency_matrix
    inverse_matrix = np.linalg.inv(M)
    propagation_matrix = inverse_matrix @ inverse_matrix.T
    propagation_matrix = _set_diagonal(propagation_matrix, 0)
    propagation_dist = -np.log(propagation_matrix)
    return process_matrix(propagation_dist)

@njit
def resistance_distance(adjacency: FloatArray, coordinates: FloatArray) -> FloatArray:
    """Compute resistance distances between all pairs of nodes."""
    n_nodes = len(adjacency)
    distances = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency[i, j]:
                dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                weight = 1.0 / (dist + 1e-12)
                distances[i, j] = weight
                distances[j, i] = weight
                
    diag = np.sum(distances, axis=1)
    laplacian = np.diag(diag) - distances
    
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    mask = eigvals > 1e-10
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[mask] = 1.0 / eigvals[mask]
    
    L_plus = (eigvecs * eigvals_inv.reshape(1, -1)) @ eigvecs.T
    resistance = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            r = L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j]
            resistance[i,j] = r
            resistance[j,i] = r
            
    return resistance

@jit_safe()
def shortest_path_distance(adjacency_matrix, coordinates):
    """Compute shortest-path distances using Floyd-Warshall algorithm."""
    N = adjacency_matrix.shape[0]
    dist_matrix = adjacency_matrix.astype(np.float64)
    
    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] == 0:
                dist_matrix[i, j] = np.inf
    
    for i in range(N):
        dist_matrix[i, i] = 0.0

    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist_matrix[i, j] = min(
                    dist_matrix[i, j],
                    dist_matrix[i, k] + dist_matrix[k, j]
                )

    return dist_matrix

@jit_safe()
def search_information(W, coordinates):
    """Calculate search information for a memoryless random walker."""
    N = W.shape[0]
    T = W / np.sum(W, axis=1)[:, None]
    
    dist_matrix = W.astype(np.float64)
    p_mat = np.zeros((N, N), dtype=np.int32) - 1

    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] == 0:
                dist_matrix[i, j] = np.inf
            else:
                p_mat[i, j] = i

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
                    p_mat[i, j] = p_mat[k, j]

    SI = np.full((N, N), np.inf)

    for i in range(N):
        for j in range(N):
            if i == j:
                SI[i, j] = 0.0
                continue

            path = []
            current = j
            while current != -1 and current != i:
                path.append(current)
                current = p_mat[i, current]
            if current == i:
                path.append(i)
                path.reverse()
            else:
                continue

            product = 1.0
            for k in range(len(path) - 1):
                product *= T[path[k], path[k + 1]]
            SI[i, j] = -np.log2(product)

    return SI

@jit_safe()
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
    """Compute payoff for a single node."""
    n_nodes = len(adjacency)
    payoff = 0.0
    
    if alpha != 0:
        distances = distance_fn(adjacency, coordinates)
        payoff -= alpha * np.sum(distances[node])
        
    if beta != 0:
        euclidean = np.sqrt(np.sum((coordinates[node] - coordinates)**2, axis=1))
        payoff -= beta * np.sum(adjacency[node] * euclidean)
        
    if connectivity_penalty != 0:
        comp_sizes = compute_component_sizes(adjacency)
        payoff -= connectivity_penalty * (n_nodes - comp_sizes[node])
        
    if noise != 0:
        payoff += noise
        
    return payoff

def process_single_flip(
    adjacency: FloatArray,
    coordinates: FloatArray,
    distance_fn: DistanceMetric,
    alpha_t: float,
    beta_t: float,
    noise_t: float,
    penalty_t: float
) -> Optional[Dict[str, Any]]:
    """Process a single edge flip in the network.
    
    Parameters
    ----------
    adjacency : FloatArray
        Current adjacency matrix
    coordinates : FloatArray
        Node coordinates
    distance_fn : DistanceMetric
        Function to compute distances (e.g., resistance_distance)
    alpha_t : float
        Current alpha parameter value
    beta_t : float
        Current beta parameter value
    noise_t : float
        Current noise parameter value
    penalty_t : float
        Current connectivity penalty value
        
    Returns
    -------
    Optional[Dict[str, Any]]
        None if flip involves same node, otherwise dictionary with:
        - 'i': first node index
        - 'j': second node index
        - 'accepted': whether flip was accepted
    """
    n_nodes = len(adjacency)
    i, j = np.random.randint(0, n_nodes, size=2)
    
    if i == j:
        return None
        
    # Compute current payoff
    current_payoff = compute_node_payoff(
        i, adjacency, coordinates, distance_fn,
        alpha_t, beta_t, noise_t, penalty_t
    )
    
    # Test flip
    adj_test = adjacency.copy()
    adj_test[i, j] = 1 - adj_test[i, j]
    adj_test[j, i] = adj_test[i, j]  # Maintain symmetry
    
    # Compute new payoff
    new_payoff = compute_node_payoff(
        i, adj_test, coordinates, distance_fn,
        alpha_t, beta_t, noise_t, penalty_t
    )
    
    return {
        'i': i,
        'j': j,
        'accepted': new_payoff > current_payoff
    }
    

def simulate_network_evolution(
    coordinates,
    n_iterations,
    distance_fn,
    alpha,
    beta,
    noise,
    connectivity_penalty,
    sampling_rate=0.1,
    sampling_centers=None,
    initial_adjacency=None,
    sigma=1.0,
    dask_config=None,
    random_seed=None
):
    """Simulate network evolution with partial sampling and spatial bias."""
    
    validate_parameters(
        n_iterations,
        trajectories=(alpha, beta, noise, connectivity_penalty, sampling_rate),
        names=('alpha', 'beta', 'noise', 'connectivity_penalty', 'sampling_rate'),
        allow_float=(True, True, False, True, True),
        allow_zero=(True, True, True, True, False)
    )
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_nodes = len(coordinates)
    
    if sampling_centers is None:
        sampling_centers = [None] * n_iterations
    elif len(sampling_centers) != n_iterations:
        raise ValueError(
            f"sampling_centers length {len(sampling_centers)} "
            f"doesn't match simulation length {n_iterations}"
        )
    
    if initial_adjacency is None:
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        idx = np.arange(n_nodes)
        adjacency[idx, (idx + 1) % n_nodes] = 1
        adjacency[(idx + 1) % n_nodes, idx] = 1
    else:
        adjacency = initial_adjacency.copy()
    
    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = adjacency
    
    client = get_or_create_dask_client(dask_config)
    
    with tqdm(total=n_iterations - 1, desc="Simulating network evolution") as pbar:
        for t in range(1, n_iterations):
            alpha_t = get_param_value(alpha, t)
            beta_t = get_param_value(beta, t)
            noise_t = get_param_value(noise, t) if isinstance(noise, (np.ndarray, da.Array)) else 0
            penalty_t = get_param_value(connectivity_penalty, t)
            rate_t = get_param_value(sampling_rate, t)
            
            active_nodes = sample_nodes_spatially(
                n_nodes, coordinates, sampling_centers[t], 
                rate_t, sigma
            )
            
            futures = [
                client.submit(
                    process_node_updates,
                    i, adjacency.copy(), coordinates, distance_fn,
                    alpha_t, beta_t, noise_t, penalty_t
                ) for i in active_nodes
            ]
            
            results = client.gather(futures)
            
            for result in results:
                if result['changes']:
                    i = result['node']
                    adjacency[i, :] = result['new_connections']
                    adjacency[:, i] = result['new_connections']  
            
            history[:, :, t] = adjacency
            pbar.update(1)
    
    return history

def process_node_updates(
    node: int,
    adjacency: FloatArray,
    coordinates: FloatArray,
    distance_fn: DistanceMetric,
    alpha_t: float,
    beta_t: float,
    noise_t: float,
    penalty_t: float
) -> Dict[str, Any]:
    """Process all possible edge updates for a single node.
    
    For each possible connection, tests if adding/removing it improves the node's payoff.
    Returns the best configuration found.
    """
    n_nodes = len(adjacency)
    best_payoff = compute_node_payoff(
        node, adjacency, coordinates, distance_fn,
        alpha_t, beta_t, noise_t, penalty_t
    )
    best_connections = adjacency[node, :].copy()
    made_changes = False
    
    # Try flipping each possible connection
    for other in range(n_nodes):
        if other == node:
            continue
            
        # Test flipping this connection
        test_connections = best_connections.copy()
        test_connections[other] = 1 - test_connections[other]
        
        # Create test adjacency matrix
        adj_test = adjacency.copy()
        adj_test[node, :] = test_connections
        adj_test[:, node] = test_connections  # Maintain symmetry
        
        # Compute new payoff
        new_payoff = compute_node_payoff(
            node, adj_test, coordinates, distance_fn,
            alpha_t, beta_t, noise_t, penalty_t
        )
        
        # Update if better
        if new_payoff > best_payoff:
            best_payoff = new_payoff
            best_connections = test_connections
            made_changes = True
    
    return {
        'node': node,
        'new_connections': best_connections,
        'changes': made_changes
    }
