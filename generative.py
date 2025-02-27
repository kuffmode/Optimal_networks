
from typing import Callable, Union, Literal, TypeAlias, Optional
import numpy as np
import numba
import numpy.typing as npt
from numba.core.errors import TypingError
from numba import njit, jit, NumbaError
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
numba.config.DISABLE_JIT_WARNINGS = 1

def jit_safe(nopython=True, **jit_kwargs):
    """
    A safe JIT wrapper that falls back to normal Python if
    Numba cannot compile the function in nopython mode.

    Parameters
    ----------
    nopython : bool
        Whether to try nopython mode first.
    jit_kwargs : dict
        Additional arguments passed to @jit.
    """
    def decorator(func):
        if not nopython:
            return func  # Skip JIT entirely if nopython=False
        
        try:
            # Try compiling in nopython mode
            jitted = jit(nopython=True, **jit_kwargs)(func)
            # Test compilation with dummy input if possible
            # (you can define test cases if needed)
            return jitted
        except (TypingError, NumbaError) as e:
            # Fallback to non-JIT version
            print(f"Numba could not compile {func.__name__}: {e}")
            return func  # Return the original function if JIT fails
    return decorator

@njit
def _diag_indices(n):
    """
    Returns the indices of the diagonal elements of an n x n matrix.
    """
    rows = np.arange(n)
    cols = np.arange(n)
    return rows, cols

@njit
def _set_diagonal(matrix:np.ndarray, value:float=0.0):
    n = matrix.shape[0]
    rows, cols = _diag_indices(n)
    for i in range(n):
        matrix[rows[i], cols[i]] = value
    return matrix

@jit_safe()
def process_matrix(matrix):
    # Replace np.nan_to_num with a manual implementation
    n, m = matrix.shape
    result = np.empty_like(matrix)
    for i in range(n):
        for j in range(m):
            val = matrix[i, j]
            if np.isnan(val):
                result[i, j] = 0
            elif val == np.inf:
                result[i, j] = 0
            elif val == -np.inf:
                result[i, j] = 0
            else:
                result[i, j] = val
    return result

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
    allow_zero: tuple[bool, ...],
    allow_none: Optional[tuple[bool, ...]] = None
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
    none_ok = allow_none if allow_none is not None else (False,) * len(trajectories)
    for traj, name, float_ok, zero_ok, none_ok in zip(
        trajectories, names, allow_float, allow_zero, none_ok
    ):
        if none_ok and traj is None:
            continue
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

@jit_safe()
def propagation_distance(adjacency_matrix, coordinates=None, alpha=0.8, eps=1e-10):
    """
    Computes the propagation distance matrix using:
        -log((I - α*A)^{-1} * (I - α*A)^{-1}.T)
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix (must be invertible for I - α*A).
    coordinates : np.ndarray
        Not used in computation, kept for signature consistency.
    alpha : float, optional
        Scaling factor for the adjacency matrix (default: 0.5).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The elementwise -log of the propagation matrix.
    """
    N = adjacency_matrix.shape[0]
    A = adjacency_matrix.astype(np.float64)
    
    # Normalize adjacency matrix
    spectral_radius = np.max(np.abs(np.linalg.eigvalsh(A)))
    if spectral_radius > eps:
        A /= spectral_radius
    
    # Compute M = I - α*A
    I = np.eye(N, dtype=np.float64)
    M = I - alpha * A
    
    # Compute inverse and propagation matrix
    M_inv = np.linalg.inv(M)
    prop_matrix = M_inv @ M_inv.T
    
    # Set diagonal to eps and ensure positivity
    prop_matrix = _set_diagonal(prop_matrix, 0.)
    
    # Manual element-wise maximum with eps (numba-friendly)
    for i in range(N):
        for j in range(N):
            if i != j and prop_matrix[i,j] < eps:
                prop_matrix[i,j] = eps
    
    # Compute distance
    return process_matrix(np.abs(-np.log(prop_matrix)))

@jit_safe()
def weighted_propagation_distance(adjacency_matrix, coordinates, alpha=0.8, eps=1e-10):
    """
    Computes the propagation distance matrix using weights based on Euclidean distances:
        -log((I - α*W)^{-1} * (I - α*W)^{-1}.T)
    
    Where W is the adjacency matrix weighted by inverse Euclidean distances.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.
    alpha : float, optional
        Scaling factor for the adjacency matrix (default: 0.8).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The elementwise -log of the propagation matrix.
    """
    N = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    W = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight (same as resistance distance)
                W[i, j] = 1.0 / dist
    
    # Normalize weighted adjacency matrix by spectral radius
    # This is a simple approximation to avoid computing eigenvalues in numba
    row_sums = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            row_sums[i] += W[i, j]
    
    max_sum = 0.0
    for i in range(N):
        if row_sums[i] > max_sum:
            max_sum = row_sums[i]
    
    if max_sum > eps:
        for i in range(N):
            for j in range(N):
                W[i, j] /= max_sum
    
    # Compute M = I - α*W
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 1.0 - alpha * W[i, j]
            else:
                M[i, j] = -alpha * W[i, j]
    
    # Compute inverse of M
    M_inv = np.linalg.inv(M)
    
    # Compute propagation matrix M_inv * M_inv.T
    prop_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                prop_matrix[i, j] += M_inv[i, k] * M_inv[j, k]
    
    # Set diagonal to eps and ensure positivity
    for i in range(N):
        prop_matrix[i, i] = 0.0
        for j in range(N):
            if i != j and prop_matrix[i, j] < eps:
                prop_matrix[i, j] = eps
    
    # Compute distance as -log(prop_matrix)
    result = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                result[i, j] = 0.0
            else:
                result[i, j] = -np.log(prop_matrix[i, j])
    
    return process_matrix(result)

@jit_safe()
def normalized_propagation_distance(adjacency_matrix, coordinates, alpha=0.1, eps=1e-10):
    """
    Computes the propagation distance matrix and then normalizes
    the resulting distance matrix based on node strengths.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.
    alpha : float, optional
        Scaling factor for the adjacency matrix (default: 0.8).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The normalized propagation distance matrix.
    """
    N = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    W = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight (same as resistance distance)
                W[i, j] = 1.0 / dist
    
    # Normalize weighted adjacency matrix by spectral radius
    # This is a simple approximation to avoid computing eigenvalues in numba
    row_sums = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            row_sums[i] += W[i, j]
    
    max_sum = 0.0
    for i in range(N):
        if row_sums[i] > max_sum:
            max_sum = row_sums[i]
    
    if max_sum > eps:
        for i in range(N):
            for j in range(N):
                W[i, j] /= max_sum
    
    # Compute M = I - α*W
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 1.0 - alpha * W[i, j]
            else:
                M[i, j] = -alpha * W[i, j]
    
    # Compute inverse of M
    M_inv = np.linalg.inv(M)
    
    # Compute propagation matrix M_inv * M_inv.T
    prop_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                prop_matrix[i, j] += M_inv[i, k] * M_inv[j, k]
    
    # Set diagonal to eps and ensure positivity
    for i in range(N):
        prop_matrix[i, i] = 0.0
        for j in range(N):
            if i != j and prop_matrix[i, j] < eps:
                prop_matrix[i, j] = eps
    
    # Compute propagation distance as -log(prop_matrix)
    prop_distance = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                prop_distance[i, j] = 0.0
            else:
                prop_distance[i, j] = -np.log(prop_matrix[i, j])
    
    # Process the matrix to handle inf and nan values
    distance_matrix = process_matrix(prop_distance)
    
    # Calculate node strengths for normalization (using the original weighted adjacency)
    strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            strength[i] += W[i, j]
    
    # Compute normalized strength (power -0.5)
    normalized_strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if strength[i] > 1e-12:  # Avoid division by zero
            normalized_strength[i] = 1.0 / np.sqrt(strength[i])
        else:
            normalized_strength[i] = 0.0
    
    # Apply normalization to distance matrix
    normalized_distance = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            normalized_distance[i, j] = normalized_strength[i] * distance_matrix[i, j] * normalized_strength[j]
            
    return normalized_distance

@jit_safe()
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

@jit_safe()
def normalized_resistance_distance(adjacency_matrix, coordinates):
    """
    Compute resistance distances between all pairs of nodes and normalize
    the resulting distance matrix based on node strengths.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Binary or weighted adjacency matrix (n_nodes, n_nodes)
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting
            
    Returns
    -------
    np.ndarray
        Matrix of normalized resistance distances (n_nodes, n_nodes)
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight
                W[i, j] = 1.0 / dist
    
    # Compute weighted Laplacian
    diag = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            diag[i] += W[i, j]
    
    laplacian = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        laplacian[i, i] = diag[i]
        for j in range(n_nodes):
            laplacian[i, j] -= W[i, j]
    
    # Compute pseudoinverse using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    
    # Filter out numerical zeros
    L_plus = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for k in range(n_nodes):
        if eigvals[k] > 1e-10:  # Only use non-zero eigenvalues
            for i in range(n_nodes):
                for j in range(n_nodes):
                    L_plus[i, j] += (eigvecs[i, k] * eigvecs[j, k]) / eigvals[k]
    
    # Compute resistance distances
    resistance = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i, n_nodes):  # Compute upper triangle only
            if i == j:
                resistance[i, j] = 0.0
            else:
                r = L_plus[i, i] + L_plus[j, j] - 2.0 * L_plus[i, j]
                resistance[i, j] = r
                resistance[j, i] = r  # Fill lower triangle (symmetric)
    
    # Now normalize the resistance distance matrix using node strengths
    # Calculate node strengths (sum of weights per row)
    strength = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            strength[i] += W[i, j]
    
    # Compute normalized strength (power -0.5)
    normalized_strength = np.zeros(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        if strength[i] > 1e-12:  # Avoid division by zero
            normalized_strength[i] = 1.0 / np.sqrt(strength[i])
        else:
            normalized_strength[i] = 0.0
    
    # Apply normalization to resistance matrix
    normalized_resistance = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            normalized_resistance[i, j] = normalized_strength[i] * resistance[i, j] * normalized_strength[j]
            
    return normalized_resistance

@jit_safe()
def heat_kernel_distance(adjacency_matrix, coordinates=None, t=0.5, eps=1e-10,normalize=False):
    """
    Computes the heat kernel distance matrix at diffusion time t.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix.
    coordinates : np.ndarray, optional
        Not used in computation, kept for signature consistency.
    t : float, optional
        Diffusion time parameter controlling the balance between local and 
        global structure (default: 1.0).
        - Small t values focus on local structure (similar to shortest path)
        - Large t values focus on global structure (approaching resistance distance)
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The heat kernel distance matrix.
    """
    N = adjacency_matrix.shape[0]
    A = adjacency_matrix.astype(np.float64)
    
    # Compute degree matrix and Laplacian
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        D[i, i] = np.sum(A[i])
    
    # Compute Laplacian: L = D - A
    L = D - A
    
    # Compute eigenvalues and eigenvectors of Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Compute heat kernel matrix: H_t = exp(-t*L)
    # Using eigendecomposition: H_t = U * exp(-t*Λ) * U^T
    H_t = np.zeros((N, N), dtype=np.float64)
    
    # Compute exp(-t*λ) for each eigenvalue
    exp_vals = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if eigvals[i] > eps:  # Exclude zero eigenvalues (connected components)
            exp_vals[i] = np.exp(-t * eigvals[i])
        else:
            exp_vals[i] = 1.0  # For numerical stability with zero eigenvalues
    
    # Compute H_t using the eigendecomposition
    for i in range(N):
        for j in range(N):
            val = 0.0
            for k in range(N):
                val += exp_vals[k] * eigvecs[i, k] * eigvecs[j, k]
            H_t[i, j] = val
    
    # Compute heat kernel distance: sqrt(H_t(i,i) + H_t(j,j) - 2*H_t(i,j))
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_ij = H_t[i, i] + H_t[j, j] - 2.0 * H_t[i, j]
                # Ensure non-negative distance due to numerical errors
                if dist_ij > 0:
                    dist_matrix[i, j] = np.sqrt(dist_ij)
                else:
                    dist_matrix[i, j] = 0.0
    if normalize:
        strength = dist_matrix.sum(1)
        normalized_strength: np.ndarray = np.power(strength, -0.5)
        diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
        normalized_dist_matrix: np.ndarray = (
            diagonalized_normalized_strength
            @ dist_matrix
            @ diagonalized_normalized_strength
        )
        return normalized_dist_matrix
    else:
        return dist_matrix

@jit_safe()
def weighted_heat_kernel_distance(adjacency_matrix, coordinates, t=1.0, eps=1e-10):
    """
    Computes the heat kernel distance matrix at diffusion time t using 
    weights based on Euclidean distances.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.
    t : float, optional
        Diffusion time parameter controlling the balance between local and 
        global structure (default: 1.0).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The heat kernel distance matrix.
    """
    N = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    A = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight (same as resistance distance)
                A[i, j] = 1.0 / dist
    
    # Compute weighted degree matrix
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += A[i, j]
        D[i, i] = row_sum
    
    # Compute weighted Laplacian: L = D - A
    L = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                L[i, j] = D[i, j]
            else:
                L[i, j] = -A[i, j]
    
    # Compute eigenvalues and eigenvectors of Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Compute heat kernel matrix: H_t = exp(-t*L)
    # Using eigendecomposition: H_t = U * exp(-t*Λ) * U^T
    H_t = np.zeros((N, N), dtype=np.float64)
    
    # Manual matrix multiplication for numba compatibility
    for i in range(N):
        for j in range(N):
            val = 0.0
            for k in range(N):
                # Only use non-zero eigenvalues
                if eigvals[k] > eps:
                    val += np.exp(-t * eigvals[k]) * eigvecs[i, k] * eigvecs[j, k]
                else:
                    # Handle zero eigenvalue differently (connected components)
                    val += eigvecs[i, k] * eigvecs[j, k]
            H_t[i, j] = val
    
    # Compute heat kernel distance: sqrt(H_t(i,i) + H_t(j,j) - 2*H_t(i,j))
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_ij = H_t[i, i] + H_t[j, j] - 2.0 * H_t[i, j]
                # Ensure non-negative distance due to numerical errors
                if dist_ij > eps:
                    dist_matrix[i, j] = np.sqrt(dist_ij)
                else:
                    dist_matrix[i, j] = 0.0
    
    return dist_matrix

@jit_safe()
def normalized_heat_kernel_distance(adjacency_matrix, coordinates, t=1.0, eps=1e-10):
    """
    Computes the heat kernel distance matrix at diffusion time t and then
    normalizes the resulting distance matrix based on node strengths.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.
    t : float, optional
        Diffusion time parameter controlling the balance between local and 
        global structure (default: 1.0).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The normalized heat kernel distance matrix.
    """
    N = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    W = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight
                W[i, j] = 1.0 / dist
    
    # Compute weighted degree matrix
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += W[i, j]
        D[i, i] = row_sum
    
    # Compute weighted Laplacian: L = D - W
    L = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                L[i, j] = D[i, j]
            else:
                L[i, j] = -W[i, j]
    
    # Compute eigenvalues and eigenvectors of Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Compute heat kernel matrix: H_t = exp(-t*L)
    # Using eigendecomposition: H_t = U * exp(-t*Λ) * U^T
    H_t = np.zeros((N, N), dtype=np.float64)
    
    # Manual matrix multiplication for numba compatibility
    for i in range(N):
        for j in range(N):
            val = 0.0
            for k in range(N):
                # Only use non-zero eigenvalues
                if eigvals[k] > eps:
                    val += np.exp(-t * eigvals[k]) * eigvecs[i, k] * eigvecs[j, k]
                else:
                    # Handle zero eigenvalue differently (connected components)
                    val += eigvecs[i, k] * eigvecs[j, k]
            H_t[i, j] = val
    
    # Compute heat kernel distance: sqrt(H_t(i,i) + H_t(j,j) - 2*H_t(i,j))
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_ij = H_t[i, i] + H_t[j, j] - 2.0 * H_t[i, j]
                # Ensure non-negative distance due to numerical errors
                if dist_ij > eps:
                    dist_matrix[i, j] = np.sqrt(dist_ij)
                else:
                    dist_matrix[i, j] = 0.0
    
    # Calculate node strengths for normalization (using the weighted adjacency)
    strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            strength[i] += W[i, j]
    
    # Compute normalized strength (power -0.5)
    normalized_strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if strength[i] > 1e-12:  # Avoid division by zero
            normalized_strength[i] = 1.0 / np.sqrt(strength[i])
        else:
            normalized_strength[i] = 0.0
    
    # Apply normalization to distance matrix
    normalized_distance = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            normalized_distance[i, j] = normalized_strength[i] * dist_matrix[i, j] * normalized_strength[j]
            
    return normalized_distance

@jit_safe()
def shortest_path_distance(adjacency_matrix,coordinates = None):
    """
    Computes shortest-path distances between all pairs of nodes
    using the Floyd-Warshall algorithm.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix where entry (i, j) represents
        the weight of the edge from node i to j. Use np.inf for no direct edge.

    Returns
    -------
    dist_matrix : np.ndarray
        A square matrix where entry (i, j) represents the shortest path
        distance from node i to j.
    """
    N = adjacency_matrix.shape[0]
    
    # Create distance matrix as a copy of the adjacency matrix
    dist_matrix = adjacency_matrix.astype(np.float64)
    
    # Convert zero entries (non-diagonal) to np.inf (no connection)
    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] == 0:
                dist_matrix[i, j] = np.inf
    
    # Ensure diagonal is 0 (distance from a node to itself)
    for i in range(N):
        dist_matrix[i, i] = 0.0

    # Floyd-Warshall algorithm
    for k in range(N):  # Intermediate node
        for i in range(N):  # Source node
            for j in range(N):  # Destination node
                # Relax the distance via intermediate node k
                dist_matrix[i, j] = min(
                    dist_matrix[i, j],
                    dist_matrix[i, k] + dist_matrix[k, j]
                )

    return dist_matrix

@jit_safe()
def weighted_shortest_path_distance(adjacency_matrix, coordinates):
    """
    Computes shortest-path distances between all pairs of nodes using the
    Floyd-Warshall algorithm with weights based on Euclidean distances.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix where 1 indicates a connection.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.

    Returns
    -------
    dist_matrix : np.ndarray
        A square matrix where entry (i, j) represents the weighted shortest
        path distance from node i to j.
    """
    N = adjacency_matrix.shape[0]
    
    # Initialize distance matrix with infinity for non-connected nodes
    dist_matrix = np.full((N, N), np.inf, dtype=np.float64)
    
    # Set diagonal to 0
    for i in range(N):
        dist_matrix[i, i] = 0.0
    
    # Calculate Euclidean distance weights for connected edges
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance between nodes
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist)
                dist_matrix[i, j] = dist
    
    # Floyd-Warshall algorithm
    for k in range(N):  # Intermediate node
        for i in range(N):  # Source node
            for j in range(N):  # Destination node
                # Relax the distance via intermediate node k
                if dist_matrix[i, k] != np.inf and dist_matrix[k, j] != np.inf:
                    new_dist = dist_matrix[i, k] + dist_matrix[k, j]
                    if new_dist < dist_matrix[i, j]:
                        dist_matrix[i, j] = new_dist
    
    return dist_matrix

@jit_safe()
def controllability_distance(adjacency_matrix, coordinates, energy_parameter=1.0, eps=1e-10):
    """
    Computes a distance matrix based on network controllability principles.
    This measure quantifies the minimum energy required to drive the system
    from one node to another.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square binary adjacency matrix.
    coordinates : np.ndarray
        Node coordinates (n_nodes, n_dimensions) used for edge weighting.
    energy_parameter : float, optional
        Parameter controlling the energy scaling (default: 1.0).
        Higher values increase the cost of control over long distances.
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The controllability-based distance matrix.
    """
    N = adjacency_matrix.shape[0]
    
    # Create weighted adjacency matrix using inverse Euclidean distances
    W = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if adjacency_matrix[i, j] > 0:
                # Calculate Euclidean distance
                dist = 0.0
                for d in range(coordinates.shape[1]):
                    dist += (coordinates[i, d] - coordinates[j, d])**2
                dist = np.sqrt(dist) + 1e-12  # Avoid division by zero
                
                # Use inverse distance as weight
                W[i, j] = 1.0 / dist
    
    # Compute controllability Gramian approximation
    # In linear systems, the controllability Gramian can be approximated using:
    # W_c = ∑_{k=0}^{∞} A^k*B*B^T*(A^T)^k
    # We'll use a simplified approach focusing on network structure
    
    # First, normalize the adjacency matrix by spectral radius
    row_sums = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            row_sums[i] += W[i, j]
    
    max_sum = 0.0
    for i in range(N):
        if row_sums[i] > max_sum:
            max_sum = row_sums[i]
    
    if max_sum > eps:
        for i in range(N):
            for j in range(N):
                W[i, j] /= max_sum
    
    # Compute a simplified controllability measure
    # We'll use the resolvent matrix (I - αW)^(-1) as an approximation
    alpha = 0.95  # Set close to 1 to capture long-range effects
    
    # Compute resolvent matrix
    I_minus_alphaW = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                I_minus_alphaW[i, j] = 1.0 - alpha * W[i, j]
            else:
                I_minus_alphaW[i, j] = -alpha * W[i, j]
                
    # Compute inverse
    resolvent = np.linalg.inv(I_minus_alphaW)
    
    # Compute controllability matrix
    # We use the energy interpretation: nodes requiring less energy to control
    # are closer in the controllability sense
    control_energy = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                control_energy[i, j] = 0.0
            else:
                # Energy is related to the inverse of the square of connection strength
                energy = 0.0
                for k in range(N):
                    energy += (resolvent[i, k] - resolvent[j, k])**2
                
                # Apply energy parameter to scale the distance
                control_energy[i, j] = energy_parameter * np.sqrt(energy)
    
    # Ensure the distance matrix is symmetric (average if needed)
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = (control_energy[i, j] + control_energy[j, i]) / 2.0
    
    # Calculate node strengths for normalization (using the weighted adjacency)
    strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            strength[i] += W[i, j]
    
    # Compute normalized strength (power -0.5)
    normalized_strength = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if strength[i] > 1e-12:  # Avoid division by zero
            normalized_strength[i] = 1.0 / np.sqrt(strength[i])
        else:
            normalized_strength[i] = 0.0
    
    # Apply normalization to distance matrix
    normalized_distance = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            normalized_distance[i, j] = normalized_strength[i] * dist_matrix[i, j] * normalized_strength[j]
            
    return normalized_distance

@jit_safe()
def search_information(W, coordinates = None,symmetric=True):
    """
    Calculate search information for a memoryless random walker.

    Parameters
    ----------
    W : (N, N) ndarray
        Weighted/unweighted, directed/undirected connection weight matrix.
    coordinates : (N, 2) ndarray
        Coordinates of nodes (n_nodes, n_dimensions).

    Returns
    -------
    SI : (N, N) ndarray
        Pairwise search information matrix. The diagonal is set to 0.
        Edges without a valid shortest path are set to np.inf.
    """
    N = W.shape[0]

    # Normalize weights to transition probabilities
    T = W / np.sum(W, axis=1)[:, None]

    # Precompute shortest path distances using Floyd-Warshall algorithm
    dist_matrix = W.astype(np.float64)
    p_mat = np.zeros((N, N), dtype=np.int32) - 1  # Predecessor matrix

    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] == 0:
                dist_matrix[i, j] = np.inf
            else:
                p_mat[i, j] = i

    for k in range(N):  # Intermediate node
        for i in range(N):  # Source node
            for j in range(N):  # Destination node
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
                    p_mat[i, j] = p_mat[k, j]

    # Initialize search information matrix
    SI = np.full((N, N), np.inf)

    # Compute search information
    for i in range(N):
        for j in range(N):
            if i == j:
                SI[i, j] = 0.0
                continue

            # Retrieve shortest path from predecessor matrix
            path = []
            current = j
            while current != -1 and current != i:
                path.append(current)
                current = p_mat[i, current]
            if current == i:
                path.append(i)
                path.reverse()
            else:
                continue  # No valid path

            # Compute search information along the shortest path
            product = 1.0
            for k in range(len(path) - 1):
                product *= T[path[k], path[k + 1]]
            SI[i, j] = -np.log2(product)

    result = np.full((N, N), np.inf)
    if symmetric:
        for i in range(N):
            result[i,i] = 0.0  # Keep diagonal zero
            for j in range(i+1, N):
                # Take minimum of both directions
                min_val = min(SI[i,j], SI[j,i])
                result[i,j] = min_val
                result[j,i] = min_val
        return result
    else:
        return SI

@jit_safe()
def topological_distance(adj_matrix, coordinates = None):
    """
    Compute pairwise cosine similarity between nodes based on their edge patterns.
    
    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix (N x N)
        
    Returns:
        np.ndarray: Matching index matrix (N x N)
    """
    N = adj_matrix.shape[0]
    matching_matrix = np.zeros((N, N))
    
    for i in range(N):
        edges_i = adj_matrix[i]
        norm_i = np.sqrt(np.sum(edges_i * edges_i))
        
        for j in range(i, N):  # Symmetric matrix, compute upper triangle
            edges_j = adj_matrix[j]
            norm_j = np.sqrt(np.sum(edges_j * edges_j))
            
            # Handle zero-degree nodes
            if norm_i == 0 or norm_j == 0:
                matching_matrix[i, j] = matching_matrix[j, i] = 0
                continue
                
            # Compute cosine similarity
            dot_product = np.sum(edges_i * edges_j)
            similarity = dot_product / (norm_i * norm_j)
            
            # Fill both triangles due to symmetry
            matching_matrix[i, j] = matching_matrix[j, i] = similarity
            
    return 1-matching_matrix

@jit_safe()
def matching_distance(adj_matrix,coordinates = None):
    """
    Compute pairwise matching index between nodes.
    Matching index = 2 * (shared connections) / (total unshared connections)
    
    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix (N x N)
        
    Returns:
        np.ndarray: Matching index matrix (N x N)
    """
    N = adj_matrix.shape[0]
    matching_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):  # Compute upper triangle only
            # Get neighbors excluding i and j
            edges_i = adj_matrix[i].copy()
            edges_j = adj_matrix[j].copy()
            
            # Remove mutual connection and self-connections
            edges_i[i] = edges_i[j] = 0
            edges_j[i] = edges_j[j] = 0
            
            # Count shared and total connections (numba-friendly)
            shared = np.sum(np.logical_and(edges_i, edges_j))
            total = np.sum(np.logical_or(edges_i, edges_j))
            
            # Compute matching index
            if total > 0:
                similarity = shared / total
            else:
                similarity = 0.0
                
            # Fill both triangles due to symmetry
            matching_matrix[i, j] = matching_matrix[j, i] = similarity
            
    return 1-matching_matrix

@jit_safe()
def combined_objectives(adj_matrix, coordinates=None,weight=2):
    sp = shortest_path_distance(adj_matrix, coordinates)
    rd = resistance_distance(adj_matrix, coordinates)
    n = len(adj_matrix)
    result = np.zeros((n, n), dtype=np.float64)
    
    # Manual mean calculation
    for i in range(n):
        for j in range(n):
            result[i,j] = (rd[i,j] + weight*sp[i,j]) / 2.0
            
    return result
    
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

@njit
def sample_nodes_with_centers(
    n_nodes: int,
    centers: FloatArray,
    coordinates: FloatArray,
    width: float,
    t: int
) -> tuple[int, int]:
    """
    Sample nodes using gaussian distribution around given centers.
    
    Args:
        n_nodes: Number of nodes in network
        centers: Center nodes for sampling (n_centers, n_iterations) or (1, n_iterations)
        coordinates: Node coordinates (n_nodes, n_dimensions)
        width: Width of gaussian distribution
        t: Current iteration
        
    Returns:
        Tuple of sampled node indices (i, j)
    """
    # Get current centers
    current_centers = centers[:, t]
    n_centers = len(current_centers)
    
    # Randomly select a center
    center_idx = np.random.randint(0, n_centers)
    center = int(current_centers[center_idx])
    
    # Compute distances from center to all nodes
    center_coords = coordinates[center]
    distances = np.sqrt(np.sum((coordinates - center_coords)**2, axis=1))
    
    # Compute gaussian probabilities
    probs = np.exp(-distances**2 / (2 * width**2))
    probs /= np.sum(probs)
    
    # Sample two nodes based on probabilities
    i = np.random.choice(n_nodes, p=probs)
    j = np.random.choice(n_nodes, p=probs)
    
    return i, j

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
    batch_size: Trajectory = 32,
    sampling_centers: Optional[FloatArray] = None,
    sampling_width: Optional[float] = None,
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
        alpha, beta, noise, connectivity_penalty, batch_size,
        names=('alpha', 'beta', 'noise', 'connectivity_penalty', 'batch_size'),
        allow_float=(True, True, False, True, True),
        allow_zero=(True, True, True, True, False)
    )
    
    if sampling_centers is not None:
        if sampling_width is None:
            raise ValueError("sampling_width must be provided when using sampling_centers")
        if sampling_centers.ndim != 2:
            raise ValueError("sampling_centers must be 2D array (n_centers, n_iterations)")
        if sampling_centers.shape[1] != n_iterations:
            raise ValueError(
                f"sampling_centers length {sampling_centers.shape[1]} doesn't match "
                f"simulation length {n_iterations}"
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
            
            # Get current batch size
            batch_size_t = get_param_value(batch_size, t)
            
            # Process edge flips in parallel batches
            def process_flip(_):
                if sampling_centers is not None:
                    i, j = sample_nodes_with_centers(
                        n_nodes, sampling_centers, coordinates, sampling_width, t
                    )
                else:
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
            n_flips = int(batch_size_t)
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

def optimize_weights(
    binary_adjacency: FloatArray,
    coordinates: FloatArray,
    n_iterations: int,
    distance_fn: DistanceMetric,
    alpha: Trajectory,
    beta: Trajectory,
    total_weight: float = 1.0,
    learning_rate: float = 0.01,
    n_jobs: int = -1,
    random_seed: Optional[int] = None
) -> FloatArray:
    """
    Optimize connection weights given a binary network structure.
    
    Args:
        binary_adjacency: Binary adjacency matrix defining network structure
        coordinates: Node coordinates (n_nodes, n_dimensions)
        n_iterations: Number of optimization steps
        distance_fn: Function computing distance metric (e.g., resistance_distance)
        alpha: Weight of distance term
        beta: Weight of wiring cost
        total_weight: Total connection weight budget per node
        learning_rate: Step size for weight updates
        n_jobs: Number of parallel jobs
        batch_size: Number of nodes to update in parallel
        random_seed: Random seed for reproducibility
        
    Returns:
        Optimized weighted adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_nodes = len(coordinates)
    
    # Initialize weights uniformly across existing connections
    weighted_adj = binary_adjacency.copy()
    for i in range(n_nodes):
        connections = binary_adjacency[i] > 0
        n_connections = np.sum(connections)
        if n_connections > 0:
            weighted_adj[i, connections] = total_weight / n_connections
    
    # Pre-compute euclidean distances for wiring cost
    euclidean_dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        euclidean_dist[i] = np.sqrt(np.sum((coordinates[i] - coordinates)**2, axis=1))
    
    def optimize_node_weights(node_idx):
        """Optimize weights for a single node while maintaining budget."""
        if np.sum(binary_adjacency[node_idx]) == 0:
            return None
            
        # Get current state
        current_weights = weighted_adj[node_idx].copy()
        current = compute_node_payoff(
            node_idx, weighted_adj, coordinates, distance_fn,
            alpha_t, beta_t, 0, 0
        )
        
        # Try adjusting each existing connection
        connections = np.where(binary_adjacency[node_idx] > 0)[0]
        best_weights = current_weights.copy()
        best_payoff = current
        
        for i in connections:
            for j in connections:
                if i != j:
                    # Try moving some weight from j to i
                    test_weights = current_weights.copy()
                    delta = min(learning_rate, test_weights[j])
                    test_weights[j] -= delta
                    test_weights[i] += delta
                    
                    # Apply changes symmetrically
                    test_adj = weighted_adj.copy()
                    test_adj[node_idx] = test_weights
                    test_adj[:, node_idx] = test_weights
                    
                    # Evaluate new payoff
                    new_payoff = compute_node_payoff(
                        node_idx, test_adj, coordinates, distance_fn,
                        alpha_t, beta_t, 0, 0
                    )
                    
                    if new_payoff > best_payoff:
                        best_payoff = new_payoff
                        best_weights = test_weights.copy()
        
        if best_payoff > current:
            return {
                'node': node_idx,
                'weights': best_weights
            }
        return None
    
    # Optimization loop
    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = weighted_adj
    
    with tqdm(total=n_iterations-1, desc="Optimizing weights") as pbar:
        for t in range(1, n_iterations):
            # Get current parameters
            alpha_t = get_param_value(alpha, t)
            beta_t = get_param_value(beta, t)
            
            # Process nodes in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(optimize_node_weights)(i) 
                for i in range(n_nodes)
            )
            
            # Apply accepted weight changes
            for result in results:
                if result is not None:
                    node = result['node']
                    weighted_adj[node] = result['weights']
                    weighted_adj[:, node] = result['weights']
                    
            history[:, :, t] = weighted_adj
            pbar.update(1)
            
    return history