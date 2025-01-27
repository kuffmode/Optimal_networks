from scipy.stats import ks_2samp
from numba import njit
import warnings
from functools import lru_cache
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Callable, TypeVar, Any
import numpy.typing as npt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from generative import simulate_network_evolution, resistance_distance

def evaluator(synthetic, empirical, euclidean_distance):
    degrees_synthetic = np.sum(synthetic, axis=0)
    degrees_empirical = np.sum(empirical, axis=0)
    ks_degree = ks_2samp(degrees_synthetic, degrees_empirical)
    
    clustering_synthetic = nx.clustering(nx.from_numpy_array(synthetic))
    clustering_empirical = nx.clustering(nx.from_numpy_array(empirical))
    ks_clustering = ks_2samp(list(clustering_synthetic.values()), list(clustering_empirical.values()))
    
    betweenness_synthetic = nx.betweenness_centrality(nx.from_numpy_array(synthetic))
    betweenness_empirical = nx.betweenness_centrality(nx.from_numpy_array(empirical))
    ks_betweenness = ks_2samp(list(betweenness_synthetic.values()), list(betweenness_empirical.values()))
    
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    distance_empirical = euclidean_distance[np.triu(empirical, 1) > 0]
    ks_distance = ks_2samp(distance_synthetic, distance_empirical)
    
    results = {"energy":np.max([ks_degree[0], 
                                ks_clustering[0], 
                                ks_betweenness[0], 
                                ks_distance[0]]),
               "ks_degrees": ks_degree[0],
               "ks_clustering": ks_clustering[0],
               "ks_betweenness": ks_betweenness[0],
               "ks_distance": ks_distance[0]}
    
    return results

@njit
def fast_degrees(adj: np.ndarray) -> np.ndarray:
    """Compute degrees efficiently with numba."""
    return np.sum(adj, axis=0)

@njit
def fast_clustering(adj: np.ndarray) -> np.ndarray:
    """
    Compute clustering coefficients efficiently with numba.
    
    For each node:
    C_i = (number of triangles connected to node i) / 
          (number of pairs of neighbors)
    """
    n = len(adj)
    clustering = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:  # Need at least 2 neighbors for triangles
            continue
        
        # Count triangles
        triangles = 0
        for j in range(k):
            for l in range(j + 1, k):
                if adj[neighbors[j], neighbors[l]] > 0:
                    triangles += 1
        
        # Compute coefficient
        possible_triangles = k * (k - 1) / 2
        clustering[i] = triangles / possible_triangles if possible_triangles > 0 else 0
    
    return clustering

@njit
def fast_betweenness(adj: np.ndarray) -> np.ndarray:
    """
    Approximate betweenness centrality using shortest paths.
    Uses a simplified algorithm suitable for numba.
    """
    n = len(adj)
    betweenness = np.zeros(n)
    
    for s in range(n):
        # BFS from source s
        dist = np.full(n, np.inf)
        sigma = np.zeros(n)  # number of shortest paths
        dist[s] = 0
        sigma[s] = 1
        
        # Queue for BFS
        queue = np.zeros(n, dtype=np.int64)
        queue[0] = s
        q_start, q_end = 0, 1
        
        # Keep track of visited nodes in order
        visited = np.zeros(n, dtype=np.int64)
        n_visited = 0
        
        while q_start < q_end:
            v = queue[q_start]
            q_start += 1
            visited[n_visited] = v
            n_visited += 1
            
            # Look at neighbors
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                    
                # First time seeing w?
                if dist[w] == np.inf:
                    queue[q_end] = w
                    q_end += 1
                    dist[w] = dist[v] + 1
                
                # Is this a shortest path to w?
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        
        # Dependency accumulation
        dependency = np.zeros(n)
        
        # Go through nodes in reverse order of visit
        for i in range(n_visited - 1, -1, -1):
            v = visited[i]
            if v == s:
                continue
                
            coeff = (1 + dependency[v]) / sigma[v]
            
            # Look at predecessors
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                if dist[w] == dist[v] - 1:
                    dependency[w] += sigma[w] * coeff
        
        betweenness += dependency
    
    # Normalize
    betweenness = betweenness / (2 * ((n-1) * (n-2)))
    return betweenness

def fast_evaluator(synthetic: np.ndarray, 
                  empirical: np.ndarray, 
                  euclidean_distance: np.ndarray,
                  cache_size: int = 128) -> float:
    """
    Optimized network comparison using KS tests on multiple metrics.
    
    Args:
        synthetic: Synthetic network adjacency matrix
        empirical: Empirical network adjacency matrix
        euclidean_distance: Pre-computed distance matrix
        cache_size: Size of LRU cache for metric computations
    
    Returns:
        Maximum KS statistic across all metrics
    """
    # Use cached versions for empirical network metrics
    @lru_cache(maxsize=cache_size)
    def get_empirical_metrics(emp_key):
        emp = np.array(emp_key)
        return (
            fast_degrees(emp),
            fast_clustering(emp),
            fast_betweenness(emp),
            euclidean_distance[np.triu(emp, 1) > 0]
        )
    
    # Convert empirical matrix to tuple for hashing
    emp_key = tuple(map(tuple, empirical))
    
    # Get cached empirical metrics
    (degrees_empirical,
     clustering_empirical,
     betweenness_empirical,
     distance_empirical) = get_empirical_metrics(emp_key)
    
    # Compute synthetic metrics
    degrees_synthetic = fast_degrees(synthetic)
    clustering_synthetic = fast_clustering(synthetic)
    betweenness_synthetic = fast_betweenness(synthetic)
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    
    # Compute KS statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore KS test warnings
        ks_stats = [
            ks_2samp(degrees_synthetic, degrees_empirical)[0],
            ks_2samp(clustering_synthetic, clustering_empirical)[0],
            ks_2samp(betweenness_synthetic, betweenness_empirical)[0],
            ks_2samp(distance_synthetic, distance_empirical)[0]
        ]
    results = {"energy":np.max(ks_stats),
               "ks_degrees": ks_stats[0],
               "ks_clustering": ks_stats[1],
               "ks_betweenness": ks_stats[2],
               "ks_distance": ks_stats[3]}
    return results

@njit
def density_distance(final_network, empirical):
    # Compute synthetic network density
    density_synthetic = np.sum(final_network) / (final_network.shape[0] * final_network.shape[0])
    
    # Compute empirical network density
    density_empirical = np.sum(empirical) / (empirical.shape[0] * empirical.shape[0])
    
    # Return absolute difference
    return abs(density_synthetic - density_empirical)

FloatArray = npt.NDArray[np.float64]
Parameter = TypeVar('Parameter', float, FloatArray)

@dataclass
class OptimizationResults:
    """Container for optimization results."""
    best_parameters: Dict[str, float]
    best_score: float
    all_parameters: List[Dict[str, float]]
    all_scores: List[float]
    convergence_info: Dict[str, Any]

class NetworkOptimizer:
    """
    Optimizer for network evolution parameters using Bayesian optimization.
    """
    
    def __init__(
        self,
        coordinates: FloatArray,
        empirical_network: FloatArray,
        n_iterations: int = 1000,
        evaluator: Optional[Callable] = None,
        n_parallel: int = -1,
        distance_fn: Optional[Callable] = resistance_distance,
        evaluator_kwargs: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            coordinates: Node coordinates (n_nodes, n_dimensions)
            empirical_network: Target network to match
            n_iterations: Number of iterations for each simulation
            evaluator: Custom evaluation function (defaults to density distance)
            n_parallel: Number of parallel jobs (-1 for all cores)
            distance_fn: Function to compute the "signalling distance" between nodes,
                         defaults to resistance distance
            evaluator_kwargs: Additional arguments for the evaluator
            random_seed: Random seed for reproducibility
        """
        self.coordinates = coordinates
        self.empirical_network = empirical_network
        self.n_iterations = n_iterations
        self.n_parallel = n_parallel
        self.random_seed = random_seed
        self.distance_fn = distance_fn
        self.evaluator_kwargs = evaluator_kwargs if evaluator_kwargs else {}
        # Set up evaluator
        if evaluator is None:
            self.evaluator = density_distance
        else:
            self.evaluator = evaluator
            
        # Compute euclidean distances once
        from scipy.spatial.distance import pdist, squareform
        self.euclidean_distance = squareform(pdist(coordinates))
        
    def _create_parameter_space(
        self,
        param_ranges: Dict[str, tuple]
    ) -> List[Real]:
        """Create skopt parameter space from ranges."""
        space = []
        for name, (low, high) in param_ranges.items():
            space.append(Real(low, high, name=name))
        return space
        
    def _simulate_with_params(
        self,
        params: Dict[str, float],
        batch_id: Optional[int] = None
    ) -> float:
        """Run simulation with given parameters and evaluate."""
        
        # Create parameter trajectories if needed
        alpha = params.get('alpha', np.full(self.n_iterations, 1.0))
        beta = params.get('beta', np.full(self.n_iterations, 0.1))
        noise = params.get('noise', np.zeros(self.n_iterations))
        penalty = params.get('connectivity_penalty', np.zeros(self.n_iterations))
        evaluator_kwargs = params.get('evaluator_kwargs', {})
        # Add trajectory parameters if present
        if 'beta_growth' in params:
            beta = np.linspace(0, beta, self.n_iterations)
        if 'noise_std' in params:
            noise = np.random.normal(0, params['noise_std'], self.n_iterations)
            
        # Run simulation
        history = simulate_network_evolution(
            coordinates=self.coordinates,
            n_iterations=self.n_iterations,
            distance_fn=self.distance_fn,
            alpha=alpha,
            beta=beta,
            noise=noise,
            connectivity_penalty=penalty,
            n_jobs=1,  # We're already parallelizing at a higher level
            random_seed=self.random_seed + batch_id if batch_id is not None else None
        )
        
        # Evaluate final network
        final_network = history[:, :, -1]
        score = self.evaluator(
            final_network,
            self.empirical_network,
            **evaluator_kwargs,
        )
        
        return score
        
    def optimize(
        self,
        param_ranges: Dict[str, tuple],
        n_calls: int = 50,
        n_initial_points: int = 10,
        n_parallel_samples: int = 16,
        acquisition_function: str = "gp_hedge",
        verbose: bool = True
    ) -> OptimizationResults:
        """
        Run Bayesian optimization to find best parameters.
        
        Args:
            param_ranges: Dictionary of parameter names and (min, max) ranges
            n_calls: Number of optimization iterations
            n_initial_points: Number of initial random points
            n_parallel_samples: Number of parallel evaluations per iteration
            acquisition_function: Acquisition function for GP
            verbose: Whether to show progress bar
        
        Returns:
            OptimizationResults containing best parameters and optimization history
        """
        space = self._create_parameter_space(param_ranges)

        # Create objective function with named arguments
        @use_named_args(space)
        def objective(**params):
            # Run multiple samples in parallel
            scores = Parallel(n_jobs=self.n_parallel)(
            delayed(self._simulate_with_params)(params, i)
            for i in tqdm(range(n_parallel_samples), 
                        desc="Optimization Iteration",
                        position=1, 
                        leave=False))
            return np.mean([score for score in scores])
            
        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sklearn warnings
            
            if verbose:
                print("Running Bayesian optimization...")
                
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=acquisition_function,
                random_state=self.random_seed
            )
        
        # Collect results
        param_names = list(param_ranges.keys())
        best_params = dict(zip(param_names, result.x))
        
        all_params = [
            dict(zip(param_names, x))
            for x in result.x_iters
        ]
        
        return OptimizationResults(
            best_parameters=best_params,
            best_score=result.fun,
            all_parameters=all_params,
            all_scores=result.func_vals,
            convergence_info={
                'models': result.models,
                'space': result.space,
                'specs': result.specs
            }
        )