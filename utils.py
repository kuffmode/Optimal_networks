"""Utility functions for network analysis and optimization."""
from typing import Optional, Callable, Dict, Any
import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy.stats import ks_2samp
from numba import njit
import warnings
from functools import lru_cache
from dask_config import DaskConfig, get_or_create_dask_client
from tqdm import tqdm

def evaluator(synthetic, empirical, euclidean_distance):
    """Network comparison using multiple metrics."""
    degrees_synthetic = np.sum(synthetic, axis=0)
    degrees_empirical = np.sum(empirical, axis=0)
    ks_degree = ks_2samp(degrees_synthetic, degrees_empirical)
    
    clustering_synthetic = nx.clustering(nx.from_numpy_array(synthetic))
    clustering_empirical = nx.clustering(nx.from_numpy_array(empirical))
    ks_clustering = ks_2samp(list(clustering_synthetic.values()), 
                            list(clustering_empirical.values()))
    
    betweenness_synthetic = nx.betweenness_centrality(nx.from_numpy_array(synthetic))
    betweenness_empirical = nx.betweenness_centrality(nx.from_numpy_array(empirical))
    ks_betweenness = ks_2samp(list(betweenness_synthetic.values()), 
                              list(betweenness_empirical.values()))
    
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    distance_empirical = euclidean_distance[np.triu(empirical, 1) > 0]
    ks_distance = ks_2samp(distance_synthetic, distance_empirical)
    
    results = {
        "energy": np.max([
            ks_degree[0], 
            ks_clustering[0], 
            ks_betweenness[0], 
            ks_distance[0]
        ]),
        "ks_degrees": ks_degree[0],
        "ks_clustering": ks_clustering[0],
        "ks_betweenness": ks_betweenness[0],
        "ks_distance": ks_distance[0]
    }
    
    return results

@njit
def fast_degrees(adj: np.ndarray) -> np.ndarray:
    """Compute degrees efficiently with numba."""
    return np.sum(adj, axis=0)

@njit
def fast_clustering(adj: np.ndarray) -> np.ndarray:
    """Compute clustering coefficients efficiently with numba."""
    n = len(adj)
    clustering = np.zeros(n)
    
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            continue
            
        triangles = 0
        for j in range(k):
            for l in range(j + 1, k):
                if adj[neighbors[j], neighbors[l]] > 0:
                    triangles += 1
                    
        possible_triangles = k * (k - 1) / 2
        clustering[i] = triangles / possible_triangles if possible_triangles > 0 else 0
    
    return clustering

@njit
def fast_betweenness(adj: np.ndarray) -> np.ndarray:
    """Approximate betweenness centrality using shortest paths."""
    n = len(adj)
    betweenness = np.zeros(n)
    
    for s in range(n):
        dist = np.full(n, np.inf)
        sigma = np.zeros(n)
        dist[s] = 0
        sigma[s] = 1
        
        queue = np.zeros(n, dtype=np.int64)
        queue[0] = s
        q_start, q_end = 0, 1
        
        visited = np.zeros(n, dtype=np.int64)
        n_visited = 0
        
        while q_start < q_end:
            v = queue[q_start]
            q_start += 1
            visited[n_visited] = v
            n_visited += 1
            
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                    
                if dist[w] == np.inf:
                    queue[q_end] = w
                    q_end += 1
                    dist[w] = dist[v] + 1
                
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        
        dependency = np.zeros(n)
        
        for i in range(n_visited - 1, -1, -1):
            v = visited[i]
            if v == s:
                continue
                
            coeff = (1 + dependency[v]) / sigma[v]
            
            for w in range(n):
                if adj[v, w] == 0:
                    continue
                if dist[w] == dist[v] - 1:
                    dependency[w] += sigma[w] * coeff
        
        betweenness += dependency
    
    betweenness = betweenness / (2 * ((n-1) * (n-2)))
    return betweenness

def fast_evaluator(
    synthetic: np.ndarray, 
    empirical: np.ndarray, 
    euclidean_distance: np.ndarray,
    cache_size: int = 128
) -> Dict[str, float]:
    """Optimized network comparison using KS tests on multiple metrics."""
    @lru_cache(maxsize=cache_size)
    def get_empirical_metrics(emp_key):
        emp = np.array(emp_key)
        return (
            fast_degrees(emp),
            fast_clustering(emp),
            fast_betweenness(emp),
            euclidean_distance[np.triu(emp, 1) > 0]
        )
    
    emp_key = tuple(map(tuple, empirical))
    
    (degrees_empirical,
     clustering_empirical,
     betweenness_empirical,
     distance_empirical) = get_empirical_metrics(emp_key)
    
    degrees_synthetic = fast_degrees(synthetic)
    clustering_synthetic = fast_clustering(synthetic)
    betweenness_synthetic = fast_betweenness(synthetic)
    distance_synthetic = euclidean_distance[np.triu(synthetic, 1) > 0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks_stats = [
            ks_2samp(degrees_synthetic, degrees_empirical)[0],
            ks_2samp(clustering_synthetic, clustering_empirical)[0],
            ks_2samp(betweenness_synthetic, betweenness_empirical)[0],
            ks_2samp(distance_synthetic, distance_empirical)[0]
        ]
    
    return {
        "energy": np.max(ks_stats),
        "ks_degrees": ks_stats[0],
        "ks_clustering": ks_stats[1],
        "ks_betweenness": ks_stats[2],
        "ks_distance": ks_stats[3]
    }

@njit
def density_distance(final_network, empirical):
    """Compute absolute difference in network densities."""
    density_synthetic = np.sum(final_network) / (final_network.shape[0] * final_network.shape[0])
    density_empirical = np.sum(empirical) / (empirical.shape[0] * empirical.shape[0])
    return abs(density_synthetic - density_empirical)


@dataclass
class PSOResults:
    """Container for PSO optimization results."""
    best_parameters: Dict[str, float]
    best_score: float
    position_history: np.ndarray
    cost_history: np.ndarray

class ParallelNetworkOptimizer:
    """Network optimizer using PSO with Dask-based parallel processing."""
    
    def __init__(
        self,
        simulation_model: Callable,
        evaluation_function: Callable,
        param_bounds: Dict[str, tuple],
        dask_config: Optional[DaskConfig] = None,
        sim_kwargs: Dict[str, Any] = {},
        eval_kwargs: Dict[str, Any] = {},
        random_seed: Optional[int] = None
    ):
        """Initialize the PSO optimizer."""
        self.simulation_model = simulation_model
        self.evaluation_function = evaluation_function
        self.sim_kwargs = sim_kwargs
        self.eval_kwargs = eval_kwargs
        self.param_names = list(param_bounds.keys())
        self.bounds = (
            np.array([b[0] for b in param_bounds.values()]),
            np.array([b[1] for b in param_bounds.values()])
        )
        self.dask_config = dask_config or DaskConfig()
        
        if random_seed is not None:
            np.random.seed(random_seed)

    def _run_single_simulation(self, params: Dict[str, float]) -> float:
        # Disable Dask client creation inside simulations
        full_params = self.sim_kwargs.copy()
        full_params.update(params)
        full_params["dask_config"] = None  # Prevent nested client creation
        
        # Convert scalar parameters to full trajectories
        for param in self.param_names:
            if isinstance(full_params[param], (int, float)):
                full_params[param] = np.full(
                    self.sim_kwargs['n_iterations'], 
                    full_params[param]
                )
        
        # Run simulation
        history = self.simulation_model(**full_params)
        return self.evaluation_function(history[:, :, -1], **self.eval_kwargs)

    def optimize(
    self,
    n_particles: int = 20,
    n_iterations: int = 50,
    pso_options: Dict[str, float] = None
) -> PSOResults:
        """Run PSO optimization with proper Dask client handling."""
        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
        if pso_options:
            options.update(pso_options)

        dimensions = len(self.param_names)
        positions = np.random.uniform(
            low=self.bounds[0], 
            high=self.bounds[1], 
            size=(n_particles, dimensions)
        )
        velocities = np.random.uniform(
            low=-(self.bounds[1] - self.bounds[0]), 
            high=(self.bounds[1] - self.bounds[0]), 
            size=(n_particles, dimensions)
        )
        
        pbest_positions = positions.copy()
        pbest_scores = np.full(n_particles, np.inf)
        position_history = []
        all_scores = []
        
        # Get existing client once before loop
        client = get_or_create_dask_client(self.dask_config)
        
        # Initial evaluation
        futures = [client.submit(
            self._run_single_simulation,
            dict(zip(self.param_names, pos))
        ) for pos in positions]
        scores = np.array(client.gather(futures))
        
        pbest_scores = scores.copy()
        gbest_idx = np.argmin(scores)
        gbest_position = positions[gbest_idx].copy()
        gbest_score = scores[gbest_idx]
        
        # Optimization loop
        with tqdm(total=n_iterations, desc=f"PSO (n_particles={n_particles})") as pbar:
            for iteration in range(n_iterations):
                r1, r2 = np.random.rand(2)
                velocities = (
                    options['w'] * velocities +
                    options['c1'] * r1 * (pbest_positions - positions) +
                    options['c2'] * r2 * (gbest_position - positions)
                )
                positions = np.clip(
                    positions + velocities,
                    self.bounds[0],
                    self.bounds[1]
                )
                
                # Evaluate new positions
                futures = [client.submit(
                    self._run_single_simulation,
                    dict(zip(self.param_names, pos))
                ) for pos in positions]
                scores = np.array(client.gather(futures))
                
                # Update personal best
                improved = scores < pbest_scores
                pbest_positions[improved] = positions[improved]
                pbest_scores[improved] = scores[improved]
                
                # Update global best
                current_best_idx = np.argmin(scores)
                if scores[current_best_idx] < gbest_score:
                    gbest_position = positions[current_best_idx].copy()
                    gbest_score = scores[current_best_idx]
                
                # Store history
                position_history.append(positions.copy())
                all_scores.append(scores.copy())
                pbar.update(1)
                
                # Optional: Add progress bar description
                pbar.set_postfix({"best_score": gbest_score})

        return PSOResults(
            best_parameters=dict(zip(self.param_names, gbest_position)),
            best_score=gbest_score,
            position_history=np.array(position_history),
            cost_history=np.array(all_scores)
        )