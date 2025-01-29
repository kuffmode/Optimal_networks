"""Utility functions for network analysis and optimization."""
from typing import Tuple, List, Optional, Union, Callable, Dict, Any
import numpy as np
import networkx as nx
from dataclasses import dataclass
import dask.distributed as dd
import dask.array as da
from dask.distributed import Client, LocalCluster, SSHCluster
from contextlib import contextmanager
import os
from scipy.stats import ks_2samp
from numba import njit
import warnings
from functools import lru_cache
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from joblib import Parallel, delayed
from tqdm import tqdm
from generative import simulate_network_evolution, resistance_distance
from pyswarms.single.global_best import GlobalBestPSO
import subprocess

@dataclass
class DaskConfig:
    """Configuration for Dask cluster setup.
    
    Attributes
    ----------
    cluster_type : str
        Type of cluster to use ('local' or 'slurm')
    n_workers : Tuple[int, int]
        (simulation_workers, spo_workers) for nested parallelization
    scheduler_options : Dict
        Additional options for the Dask scheduler
    worker_options : Dict
        Additional options for Dask workers
    slurm_options : Optional[Dict]
        SLURM-specific options when using SLURM cluster
    """
    cluster_type: str = 'local'
    n_workers: Tuple[int, int] = (4, 2)  # (simulation_workers, spo_workers)
    scheduler_options: Dict = None
    worker_options: Dict = None
    slurm_options: Optional[Dict] = None

@contextmanager
def dask_cluster(config: DaskConfig):
    """Context manager for setting up and tearing down a Dask cluster.
    
    Parameters
    ----------
    config : DaskConfig
        Configuration for the Dask cluster
        
    Yields
    -------
    client : dask.distributed.Client
        Connected Dask client
    """
    client = None
    try:
        if config.cluster_type == 'local':
            cluster = LocalCluster(
                n_workers=sum(config.n_workers),
                threads_per_worker=1,
                **(config.scheduler_options or {}),
                **(config.worker_options or {})
            )
        elif config.cluster_type == 'slurm':
            if not config.slurm_options:
                raise ValueError("SLURM options required for SLURM cluster")
                
            # Create SLURM cluster
            cluster = dd.SSHCluster(
                scheduler_options=config.scheduler_options,
                worker_options=config.worker_options,
                **config.slurm_options
            )
        else:
            raise ValueError(f"Unknown cluster type: {config.cluster_type}")
            
        client = Client(cluster)
        yield client
    finally:
        if client:
            client.close()
            
def get_available_cores() -> int:
    """Get number of available CPU cores, accounting for SLURM if present."""
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    return len(os.sched_getaffinity(0))

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
class OptimizationResults:
    """Container for optimization results."""
    best_parameters: Dict[str, float]
    best_score: float
    all_parameters: List[Dict[str, float]]
    all_scores: List[float]
    convergence_info: Dict[str, Any]

class NetworkOptimizer:
    """Optimizer for network evolution parameters using Dask-based parallel processing."""
    
    def __init__(
        self,
        coordinates: np.ndarray,
        empirical_network: np.ndarray,
        n_iterations: int = 1000,
        evaluator: Optional[Callable] = None,
        dask_config: Optional[DaskConfig] = None,
        distance_fn: Optional[Callable] = resistance_distance,
        evaluator_kwargs: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize the optimizer with Dask support."""
        self.coordinates = coordinates
        self.empirical_network = empirical_network
        self.n_iterations = n_iterations
        self.dask_config = dask_config or DaskConfig()
        self.random_seed = random_seed
        self.distance_fn = distance_fn
        self.evaluator_kwargs = evaluator_kwargs if evaluator_kwargs else {}
        
        self.evaluator = evaluator if evaluator else density_distance
        self.euclidean_distance = self._compute_euclidean_distances()
        
    def _compute_euclidean_distances(self) -> np.ndarray:
        """Compute euclidean distances once."""
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(self.coordinates))
        
    def _create_parameter_space(
        self,
        param_ranges: Dict[str, tuple]
    ) -> List[Real]:
        """Create skopt parameter space from ranges."""
        return [
            Real(low, high, name=name)
            for name, (low, high) in param_ranges.items()
        ]
        
    def _simulate_with_params(
        self,
        params: Dict[str, float],
        client: dd.Client,
        batch_id: Optional[int] = None
    ) -> float:
        """Run simulation with given parameters using Dask."""
        # Create parameter trajectories
        alpha = params.get('alpha', np.full(self.n_iterations, 1.0))
        beta = params.get('beta', np.full(self.n_iterations, 0.1))
        noise = params.get('noise', np.zeros(self.n_iterations))
        penalty = params.get('connectivity_penalty', np.zeros(self.n_iterations))
        
        if 'beta_growth' in params:
            beta = np.linspace(0, beta, self.n_iterations)
        if 'noise_std' in params:
            noise = np.random.normal(0, params['noise_std'], self.n_iterations)
            
        # Convert arrays to Dask arrays for distributed computing
        alpha_da = da.from_array(alpha, chunks='auto')
        beta_da = da.from_array(beta, chunks='auto')
        noise_da = da.from_array(noise, chunks='auto')
        penalty_da = da.from_array(penalty, chunks='auto')
        
        # Run simulation with Dask
        history = simulate_network_evolution(
            coordinates=self.coordinates,
            n_iterations=self.n_iterations,
            distance_fn=self.distance_fn,
            alpha=alpha_da,
            beta=beta_da,
            noise=noise_da,
            connectivity_penalty=penalty_da,
            n_jobs=self.dask_config.n_workers[0],  # Use simulation workers
            random_seed=self.random_seed + batch_id if batch_id is not None else None
        )
        
        # Evaluate final network
        final_network = history[:, :, -1]
        score = self.evaluator(
            final_network,
            self.empirical_network,
            self.euclidean_distance,
            **self.evaluator_kwargs
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
        """Run Bayesian optimization with Dask-based parallel processing."""
        space = self._create_parameter_space(param_ranges)
        
        with dask_cluster(self.dask_config) as client:
            @use_named_args(space)
            def objective(**params):
                futures = [
                    client.submit(
                        self._simulate_with_params,
                        params,
                        client,
                        i
                    )
                    for i in range(n_parallel_samples)
                ]
                scores = client.gather(futures)
                return np.mean([score["energy"] if isinstance(score, dict) else score 
                              for score in scores])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if verbose:
                    print(f"Running Bayesian optimization with Dask ({self.dask_config.cluster_type} cluster)")
                    
                result = gp_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    acq_func=acquisition_function,
                    random_state=self.random_seed
                )
        
        param_names = list(param_ranges.keys())
        best_params = dict(zip(param_names, result.x))
        all_params = [dict(zip(param_names, x)) for x in result.x_iters]
        
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
        """Initialize the PSO optimizer with Dask support."""
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

    def _objective(self, positions: np.ndarray, client: dd.Client) -> np.ndarray:
        """Evaluate positions using Dask."""
        futures = []
        
        for pos in positions:
            params = dict(zip(self.param_names, pos))
            
            # Submit simulation to Dask
            future = client.submit(
                self.simulation_model,
                **params,
                **self.sim_kwargs
            )
            futures.append(future)
            
        # Gather results and evaluate
        results = client.gather(futures)
        scores = []
        
        for result in results:
            score = self.evaluation_function(
                result[:, :, -1],
                **self.eval_kwargs
            )
            scores.append(score["energy"] if isinstance(score, dict) else score)
            
        return np.array(scores)

    def optimize(
        self,
        n_particles: int = 20,
        n_iterations: int = 50,
        pso_kwargs: Dict[str, Any] = {}
    ) -> PSOResults:
        """Run PSO optimization with Dask-based parallel processing."""
        with dask_cluster(self.dask_config) as client:
            optimizer = GlobalBestPSO(
                n_particles=n_particles,
                dimensions=len(self.param_names),
                options={
                    'c1': 1.5,
                    'c2': 1.5,
                    'w': 0.7,
                    'k': self.dask_config.n_workers[1],  # Use SPO workers
                    'p': 2,
                    **pso_kwargs
                },
                bounds=self.bounds
            )

            best_cost, best_pos = optimizer.optimize(
                lambda x: self._objective(x, client),
                iters=n_iterations,
                n_processes=self.dask_config.n_workers[1],
                verbose=True
            )

        return PSOResults(
            best_parameters=dict(zip(self.param_names, best_pos)),
            best_score=best_cost,
            position_history=optimizer.pos_history,
            cost_history=optimizer.cost_history
        )
