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
from tqdm_joblib import tqdm_joblib
from sklearn.metrics.pairwise import cosine_similarity
from generative import resistance_distance, shortest_path_distance, propagation_distance, topological_distance
import pandas as pd
import bct

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

def check_density(adj):
    return np.sum(adj)/(adj.shape[0] * (adj.shape[0]-1))

def calculate_wiring_cost(adj, euclidean_distance):
    return np.mean(adj * euclidean_distance)

def calculate_endpoint_similarity(synthetic_matrix, empirical_matrix):
    similarities = np.zeros(synthetic_matrix.shape[0])
    for i in range(synthetic_matrix.shape[0]):
        similarities[i] = cosine_similarity(synthetic_matrix[i].reshape(1, -1),
                                            empirical_matrix[i].reshape(1, -1))
    return similarities


def evaluate_adjacency(empirical_adj, simulated_adj):
    """
    Compute Accuracy and F1 score between two binary adjacency matrices.
    
    Args:
        empirical_adj (np.ndarray): Empirical adjacency (0/1), shape (n, n)
        simulated_adj (np.ndarray): Simulated adjacency (0/1), shape (n, n)
    
    Returns:
        (accuracy, f1): Tuple of floats
            - accuracy is in [0, 100] (percentage)
            - f1 is in [0, 1]
    """
    # Ensure inputs are numpy arrays and flattened (if we want to ignore diagonal, filter it out)
    # Here, we keep the entire matrix as is, including diagonal if present.
    A_emp = empirical_adj.astype(int).ravel()
    A_sim = simulated_adj.astype(int).ravel()

    # Calculate confusion matrix components:
    TP = np.sum((A_emp == 1) & (A_sim == 1))
    FP = np.sum((A_emp == 0) & (A_sim == 1))
    TN = np.sum((A_emp == 0) & (A_sim == 0))
    FN = np.sum((A_emp == 1) & (A_sim == 0))

    # Accuracy (percentage)
    total = TP + FP + TN + FN
    accuracy = 100.0 * (TP + TN) / total if total > 0 else 0.0

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score (in [0, 1])
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return accuracy, f1

def randomize_graph(G, nswap=None, max_tries=None):
    """Randomizes the graph using double edge swap."""
    if nswap is None:
        nswap = 20 * G.number_of_edges()  # Adjust as needed
    if max_tries is None:
        max_tries = nswap * 20
    G_random = nx.double_edge_swap(G.copy(), nswap=nswap, max_tries=max_tries)
    return G_random

def compute_random_metrics(G, nrandomizations=10, nswap=None, max_tries=None):
    """
    Computes the average clustering and average shortest path length over multiple randomized versions
    of the input graph.
    """
    clustering_vals = []
    path_length_vals = []
    
    for _ in range(nrandomizations):
        Gr = randomize_graph(G, nswap, max_tries)
        clustering_vals.append(nx.average_clustering(Gr))
        try:
            path_length_vals.append(nx.average_shortest_path_length(Gr))
        except nx.NetworkXError:
            # In case the randomized graph is disconnected, compute the metric for the largest connected component.
            largest_cc = max(nx.connected_components(Gr), key=len)
            Gr_sub = Gr.subgraph(largest_cc)
            path_length_vals.append(nx.average_shortest_path_length(Gr_sub))
    
    return np.mean(clustering_vals), np.mean(path_length_vals)

def compute_sigma(adj_matrix, nrandomizations=32, nswap=None, max_tries=None):
    """
    Computes the smallworld index sigma for a graph given its adjacency matrix.
    
    Sigma is defined as:
        sigma = (C / C_rand) / (L / L_rand)
    where:
        C      = average clustering coefficient of the graph
        L      = characteristic path length of the graph
        C_rand = average clustering coefficient of randomized graphs
        L_rand = average characteristic path length of randomized graphs
    """
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Compute original graph metrics
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # For disconnected graphs, compute L on the largest connected component.
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
    
    # Compute metrics for randomized graphs
    Crand, Lrand = compute_random_metrics(G, nrandomizations, nswap, max_tries)
    
    # Calculate and return sigma
    sigma = (C / Crand) / (L / Lrand)
    return sigma

def compute_omega(adj_matrix, nrandomizations=100, nswap=None, max_tries=None):
    """
    Computes the omega (ω) small-world metric for an undirected network given its adjacency matrix.
    
    ω is defined as:
        ω = (L_rand / L) - (C / C_latt)
    
    where:
        L     = average shortest path length of the original graph
        C     = average clustering coefficient of the original graph
        L_rand = average shortest path length of a randomized version of the graph (averaged over multiple realizations)
        C_latt = clustering coefficient of a lattice (regular) graph generated to approximate the equivalent lattice
    
    Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix.
        nrandomizations (int): Number of random networks to average for L_rand.
        nswap, max_tries: Parameters passed to the double edge swap (randomization).
    
    Returns:
        omega (float): The small-world omega metric.
    """
    # Create the original graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    n = G.number_of_nodes()
    
    # Compute clustering (C) and path length (L) for the original network.
    C = nx.average_clustering(G)
    try:
        L = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        # If the graph is disconnected, use the largest connected component.
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
    
    # Compute L_rand: average shortest path length of randomized networks.
    L_rand_values = []
    for _ in range(nrandomizations):
        Gr = randomize_graph(G, nswap, max_tries)
        try:
            L_rand_values.append(nx.average_shortest_path_length(Gr))
        except nx.NetworkXError:
            # Use the largest connected component if Gr is disconnected.
            largest_cc = max(nx.connected_components(Gr), key=len)
            Gr_sub = Gr.subgraph(largest_cc)
            L_rand_values.append(nx.average_shortest_path_length(Gr_sub))
    L_rand = np.mean(L_rand_values)
    
    laticized = bct.latmio_und(adj_matrix,itr=100)[0]
    G_lattice = nx.from_numpy_array(laticized)
    C_latt = nx.average_clustering(G_lattice)
    
    # Compute omega according to the formula.
    omega = (L_rand / L) - (C / C_latt)
    return omega

def compute_graph_metrics(simulated_tensor,
                          empirical_adjmat,
                          euclidean_distance,
                          coordinates):
    measure_labels = ["diffusion distance",
                      "shortest path distance",
                      "propagation distance",
                      "topological distance",
                      "density",
                      "wiring cost",
                      "average clustering",
                      "degree assortativity",
                      "small-worldness",
                      "endpoint similarity",
                      "accuracy",
                      "F1 score"]
    measures = np.zeros((len(measure_labels),
                         simulated_tensor.shape[2]))

    for timepoint in tqdm(range(simulated_tensor.shape[2]),desc="Computing graph metrics"):
        G = nx.from_numpy_array(simulated_tensor[:,:,timepoint])
        measures[0,timepoint] = resistance_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[1,timepoint] = shortest_path_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[2,timepoint] = propagation_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        measures[3,timepoint] = topological_distance(simulated_tensor[:,:,timepoint],coordinates).mean()
        
        measures[4,timepoint] = check_density(simulated_tensor[:,:,timepoint])
        measures[5,timepoint] = calculate_wiring_cost(simulated_tensor[:,:,timepoint],euclidean_distance)
        
        measures[6,timepoint] = nx.average_clustering(G)
        measures[7,timepoint] = nx.degree_assortativity_coefficient(G)
        measures[8,timepoint] = nx.smallworld.sigma(G)
        
        measures[9,timepoint] = calculate_endpoint_similarity(simulated_tensor[:,:,timepoint],empirical_adjmat).mean()
        measures[10,timepoint], measures[11,timepoint] = evaluate_adjacency(empirical_adjmat,simulated_tensor[:,:,timepoint])
    return pd.DataFrame(measures.T,columns=measure_labels)

def _compute_metrics_for_timepoint(timepoint, 
                                   simulated_tensor, 
                                   empirical_adjmat, 
                                   euclidean_distance, 
                                   coordinates):
    """
    Computes all metrics for a given timepoint and returns them as a 1D array.
    """
    sim = simulated_tensor[:, :, timepoint]
    G = nx.from_numpy_array(sim)
    
    metrics = np.zeros(12)
    metrics[0] = resistance_distance(sim, coordinates).mean()
    metrics[1] = shortest_path_distance(sim, coordinates).mean()
    metrics[2] = propagation_distance(sim, coordinates).mean()
    metrics[3] = topological_distance(sim, coordinates).mean()
    metrics[4] = check_density(sim)
    metrics[5] = calculate_wiring_cost(sim, euclidean_distance)
    metrics[6] = nx.average_clustering(G)
    metrics[7] = nx.degree_assortativity_coefficient(G)
    metrics[8] = compute_omega(sim)
    metrics[9] = calculate_endpoint_similarity(sim, empirical_adjmat).mean()
    metrics[10], metrics[11] = evaluate_adjacency(empirical_adjmat, sim)
    
    return metrics

def compute_graph_metrics_parallel(simulated_tensor, 
                                   empirical_adjmat, 
                                   euclidean_distance, 
                                   coordinates):
    measure_labels = [
        "diffusion distance",
        "shortest path distance",
        "propagation distance",
        "topological distance",
        "density",
        "wiring cost",
        "average clustering",
        "degree assortativity",
        "small-worldness",
        "endpoint similarity",
        "accuracy",
        "F1 score"
    ]
    
    n_timepoints = simulated_tensor.shape[2]
    
    # Use tqdm_joblib to wrap joblib.Parallel for a progress bar.
    with tqdm_joblib(tqdm(total=n_timepoints, desc="Computing graph metrics")):
        results = Parallel(n_jobs=-1)(
            delayed(_compute_metrics_for_timepoint)(tp, simulated_tensor, empirical_adjmat, euclidean_distance, coordinates)
            for tp in range(n_timepoints)
        )
    
    # Convert list of 1D arrays into a 2D NumPy array.
    # Each element in 'results' is of shape (12,), so we stack along axis 1.
    measures = np.array(results).T  # Shape: (12, n_timepoints)
    
    # Return a DataFrame with rows corresponding to timepoints.
    return pd.DataFrame(measures.T, columns=measure_labels)

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
    
    This class provides a flexible framework for optimizing network evolution parameters
    by allowing injection of custom simulation models and evaluation functions.
    """
    
    def __init__(
        self,
        simulation_model: Callable,
        evaluation_function: Callable,
        param_ranges: Dict[str, tuple],
        sim_kwargs: Dict[str, Any],
        eval_kwargs: Dict[str, Any],
        n_parallel: int = -1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            simulation_model: Function that simulates network evolution
            evaluation_function: Function that evaluates network quality
            param_ranges: Dictionary of parameter names and (min, max) ranges to optimize
            sim_kwargs: Fixed keyword arguments for simulation model
            eval_kwargs: Fixed keyword arguments for evaluation function
            n_parallel: Number of parallel jobs (-1 for all cores)
            random_seed: Random seed for reproducibility
        """
        self.simulation_model = simulation_model
        self.evaluation_function = evaluation_function
        self.param_ranges = param_ranges
        self.sim_kwargs = sim_kwargs
        self.eval_kwargs = eval_kwargs
        self.n_parallel = n_parallel
        self.random_seed = random_seed
        
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
        """
        Run simulation with given parameters and evaluate.
        
        Args:
            params: Dictionary of parameters to optimize
            batch_id: Optional batch identifier for parallel runs
        
        Returns:
            float: Evaluation score
        """
        # Update simulation kwargs with optimization parameters
        sim_kwargs = self.sim_kwargs.copy()
        sim_kwargs.update(params)
        
        # Add batch-specific random seed if needed
        if batch_id is not None and 'random_seed' in sim_kwargs:
            sim_kwargs['random_seed'] = sim_kwargs['random_seed'] + batch_id
            
        # Run simulation
        history = self.simulation_model(**sim_kwargs)
        
        # Get final network state
        final_network = history[:, :, -1] if history.ndim == 3 else history
        
        # Evaluate network
        score = self.evaluation_function(final_network, **self.eval_kwargs)
        
        # Handle both dict and float return types
        return score['energy'] if isinstance(score, dict) else score
        
    def optimize(
        self,
        n_calls: int = 50,
        n_initial_points: int = 10,
        n_parallel_samples: int = 16,
        acquisition_function: str = "gp_hedge",
        verbose: bool = True,
        **optimizer_kwargs
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
        space = self._create_parameter_space(self.param_ranges)

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
        param_names = list(self.param_ranges.keys())
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
