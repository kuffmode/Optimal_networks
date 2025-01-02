"""
Implementation of Network Navigation Game (NNG) based on Gulyás et al. 2015.
Includes both deterministic and parametric versions with options for symmetry and unique solutions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import numpy as np
import numba as nb
from joblib import Parallel, delayed
import warnings
import psutil

class GameType(Enum):
    DETERMINISTIC = "deterministic"  # Original formulation (β = ∞)
    PARAMETRIC = "parametric"        # Modified version with finite α, β

@dataclass
class NNGConfig:
    """Configuration for Network Navigation Game."""
    # Core settings
    find_unique: bool = True
    enforce_symmetry: bool = False
    n_jobs: int = -1
    chunk_size: int = 1000
    seed: Optional[int] = None
    verbose: bool = True
    
    # Parametric game settings
    game_type: GameType = GameType.DETERMINISTIC
    alpha: float = 1.0  # Weight for wiring cost
    beta: float = 1.0   # Weight for navigation penalty
    
    def validate_config(self):
        """Validate configuration settings and their interactions."""
        # Deterministic case should ignore α and β
        if self.game_type == GameType.DETERMINISTIC:
            if self.alpha != 1.0 or self.beta != 1.0:
                warnings.warn(
                    "Alpha and beta parameters are ignored in deterministic mode.",
                    RuntimeWarning
                )
                
        # Warn about uniqueness in parametric case
        if self.game_type == GameType.PARAMETRIC and self.find_unique:
            warnings.warn(
                "Unique solution is not guaranteed in parametric mode. "
                "Setting find_unique=True may not yield expected results.",
                RuntimeWarning
            )
        
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Alpha and beta must be positive")

@nb.njit(fastmath=True)
def compute_distances(coordinates: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between points."""
    n = len(coordinates)
    distances = np.zeros((n, n))
    
    for i in nb.prange(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
            distances[i, j] = distances[j, i] = dist
    
    return distances

@nb.njit(parallel=True, fastmath=True)
def compute_coverage_areas(node_idx: int, distances: np.ndarray) -> np.ndarray:
    """Compute coverage areas for all potential neighbors of a node."""
    n = len(distances)
    coverage = np.zeros((n, n), dtype=nb.boolean)
    eps = 1e-10  # Numerical tolerance
    
    for i in nb.prange(n):
        if i != node_idx:
            for j in range(n):
                if j != node_idx and j != i:
                    coverage[i, j] = distances[j, i] < distances[j, node_idx] - eps
            coverage[i, i] = True
    
    return coverage

@nb.njit
def solve_min_set_cover(universe: np.ndarray, coverage_sets: np.ndarray) -> np.ndarray:
    """Solve minimum set cover using greedy algorithm."""
    n_sets = len(coverage_sets)
    remaining = universe.copy()
    selected = np.zeros(n_sets, dtype=nb.boolean)
    
    while np.any(remaining):
        best_coverage = 0
        best_set = -1
        
        for i in range(n_sets):
            if not selected[i]:
                coverage = np.sum(coverage_sets[i] & remaining)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_set = i
        
        if best_set == -1 or best_coverage == 0:
            break
            
        selected[best_set] = True
        remaining &= ~coverage_sets[best_set]
    
    return np.where(selected)[0]

def check_navigability(
    adjacency: np.ndarray,
    distances: np.ndarray,
    source: int,
    target: int,
    max_steps: Optional[int] = None
) -> bool:
    """Check if greedy routing can navigate from source to target."""
    if max_steps is None:
        max_steps = len(adjacency)
        
    current = source
    visited = {source}
    steps = 0
    current_dist = distances[source, target]
    
    while current != target and steps < max_steps:
        # Get all neighbors
        neighbors = set(np.where(adjacency[current])[0])
        if not neighbors:
            return False
            
        # Find neighbor closest to target
        best_dist = float('inf')
        best_neighbor = None
        
        for neighbor in neighbors:
            dist = distances[neighbor, target]
            if dist < best_dist and dist < current_dist:
                best_dist = dist
                best_neighbor = neighbor
                
        if best_neighbor is None:
            return False
            
        if best_neighbor in visited:
            return False
            
        current = best_neighbor
        current_dist = distances[current, target]
        visited.add(current)
        steps += 1
    
    return current == target

class NavigationGame:
    """Enhanced Network Navigation Game implementation."""
    
    def __init__(self, config: NNGConfig):
        self.config = config
        self.adjacency = None
        self.distances = None
        self.coordinates = None
        
    def validate_state(self):
        """Comprehensive validation of game state and settings."""
        if self.coordinates is None or self.distances is None:
            raise ValueError("Coordinates and distances must be set before building network")
            
        n = len(self.coordinates)
        
        if self.config.game_type == GameType.PARAMETRIC and self.config.beta > 1e6:
            warnings.warn(
                "Very large beta values may cause numerical instability. "
                "Consider using deterministic mode instead.",
                RuntimeWarning
            )
            
        if n > 100 and self.config.find_unique:
            warnings.warn(
                "Finding unique solution for large networks may be computationally expensive",
                RuntimeWarning
            )
            
        estimated_memory = self._estimate_memory_requirement()
        available_memory = psutil.virtual_memory().available
        if estimated_memory > 0.8 * available_memory:
            warnings.warn(
                "Operation may exceed available memory",
                ResourceWarning
            )
    
    def _estimate_memory_requirement(self) -> int:
        """Estimate memory requirement for computation."""
        n = len(self.coordinates)
        # Basic matrices (adjacency, distances, coverage)
        basic_memory = 3 * n * n * 8  # 8 bytes per float64
        
        # Additional memory for unique solution search
        if self.config.find_unique:
            # Rough estimate for storing multiple solutions
            basic_memory *= 10
            
        return basic_memory
        
    def build_network(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build Nash equilibrium network."""
        self.coordinates = coordinates
        self.distances = compute_distances(coordinates)
        
        # Validate configuration and state
        self.config.validate_config()
        self.validate_state()
        
        if self.config.game_type == GameType.DETERMINISTIC:
            self.adjacency = self._build_deterministic_network()
        else:
            self.adjacency = self._build_parametric_network()
            
        if self.config.enforce_symmetry:
            self.adjacency = self._symmetrize_solution(self.adjacency)
            
        return self.adjacency, self.distances
    
    def _build_deterministic_network(self) -> np.ndarray:
        """Build network using original deterministic formulation."""
        n = len(self.coordinates)
        
        if self.config.find_unique:
            return self._find_unique_minimal_solution()
        
        # Basic Nash equilibrium using minimum set cover
        adjacency = np.zeros((n, n), dtype=bool)
        for i in range(n):
            strategy = self._compute_deterministic_strategy(i)
            adjacency[i] = strategy
            
        return adjacency
    
    def _build_parametric_network(self, max_iterations: int = 100) -> np.ndarray:
        """Non-recursive parametric network construction."""
        n = len(self.coordinates)
        adjacency = np.zeros((n, n), dtype=bool)
        
        for iteration in range(max_iterations):
            old_adjacency = adjacency.copy()
            
            for i in range(n):
                strategy = self._compute_parametric_strategy(i, adjacency)
                adjacency[i] = strategy
                
            if np.array_equal(adjacency, old_adjacency):
                return adjacency
                
        warnings.warn(f"Network did not converge after {max_iterations} iterations")
        return adjacency
    
    def _compute_deterministic_strategy(self, node_idx: int) -> np.ndarray:
        """Compute strategy for deterministic case using minimum set cover."""
        n = len(self.distances)
        universe = np.ones(n, dtype=bool)
        universe[node_idx] = False
        
        coverage_areas = compute_coverage_areas(node_idx, self.distances)
        selected_nodes = solve_min_set_cover(universe, coverage_areas)
        
        connections = np.zeros(n, dtype=bool)
        connections[selected_nodes] = True
        
        # Add frame edges
        for j in range(n):
            if j != node_idx:
                is_closest = True
                for k in range(n):
                    if k != j and k != node_idx:
                        if self.distances[j, k] < self.distances[j, node_idx]:
                            is_closest = False
                            break
                if is_closest:
                    connections[j] = True
        
        return connections
    
    def _compute_parametric_strategy(self, node_idx: int, current_adjacency: np.ndarray) -> np.ndarray:
        """Compute strategy for parametric case optimizing cost-navigation trade-off."""
        n = len(self.distances)
        best_cost = float('inf')
        best_strategy = None
        
        # For small networks, try all strategies
        # For larger networks, would need more sophisticated optimization
        for strategy in self._iterate_possible_strategies(n, node_idx):
            wiring_cost = self.config.alpha * np.sum(
                strategy * self.distances[node_idx]
            )
            nav_success = self._compute_navigation_success(
                node_idx, strategy, current_adjacency
            )
            nav_cost = self.config.beta * (1 - nav_success)
            
            total_cost = wiring_cost + nav_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_strategy = strategy.copy()
                
        return best_strategy
    
    def _iterate_possible_strategies(self, n: int, node_idx: int):
        """Generate possible strategies for a node."""
        # For small networks, generate all possibilities
        # This should be replaced with more sophisticated search for larger networks
        strategy = np.zeros(n, dtype=bool)
        for i in range(2**(n-1)):
            idx = 0
            for j in range(n):
                if j != node_idx:
                    strategy[j] = bool(i & (1 << idx))
                    idx += 1
            yield strategy.copy()
    
    def _compute_navigation_success(
        self, 
        node_idx: int, 
        strategy: np.ndarray,
        current_adjacency: np.ndarray
    ) -> float:
        """Compute fraction of successful navigations from node to all others."""
        n = len(self.distances)
        successful = 0
        total = n - 1  # Exclude self
        
        temp_adj = current_adjacency.copy()
        temp_adj[node_idx] = strategy
        
        for target in range(n):
            if target != node_idx:
                if check_navigability(temp_adj, self.distances, node_idx, target):
                    successful += 1
                    
        return successful / total
    
    def _find_unique_minimal_solution(self, max_attempts: int = 1000) -> np.ndarray:
        """Non-recursive unique solution finder."""
        n = len(self.coordinates)
        basic_solution = self._build_deterministic_network()
        
        if not self.config.find_unique:
            return basic_solution
            
        best_distance = self._compute_total_edge_distance(basic_solution)
        best_solution = basic_solution
        
        all_node_solutions = []
        for i in range(n):
            coverage_areas = compute_coverage_areas(i, self.distances)
            solutions = self._find_all_minimal_covers(i, coverage_areas)
            all_node_solutions.append(solutions)
            
        for combination in self._iterate_valid_combinations(all_node_solutions, max_attempts):
            if self._verify_nash_equilibrium(combination):
                total_dist = self._compute_total_edge_distance(combination)
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_solution = combination.copy()
                    
        return best_solution
    
    def _find_all_minimal_covers(self, node_idx: int, coverage_areas: np.ndarray, max_attempts: int = 1000) -> List[np.ndarray]:
        """Find all minimal set covers for a node using iteration instead of recursion."""
        n = len(self.distances)
        universe = np.ones(n, dtype=bool)
        universe[node_idx] = False
        
        # Get size bound from greedy solution
        greedy_cover = solve_min_set_cover(universe, coverage_areas)
        min_size = np.sum(greedy_cover)
        minimal_covers = [greedy_cover]
        
        def is_valid_cover(cover: np.ndarray) -> bool:
            reached = np.zeros(n, dtype=bool)
            for i, included in enumerate(cover):
                if included:
                    reached |= coverage_areas[i]
            return np.all(reached[universe])
        
        # Iterative approach using systematic enumeration
        attempts = 0
        for attempt in range(max_attempts):
            cover = np.zeros(n, dtype=bool)
            selected = 0
            
            # Randomly select nodes until we have min_size selections
            idxs = np.random.permutation(n)
            for idx in idxs:
                if selected >= min_size:
                    break
                if idx != node_idx:
                    cover[idx] = True
                    selected += 1
                    
            if is_valid_cover(cover) and not any(np.array_equal(cover, existing) for existing in minimal_covers):
                minimal_covers.append(cover)
                
            attempts += 1
            if attempts >= max_attempts:
                break
                
        return minimal_covers
    
    def _iterate_valid_combinations(self, all_node_solutions: List[List[np.ndarray]], 
                              max_attempts: int = 1000) -> np.ndarray:
        """Non-recursive combination generator."""
        n = len(all_node_solutions)
        
        for _ in range(max_attempts):
            # Random combination
            combination = [
                solutions[np.random.randint(len(solutions))]
                for solutions in all_node_solutions
            ]
            yield np.array(combination)
    
    def _compute_total_edge_distance(self, adjacency: np.ndarray) -> float:
        """Compute total distance spanned by edges."""
        return np.sum(adjacency * self.distances)
    
    def _symmetrize_solution(self, adjacency: np.ndarray) -> np.ndarray:
        """Create symmetric version of solution while preserving navigability."""
        n = len(adjacency)
        
        # Initial symmetrization
        symmetric = np.logical_or(adjacency, adjacency.T)
        
        if self._verify_nash_equilibrium(symmetric):
            return self._minimize_symmetric_solution(symmetric)
        
        # If basic symmetrization breaks Nash equilibrium,
        # add minimal symmetric pairs until we achieve it
        current = adjacency.copy()
        
        for i in range(n):
            for j in range(i+1, n):
                if adjacency[i,j] != adjacency[j,i]:
                    current[i,j] = current[j,i] = True
                    if not self._verify_nash_equilibrium(current):
                        current[i,j] = adjacency[i,j]
                        current[j,i] = adjacency[j,i]
                        
        return current
    
    def _minimize_symmetric_solution(self, symmetric: np.ndarray) -> np.ndarray:
        """Try to remove symmetric pairs while maintaining Nash equilibrium."""
        n = len(symmetric)
        current = symmetric.copy()
        
        # Try removing symmetric pairs in order of total distance
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if symmetric[i,j]:
                    dist = self.distances[i,j] + self.distances[j,i]
                    edges.append((dist, i, j))
                    
        # Sort by distance descending - try to remove longest edges first
        edges.sort(reverse=True)
        
        for _, i, j in edges:
            current[i,j] = current[j,i] = False
            if not self._verify_nash_equilibrium(current):
                current[i,j] = current[j,i] = True
                
        return current
    
    def _verify_nash_equilibrium(self, adjacency: np.ndarray, max_attempts: int = 1000) -> bool:
        """Non-recursive Nash equilibrium verification."""
        n = len(adjacency)
        
        for i in range(n):
            current_cost = self._compute_strategy_cost(i, adjacency[i], adjacency)
            
            # Use iterator instead of recursion
            for attempt in range(max_attempts):
                # Generate random strategy
                alt_strategy = np.random.randint(2, size=n, dtype=bool)
                alt_strategy[i] = False  # No self-loops
                
                alt_cost = self._compute_strategy_cost(i, alt_strategy, adjacency)
                if alt_cost < current_cost - 1e-10:
                    return False
                    
        return True
    
    def _compute_strategy_cost(self, node_idx: int, strategy: np.ndarray, adjacency: np.ndarray) -> float:
        """Compute cost of a strategy for a node."""
        if self.config.game_type == GameType.DETERMINISTIC:
            # In deterministic case, cost is number of edges if fully navigable, infinity otherwise
            nav_success = self._compute_navigation_success(node_idx, strategy, adjacency)
            return np.sum(strategy) if nav_success == 1.0 else float('inf')
        else:
            # In parametric case, weighted sum of wiring cost and navigation penalty
            wiring_cost = self.config.alpha * np.sum(strategy * self.distances[node_idx])
            nav_success = self._compute_navigation_success(node_idx, strategy, adjacency)
            nav_cost = self.config.beta * (1 - nav_success)
            return wiring_cost + nav_cost
