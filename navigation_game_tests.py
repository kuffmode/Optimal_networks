"""
Unit tests for the Network Navigation Game implementation.
"""

import numpy as np
import pytest
import warnings
from navigation_game import NavigationGame, NNGConfig, GameType

def create_sample_coordinates(n_nodes: int = 5, seed: int = 42) -> np.ndarray:
    """Create sample coordinates for testing."""
    np.random.seed(seed)
    return np.random.rand(n_nodes, 2)

def test_parameter_independence():
    """Test that deterministic mode ignores alpha and beta parameters."""
    coords = create_sample_coordinates()
    
    # Reference solution with default parameters
    config = NNGConfig(game_type=GameType.DETERMINISTIC)
    game = NavigationGame(config)
    ref_adj, _ = game.build_network(coords)
    
    # Solution with different alpha/beta should be identical
    config = NNGConfig(
        game_type=GameType.DETERMINISTIC,
        alpha=100.0,
        beta=1000.0
    )
    game = NavigationGame(config)
    
    with pytest.warns(RuntimeWarning, match="Alpha and beta parameters are ignored"):
        test_adj, _ = game.build_network(coords)
        
    np.testing.assert_array_equal(ref_adj, test_adj)

def test_parametric_convergence():
    """Test that parametric solution converges to deterministic with large beta."""
    coords = create_sample_coordinates()
    
    # Get deterministic solution
    det_config = NNGConfig(game_type=GameType.DETERMINISTIC)
    det_game = NavigationGame(det_config)
    det_adj, _ = det_game.build_network(coords)
    
    # Test convergence with increasing beta
    betas = [1, 10, 100, 1000]
    diffs = []
    
    for beta in betas:
        param_config = NNGConfig(
            game_type=GameType.PARAMETRIC,
            alpha=1.0,
            beta=beta
        )
        param_game = NavigationGame(param_config)
        param_adj, _ = param_game.build_network(coords)
        
        diff = np.sum(np.abs(det_adj - param_adj))
        diffs.append(diff)
    
    # Differences should decrease with increasing beta
    assert all(d1 >= d2 for d1, d2 in zip(diffs[:-1], diffs[1:]))
    
def test_symmetry_consistency():
    """Test that symmetry is maintained across game types."""
    coords = create_sample_coordinates()
    
    for game_type in GameType:
        config = NNGConfig(
            game_type=game_type,
            enforce_symmetry=True
        )
        game = NavigationGame(config)
        adj, _ = game.build_network(coords)
        
        # Check symmetry
        assert np.allclose(adj, adj.T), f"Symmetry violated in {game_type.value} mode"
        
        # Check it's still a valid solution
        assert game._verify_nash_equilibrium(adj), \
            f"Symmetric solution is not Nash equilibrium in {game_type.value} mode"

def test_uniqueness_warning():
    """Test warning about uniqueness in parametric mode."""
    coords = create_sample_coordinates()
    
    config = NNGConfig(
        game_type=GameType.PARAMETRIC,
        find_unique=True
    )
    game = NavigationGame(config)
    
    with pytest.warns(RuntimeWarning, match="Unique solution is not guaranteed"):
        game.build_network(coords)

def test_full_navigation_deterministic():
    """Test that deterministic solution achieves full navigation."""
    coords = create_sample_coordinates()
    
    config = NNGConfig(game_type=GameType.DETERMINISTIC)
    game = NavigationGame(config)
    adj, distances = game.build_network(coords)
    
    n = len(coords)
    for i in range(n):
        for j in range(n):
            if i != j:
                assert game._compute_navigation_success(i, adj[i], adj) == 1.0

def test_memory_warning():
    """Test memory requirement warning for large networks."""
    coords = create_sample_coordinates(n_nodes=1000)
    
    config = NNGConfig(find_unique=True)
    game = NavigationGame(config)
    
    with pytest.warns(RuntimeWarning, match="computationally expensive"):
        game.build_network(coords)

def test_invalid_parameters():
    """Test validation of alpha and beta parameters."""
    coords = create_sample_coordinates()
    
    # Test negative alpha
    config = NNGConfig(
        game_type=GameType.PARAMETRIC,
        alpha=-1.0
    )
    game = NavigationGame(config)
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        game.build_network(coords)
    
    # Test zero beta
    config = NNGConfig(
        game_type=GameType.PARAMETRIC,
        beta=0.0
    )
    game = NavigationGame(config)
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        game.build_network(coords)

def test_numerical_stability_warning():
    """Test warning about numerical stability with large beta."""
    coords = create_sample_coordinates()
    
    config = NNGConfig(
        game_type=GameType.PARAMETRIC,
        beta=1e7
    )
    game = NavigationGame(config)
    
    with pytest.warns(RuntimeWarning, match="numerical instability"):
        game.build_network(coords)

def test_consistent_results():
    """Test that multiple runs with same parameters give same results."""
    coords = create_sample_coordinates()
    
    # Test both game types
    for game_type in GameType:
        config = NNGConfig(game_type=game_type)
        game1 = NavigationGame(config)
        game2 = NavigationGame(config)
        
        adj1, _ = game1.build_network(coords)
        adj2, _ = game2.build_network(coords)
        
        np.testing.assert_array_equal(adj1, adj2)

def test_edge_cases():
    """Test edge cases and special configurations."""
    
    # Test minimal network (2 nodes)
    coords = create_sample_coordinates(n_nodes=2)
    config = NNGConfig()
    game = NavigationGame(config)
    adj, _ = game.build_network(coords)
    assert np.sum(adj) > 0  # Should have at least one connection
    
    # Test line configuration
    coords = np.array([[0,0], [1,0], [2,0]])
    config = NNGConfig()
    game = NavigationGame(config)
    adj, _ = game.build_network(coords)
    assert game._verify_nash_equilibrium(adj)
    
    # Test square configuration
    coords = np.array([[0,0], [0,1], [1,0], [1,1]])
    config = NNGConfig()
    game = NavigationGame(config)
    adj, _ = game.build_network(coords)
    assert game._verify_nash_equilibrium(adj)