import numpy as np
import pytest
from navigation_game import NavigableNetwork

def test_two_nodes():
    """Test the simplest possible case - two nodes."""
    coords = np.array([
        [0, 0],  # Node 0
        [1, 0],  # Node 1
    ])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    # Should get bidirectional connection
    expected = np.array([
        [False, True],
        [True, False]
    ])
    np.testing.assert_array_equal(nash, expected)
    assert network.verify_navigability(nash)

def test_triangle():
    """Test with isosceles triangle configuration."""
    coords = np.array([
        [0, 2],    # A (node 0)
        [-1, 0],   # B (node 1)
        [1, 0],    # C (node 2)
    ])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    # The expected adjacency should minimize connections while ensuring navigability
    expected = np.array([
        [False, True, True],
        [True, False, False],
        [True, False, False],
    ])
    
    print("Generated adjacency matrix:")
    print(nash)

    np.testing.assert_array_equal(nash, expected)
    assert network.verify_navigability(nash)

def test_square():
    """Test square configuration."""
    coords = np.array([
        [0, 1],  # Top left
        [1, 1],  # Top right
        [0, 0],  # Bottom left
        [1, 0],  # Bottom right
    ])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    # The expected adjacency should minimize connections while ensuring navigability
    expected = np.array([
        [False, True, True, False],
        [True, False, False, True],
        [True, False, False, True],
        [False, True, True, False],
    ])

    print("Generated adjacency matrix:")
    print(nash)

    np.testing.assert_array_equal(nash, expected)
    assert network.verify_navigability(nash)

def test_star_configuration():
    """Test 5-node star configuration with central hub."""
    # Create a star with one central node and 4 nodes around it
    r = 1.0  # radius
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    coords = np.zeros((5, 2))
    # Center node at origin
    coords[0] = [0, 0]
    # Surrounding nodes
    for i in range(4):
        coords[i+1] = [r * np.cos(angles[i]), r * np.sin(angles[i])]
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    assert nash.shape == (5, 5)
    assert not nash.diagonal().any()
    assert network.verify_navigability(nash)
    
    # Center node (0) should connect to all others
    assert np.all(nash[0, 1:])
    # All peripheral nodes should connect to center
    assert np.all(nash[1:, 0])

def test_line():
    """Test nodes arranged in a straight line."""
    coords = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
    ])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    assert nash.shape == (5, 5)
    assert not nash.diagonal().any()
    assert network.verify_navigability(nash)
    
    # Each node should at least connect to its immediate neighbors
    for i in range(4):
        assert nash[i, i+1] or nash[i+1, i]

def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    # Empty coordinates
    with pytest.raises(ValueError, match="Coordinates array cannot be empty"):
        NavigableNetwork(np.array([]))
    
    # Wrong dimensionality
    with pytest.raises(ValueError, match="Coordinates must be a 2D array"):
        NavigableNetwork(np.array([0, 0]))
    
    # Single node (this should work)
    single_node = NavigableNetwork(np.array([[0, 0]]))
    nash = single_node.build_nash_equilibrium()
    assert nash.shape == (1, 1)
    assert not nash[0,0]  # No self-loops
    
    # Nodes at same position
    coords = np.array([
        [0, 0],
        [0, 0],
        [1, 0]
    ])
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    assert network.verify_navigability(nash)

def test_pentagon():
    """Test regular pentagon configuration."""
    # Create regular pentagon vertices
    r = 1.0  # radius
    angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 equally spaced angles
    coords = np.array([[r * np.cos(angle), r * np.sin(angle)] for angle in angles])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    assert nash.shape == (5, 5)
    assert not nash.diagonal().any()
    assert network.verify_navigability(nash)

if __name__ == "__main__":
    pytest.main([__file__])
