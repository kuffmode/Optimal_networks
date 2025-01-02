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


def test_line_symmetric():
    """Test a symmetric line configuration."""
    coords = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0]
    ])

    network = NavigableNetwork(coords)
    adjacency = network.build_nash_equilibrium(symmetry='forced')

    # Expected symmetric adjacency matrix
    expected = np.array([
        [False, True,  False, False],
        [True,  False, True,  False],
        [False, True,  False, True ],
        [False, False, True,  False]
    ])

    np.testing.assert_array_equal(adjacency, expected)
    assert network.verify_navigability(adjacency)


def test_pentagon_symmetric():
    """Test a symmetric pentagon configuration."""
    coords = np.array([
        [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 6)[:-1]
    ])

    network = NavigableNetwork(coords)
    adjacency = network.build_nash_equilibrium(symmetry='forced')

    # Expected symmetric adjacency matrix
    expected = np.array([
        [False, True,  False, False, True ],
        [True,  False, True,  False, False],
        [False, True,  False, True,  False],
        [False, False, True,  False, True ],
        [True,  False, False, True,  False]
    ])

    np.testing.assert_array_equal(adjacency, expected)
    assert network.verify_navigability(adjacency)

if __name__ == "__main__":
    pytest.main([__file__])
