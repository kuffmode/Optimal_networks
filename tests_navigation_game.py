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

if __name__ == "__main__":
    pytest.main([__file__])
