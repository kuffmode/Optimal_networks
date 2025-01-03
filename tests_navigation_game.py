import numpy as np
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

def test_toy_example():
    """Test the example in the supplementary material (S1)."""
    coords = np.array([
        [1, 1],  # A
        [3, 1],  # B
        [3, 2],  # C
        [1, 5],  # D
    ])
    
    network = NavigableNetwork(coords)
    nash = network.build_nash_equilibrium()
    
    # Should get this
    expected = np.array([[False, False,  True, False],
                         [ True, False,  True, False],
                         [False,  True, False,  True],
                         [False, False,  True, False]])
    
    assert expected.all() == nash.all()