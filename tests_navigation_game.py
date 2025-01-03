import numpy as np
import pytest
from navigation_game import NavigableNetwork
from tqdm import tqdm

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

@pytest.mark.parametrize("num_nodes", range(10, 101, 10))
def test_nash_network_navigability(num_nodes):
    """
    Test that removing any edge from a Nash equilibrium network results in a loss of full navigability.

    This test generates networks of increasing size, verifies navigability,
    and iteratively removes links to ensure no remaining configuration is navigable.
    """
    # Generate random coordinates for nodes
    rng = np.random.default_rng(11)

    coordinates = rng.random((num_nodes, 2)) * 100  # Nodes scattered in a 100x100 space

    # Initialize the network
    network = NavigableNetwork(coordinates)
    adjacency_matrix = network.build_nash_equilibrium()

    # Verify initial full navigability
    assert network.verify_navigability(adjacency_matrix), "Initial network is not fully navigable."

    # Convert adjacency matrix to a list of edges
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if adjacency_matrix[i, j]]

    # Use tqdm to track progress
    for edge in tqdm(edges, desc=f"Testing {num_nodes}-node network", unit="edge"):
        modified_adjacency = adjacency_matrix.copy()
        modified_adjacency[edge[0], edge[1]] = False  # Remove the edge

        # Ensure the modified network is no longer fully navigable
        assert not network.verify_navigability(modified_adjacency), f"Network remained navigable after removing edge {edge}."

