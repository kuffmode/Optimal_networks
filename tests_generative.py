import unittest
import numpy as np
from generative import (
    jit_safe,
    _diag_indices,
    _set_diagonal,
    process_matrix,
    validate_parameters,
    get_param_value,
    compute_component_sizes,
    propagation_distance,
    resistance_distance,
    normalized_resistance_distance,
    heat_kernel_distance,
    shortest_path_distance,
    topological_distance,
    combined_objectives,
    compute_node_payoff,
)


class TestGenerative(unittest.TestCase):
    def test_jit_safe(self):
        @jit_safe()
        def add(x, y):
            return x + y

        self.assertEqual(add(2, 3), 5)

    def test_diag_indices(self):
        rows, cols = _diag_indices(3)
        self.assertTrue(np.array_equal(rows, np.array([0, 1, 2])))
        self.assertTrue(np.array_equal(cols, np.array([0, 1, 2])))

    def test_set_diagonal(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
        result = _set_diagonal(matrix.copy())
        self.assertTrue(np.array_equal(result, expected))

        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array([[2, 2, 3], [4, 2, 6], [7, 8, 2]])
        result = _set_diagonal(matrix.copy(), 2)
        self.assertTrue(np.array_equal(result, expected))

    def test_process_matrix(self):
        matrix = np.array(
            [[1.0, np.nan, np.inf], [-np.inf, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = process_matrix(matrix)
        self.assertTrue(np.array_equal(result, expected))

    def test_validate_parameters(self):
        # Test correct inputs
        validate_parameters(
            1000,
            1.0,
            np.ones(1000),
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )

        # Test incorrect inputs
        with self.assertRaises(ValueError):
            validate_parameters(
                1000,
                np.ones(1000),
                1.0,
                np.ones(1000),
                np.ones(1000),
                np.ones(1000) * 2,
                names=("alpha", "beta", "noise", "connectivity", "batch_size"),
                allow_float=(True, False, False, True, True),
                allow_zero=(False, False, True, False, False),
            )
        with self.assertRaises(ValueError):
            validate_parameters(
                1000,
                0.0,
                np.ones(1000),
                np.ones(1000),
                np.ones(1000),
                np.ones(1000) * 2,
                names=("alpha", "beta", "noise", "connectivity", "batch_size"),
                allow_float=(True, False, False, True, True),
                allow_zero=(False, False, True, False, False),
            )
        with self.assertRaises(ValueError):
            validate_parameters(
                1000,
                1.0,
                np.zeros(1000),
                np.ones(1000),
                np.ones(1000),
                np.ones(1000) * 2,
                names=("alpha", "beta", "noise", "connectivity", "batch_size"),
                allow_float=(True, False, False, True, True),
                allow_zero=(False, False, True, False, False),
            )
        with self.assertRaises(ValueError):
            validate_parameters(
                1000,
                1.0,
                np.ones(100),
                np.ones(1000),
                np.ones(1000),
                np.ones(1000) * 2,
                names=("alpha", "beta", "noise", "connectivity", "batch_size"),
                allow_float=(True, False, False, True, True),
                allow_zero=(False, False, True, False, False),
            )
        with self.assertRaises(ValueError):
            validate_parameters(
                1000,
                "hello",
                np.ones(1000),
                np.ones(1000),
                np.ones(1000),
                np.ones(1000) * 2,
                names=("alpha", "beta", "noise", "connectivity", "batch_size"),
                allow_float=(True, False, False, True, True),
                allow_zero=(False, False, True, False, False),
            )
        validate_parameters(
            1000,
            1.0,
            np.ones(1000),
            None,
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
            allow_none=(False, False, True, False, False),
        )

    def test_get_param_value(self):
        self.assertEqual(get_param_value(1.0, 0), 1.0)
        self.assertEqual(get_param_value(np.array([1.0, 2.0, 3.0]), 1), 2.0)

    def test_compute_component_sizes(self):
        adjacency = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        expected = np.array([2.0, 2.0, 1.0])
        result = compute_component_sizes(adjacency)
        self.assertTrue(np.array_equal(result, expected))

    def test_propagation_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = propagation_distance(adjacency)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))

    def test_resistance_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        coordinates = np.array([[0, 0], [1, 0], [0, 1]])
        result = resistance_distance(adjacency, coordinates)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))

    def test_normalized_resistance_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        coordinates = np.array([[0, 0], [1, 0], [0, 1]])
        result = normalized_resistance_distance(adjacency, coordinates)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))

    def test_heat_kernel_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = heat_kernel_distance(adjacency)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))
        result = heat_kernel_distance(adjacency, normalize=True)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))

    def test_shortest_path_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = shortest_path_distance(adjacency)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < np.inf))
        
    def test_topological_distance(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = topological_distance(adjacency)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result<=1))

    def test_combined_objectives(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        coordinates = np.array([[0, 0], [1, 0], [0, 1]])
        result = combined_objectives(adjacency, coordinates)
        self.assertEqual(result.shape, (3, 3))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result< np.inf))

    def test_compute_node_payoff(self):
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        coordinates = np.array([[0, 0], [1, 0], [0, 1]])
        distance_fn = shortest_path_distance
        result = compute_node_payoff(
            0, adjacency, coordinates, distance_fn, 1.0, 1.0, 0.0, 1.0
        )
        self.assertIsInstance(result, float)
