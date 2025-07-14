"""
Tests for the graph_priors module.
"""

import pytest
import numpy as np
from apriomics.priors.graph_priors import build_laplacian_matrix

class TestBuildLaplacianMatrix:
    """Test the build_laplacian_matrix function."""

    def test_simple_graph(self):
        """Test with a simple, connected graph."""
        metabolite_ids = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]

        # Expected Adjacency Matrix:
        #   A B C
        # A 0 1 0
        # B 1 0 1
        # C 0 1 0

        # Expected Degree Matrix:
        #   A B C
        # A 1 0 0
        # B 0 2 0
        # C 0 0 1

        # Expected Laplacian (L = D - A):
        #   A  B  C
        # A  1 -1  0
        # B -1  2 -1
        # C  0 -1  1
        expected_laplacian = np.array([
            [1, -1,  0],
            [-1, 2, -1],
            [0, -1,  1]
        ])

        result = build_laplacian_matrix(metabolite_ids, edges)
        np.testing.assert_array_equal(result, expected_laplacian)

    def test_disconnected_graph(self):
        """Test a graph with a disconnected component."""
        metabolite_ids = ["A", "B", "C", "D"]
        edges = [("A", "B")] # C and D are disconnected

        # Expected Laplacian:
        #   A  B  C  D
        # A  1 -1  0  0
        # B -1  1  0  0
        # C  0  0  0  0
        # D  0  0  0  0
        expected_laplacian = np.array([
            [1, -1, 0, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        result = build_laplacian_matrix(metabolite_ids, edges)
        np.testing.assert_array_equal(result, expected_laplacian)

    def test_empty_graph(self):
        """Test with no metabolites."""
        result = build_laplacian_matrix([], [])
        assert result.shape == (0,)

    def test_no_edges(self):
        """Test with metabolites but no edges."""
        metabolite_ids = ["A", "B", "C"]
        edges = []
        expected_laplacian = np.zeros((3, 3))
        result = build_laplacian_matrix(metabolite_ids, edges)
        np.testing.assert_array_equal(result, expected_laplacian)

    def test_edge_with_unknown_metabolite(self):
        """Test that edges with metabolites not in the main list are ignored."""
        metabolite_ids = ["A", "B"]
        edges = [("A", "B"), ("B", "C")] # C is not in metabolite_ids

        expected_laplacian = np.array([
            [1, -1],
            [-1, 1]
        ])

        result = build_laplacian_matrix(metabolite_ids, edges)
        np.testing.assert_array_equal(result, expected_laplacian)
