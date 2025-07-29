"""
Functions for building graph-based priors from metabolic networks.
"""

import numpy as np
from typing import List, Tuple, Dict


def build_laplacian_matrix(
    metabolite_ids: List[str], edges: List[Tuple[str, str]]
) -> np.ndarray:
    """
    Builds the graph Laplacian matrix from a list of metabolites and their connections.

    The graph Laplacian is defined as L = D - A, where D is the degree matrix
    and A is the adjacency matrix.

    Args:
        metabolite_ids: A list of all HMDB IDs included in the analysis.
                        The order of this list determines the order of the
                        rows and columns in the resulting matrix.
        edges: A list of tuples, where each tuple is a pair of HMDB IDs
               representing an edge in the metabolic network.

    Returns:
        A numpy array representing the graph Laplacian matrix.
    """
    num_metabolites = len(metabolite_ids)
    if num_metabolites == 0:
        return np.array([])

    # Create a mapping from HMDB ID to index for quick lookups
    id_to_index: Dict[str, int] = {
        hmdb_id: i for i, hmdb_id in enumerate(metabolite_ids)
    }

    # Initialize adjacency matrix (A) and degree matrix (D)
    adjacency_matrix = np.zeros((num_metabolites, num_metabolites))
    degree_matrix = np.zeros((num_metabolites, num_metabolites))

    for edge in edges:
        # Ensure both metabolites in the edge are in our list
        if edge[0] in id_to_index and edge[1] in id_to_index:
            idx1 = id_to_index[edge[0]]
            idx2 = id_to_index[edge[1]]

            # Update adjacency matrix for the undirected graph
            adjacency_matrix[idx1, idx2] = 1
            adjacency_matrix[idx2, idx1] = 1

    # Calculate the degree matrix
    for i in range(num_metabolites):
        degree_matrix[i, i] = np.sum(adjacency_matrix[i, :])

    # Calculate the Laplacian: L = D - A
    laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix
