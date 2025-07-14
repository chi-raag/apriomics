"""
Tests for the priors.base module.
"""

import pytest
import numpy as np
import pandas as pd
from apriomics.priors.base import load_mtbls1_data

class TestLoadMTBLS1Data:
    """Test the load_mtbls1_data function."""

    def test_load_mtbls1_data_success(self, tmp_path):
        """Test that the function correctly loads a sample MTBLS1 file."""
        # Create a dummy MTBLS1 file
        data = {
            'metabolite_identification': ['A', 'B', 'C'],
            'ADG10003u_007': [1.0, 2.0, 3.0],
            'ADG10003u_008': [4.0, 5.0, 6.0]
        }
        df = pd.DataFrame(data)
        file_path = tmp_path / "mtbls1_data.tsv"
        df.to_csv(file_path, sep='\t', index=False)

        metabolite_names, sample_names, abundance_data = load_mtbls1_data(file_path)

        assert metabolite_names == ['A', 'B', 'C']
        assert sample_names == ['ADG10003u_007', 'ADG10003u_008']
        np.testing.assert_array_equal(abundance_data, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).T)

    def test_load_mtbls1_data_file_not_found(self):
        """Test that the function raises an error if the file is not found."""
        with pytest.raises(FileNotFoundError):
            load_mtbls1_data("non_existent_file.tsv")

