"""
Tests for HMDB integration in priors module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from apriomics.priors import (
    PriorData,
    get_hmdb_contexts,
    get_metabolite_context_for_llm,
    get_llm_differential_priors,
    get_network_priors,
    save_results,
)
from apriomics.hmdb_utils import EXAMPLE_METABOLITE_MAPPINGS


class TestHMDBContextIntegration:
    """Test HMDB context integration in priors."""

    def test_priordata_with_hmdb_contexts(self):
        """Test PriorData class with HMDB contexts."""
        hmdb_contexts = {"glucose": "Context for glucose"}
        priors = PriorData(dimensions=512, hmdb_contexts=hmdb_contexts)
        assert priors.hmdb_contexts == hmdb_contexts
        assert priors.dimensions == 512

    @patch("apriomics.priors.batch_get_metabolite_contexts")
    def test_get_hmdb_contexts(self, mock_batch_get):
        """Test get_hmdb_contexts function."""
        mock_batch_get.return_value = {
            "glucose": "Glucose context",
            "pyruvate": "Pyruvate context",
        }

        priors = PriorData(dimensions=1024)
        metabolites = ["glucose", "pyruvate"]

        result = get_hmdb_contexts(priors, metabolites)

        assert result.hmdb_contexts == {
            "glucose": "Glucose context",
            "pyruvate": "Pyruvate context",
        }
        assert result.dimensions == 1024
        mock_batch_get.assert_called_once_with(metabolites, EXAMPLE_METABOLITE_MAPPINGS)

    @patch("apriomics.priors.batch_get_metabolite_contexts")
    def test_get_hmdb_contexts_custom_mapping(self, mock_batch_get):
        """Test get_hmdb_contexts with custom mapping."""
        custom_mapping = {"glucose": "HMDB0000122"}
        mock_batch_get.return_value = {"glucose": "Custom glucose context"}

        priors = PriorData()
        metabolites = ["glucose"]

        result = get_hmdb_contexts(priors, metabolites, custom_mapping)

        mock_batch_get.assert_called_once_with(metabolites, custom_mapping)


class TestMetaboliteContextForLLM:
    """Test metabolite context generation for LLM."""

    def test_context_with_condition_only(self):
        """Test context generation with condition only."""
        priors = PriorData()
        condition = "diabetes vs control"

        result = get_metabolite_context_for_llm(priors, condition)
        assert "Study condition: diabetes vs control" in result

    def test_context_with_hmdb_data(self):
        """Test context generation with HMDB data."""
        hmdb_contexts = {
            "glucose": "Glucose: blood sugar",
            "pyruvate": "Pyruvate: glycolysis product",
        }
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        result = get_metabolite_context_for_llm(priors, "diabetes study")

        assert "Study condition: diabetes study" in result
        assert "Metabolite information from HMDB:" in result
        assert "- Glucose: blood sugar" in result
        assert "- Pyruvate: glycolysis product" in result

    def test_context_with_similarity_data(self):
        """Test context generation with similarity data."""
        similarity_matrix = np.array(
            [[1.0, 0.9, 0.3], [0.9, 1.0, 0.2], [0.3, 0.2, 1.0]]
        )
        metabolite_names = ["glucose", "fructose", "alanine"]

        priors = PriorData(
            similarity_matrix=similarity_matrix, metabolite_names=metabolite_names
        )

        result = get_metabolite_context_for_llm(priors)

        assert "Chemical similarity data available for 3 metabolites." in result
        assert "Highly similar metabolite pairs" in result
        assert "glucose ↔ fructose (similarity: 0.90)" in result

    def test_context_with_all_data(self):
        """Test context generation with all data types."""
        hmdb_contexts = {"glucose": "Glucose context"}
        similarity_matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
        metabolite_names = ["glucose", "fructose"]

        priors = PriorData(
            hmdb_contexts=hmdb_contexts,
            similarity_matrix=similarity_matrix,
            metabolite_names=metabolite_names,
        )

        result = get_metabolite_context_for_llm(priors, "test condition")

        assert "Study condition: test condition" in result
        assert "Metabolite information from HMDB:" in result
        assert "Chemical similarity data available" in result


class TestLLMDifferentialPriors:
    """Test LLM differential priors generation."""

    def test_no_hmdb_contexts_raises_error(self):
        """Test that missing HMDB contexts raises error."""
        priors = PriorData()

        with pytest.raises(ValueError, match="No HMDB contexts available"):
            get_llm_differential_priors(priors, "condition")

    def test_no_llm_scorer_returns_uniform(self, capsys):
        """Test that no LLM scorer returns uniform priors."""
        hmdb_contexts = {"glucose": "Glucose context", "pyruvate": "Pyruvate context"}
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        result = get_llm_differential_priors(priors, "diabetes", llm_scorer=None)

        assert result == {"glucose": 0.5, "pyruvate": 0.5}
        captured = capsys.readouterr()
        assert "Warning: No LLM scorer provided" in captured.err

    def test_with_mock_llm_scorer(self):
        """Test with mock LLM scorer."""
        hmdb_contexts = {"glucose": "Glucose context", "pyruvate": "Pyruvate context"}
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        # Mock LLM scorer
        mock_scorer = Mock()
        mock_score_1 = Mock()
        mock_score_1.metabolite = "glucose"
        mock_score_1.score = 0.8
        mock_score_2 = Mock()
        mock_score_2.metabolite = "pyruvate"
        mock_score_2.score = 0.3

        mock_scorer.score_batch.return_value = [mock_score_1, mock_score_2]

        result = get_llm_differential_priors(priors, "diabetes", mock_scorer)

        assert result == {"glucose": 0.8, "pyruvate": 0.3}
        mock_scorer.score_batch.assert_called_once()

    def test_llm_scorer_error_handling(self, capsys):
        """Test error handling in LLM scorer."""
        hmdb_contexts = {"glucose": "Glucose context"}
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        mock_scorer = Mock()
        mock_scorer.score_batch.side_effect = Exception("LLM error")

        result = get_llm_differential_priors(priors, "condition", mock_scorer)

        assert result == {"glucose": 0.5}
        captured = capsys.readouterr()
        assert "Error processing LLM batch" in captured.err

    def test_batch_processing(self):
        """Test batch processing of metabolites."""
        hmdb_contexts = {f"metabolite_{i}": f"Context {i}" for i in range(25)}
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        mock_scorer = Mock()

        def mock_score_batch(condition, batch_contexts):
            scores = []
            for metabolite in batch_contexts.keys():
                mock_score = Mock()
                mock_score.metabolite = metabolite
                mock_score.score = 0.6
                scores.append(mock_score)
            return scores

        mock_scorer.score_batch.side_effect = mock_score_batch

        result = get_llm_differential_priors(
            priors, "condition", mock_scorer, batch_size=10
        )

        # Should call score_batch 3 times (25 metabolites / 10 batch size = 3 batches)
        assert mock_scorer.score_batch.call_count == 3
        assert len(result) == 25
        assert all(score == 0.6 for score in result.values())


class TestNetworkPriors:
    """Test network priors generation."""

    def test_no_similarity_matrix_raises_error(self):
        """Test that missing similarity matrix raises error."""
        priors = PriorData()

        with pytest.raises(ValueError, match="No similarity matrix available"):
            get_network_priors(priors)

    def test_network_priors_generation(self):
        """Test network priors generation from similarity matrix."""
        similarity_matrix = np.array(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.75], [0.3, 0.75, 1.0]]
        )
        metabolite_names = ["glucose", "fructose", "alanine"]

        priors = PriorData(
            similarity_matrix=similarity_matrix, metabolite_names=metabolite_names
        )

        result = get_network_priors(priors, threshold=0.7)

        expected = {("glucose", "fructose"): 0.8, ("fructose", "alanine"): 0.75}
        assert result == expected

    def test_network_priors_different_threshold(self):
        """Test network priors with different threshold."""
        similarity_matrix = np.array(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.75], [0.3, 0.75, 1.0]]
        )
        metabolite_names = ["glucose", "fructose", "alanine"]

        priors = PriorData(
            similarity_matrix=similarity_matrix, metabolite_names=metabolite_names
        )

        result = get_network_priors(priors, threshold=0.9)

        # No pairs should meet the 0.9 threshold
        assert result == {}

    def test_network_priors_low_threshold(self):
        """Test network priors with low threshold."""
        similarity_matrix = np.array(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.75], [0.3, 0.75, 1.0]]
        )
        metabolite_names = ["glucose", "fructose", "alanine"]

        priors = PriorData(
            similarity_matrix=similarity_matrix, metabolite_names=metabolite_names
        )

        result = get_network_priors(priors, threshold=0.2)

        expected = {
            ("glucose", "fructose"): 0.8,
            ("glucose", "alanine"): 0.3,
            ("fructose", "alanine"): 0.75,
        }
        assert result == expected


class TestSaveResults:
    """Test save results with new prior types."""

    def test_save_differential_priors(self, tmp_path):
        """Test saving differential priors."""
        priors = PriorData()
        differential_priors = {"glucose": 0.8, "pyruvate": 0.3}

        files = save_results(
            priors, output_dir=str(tmp_path), differential_priors=differential_priors
        )

        assert "differential_priors" in files
        diff_file = tmp_path / "differential_priors.csv"
        assert diff_file.exists()

        df = pd.read_csv(diff_file)
        assert len(df) == 2
        assert set(df["metabolite"].tolist()) == {"glucose", "pyruvate"}
        assert set(df["importance_score"].tolist()) == {0.8, 0.3}

    def test_save_network_priors(self, tmp_path):
        """Test saving network priors."""
        priors = PriorData()
        network_priors = {("glucose", "fructose"): 0.9, ("pyruvate", "lactate"): 0.7}

        files = save_results(
            priors, output_dir=str(tmp_path), network_priors=network_priors
        )

        assert "network_priors" in files
        network_file = tmp_path / "network_priors.csv"
        assert network_file.exists()

        df = pd.read_csv(network_file)
        assert len(df) == 2
        assert set(df["similarity_score"].tolist()) == {0.9, 0.7}

    def test_save_all_prior_types(self, tmp_path):
        """Test saving all types of priors together."""
        hmdb_contexts = {"glucose": "Glucose context"}
        priors = PriorData(hmdb_contexts=hmdb_contexts)

        differential_priors = {"glucose": 0.8}
        network_priors = {("glucose", "fructose"): 0.9}

        files = save_results(
            priors,
            output_dir=str(tmp_path),
            differential_priors=differential_priors,
            network_priors=network_priors,
        )

        assert "hmdb_contexts" in files
        assert "differential_priors" in files
        assert "network_priors" in files

        # Verify all files exist
        assert (tmp_path / "hmdb_contexts.csv").exists()
        assert (tmp_path / "differential_priors.csv").exists()
        assert (tmp_path / "network_priors.csv").exists()


# Fixtures
@pytest.fixture
def sample_priors_with_hmdb():
    """Fixture providing sample PriorData with HMDB contexts."""
    return PriorData(
        dimensions=1024,
        hmdb_contexts={
            "glucose": "Glucose: primary blood sugar",
            "pyruvate": "Pyruvate: end product of glycolysis",
        },
    )


@pytest.fixture
def sample_priors_with_similarity():
    """Fixture providing sample PriorData with similarity matrix."""
    similarity_matrix = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.2], [0.3, 0.2, 1.0]])
    metabolite_names = ["glucose", "fructose", "alanine"]

    return PriorData(
        similarity_matrix=similarity_matrix, metabolite_names=metabolite_names
    )


@pytest.fixture
def mock_llm_scorer():
    """Fixture providing a mock LLM scorer."""
    scorer = Mock()

    def mock_score_batch(condition, batch_contexts):
        scores = []
        for metabolite in batch_contexts.keys():
            mock_score = Mock()
            mock_score.metabolite = metabolite
            mock_score.score = 0.7  # Default score
            scores.append(mock_score)
        return scores

    scorer.score_batch.side_effect = mock_score_batch
    return scorer
