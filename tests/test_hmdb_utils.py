"""
Tests for hmdb_utils module.
"""

import pytest
from unittest.mock import patch, Mock
import xml.etree.ElementTree as ET
import requests
from apriomics.hmdb_utils import (
    HMDBMetabolite,
    search_hmdb_metabolite,
    get_hmdb_metabolite_data,
    get_metabolite_context,
    batch_get_metabolite_contexts,
    _get_xml_text,
    EXAMPLE_METABOLITE_MAPPINGS,
    build_hmdb_graph,
)


class TestHMDBMetabolite:
    """Test the HMDBMetabolite dataclass."""

    def test_init_with_defaults(self):
        metabolite = HMDBMetabolite(hmdb_id="HMDB0000001", name="Test Metabolite")
        assert metabolite.hmdb_id == "HMDB0000001"
        assert metabolite.name == "Test Metabolite"
        assert metabolite.pathways == []
        assert metabolite.diseases == []
        assert metabolite.tissues == []
        assert metabolite.biospecimens == []
        assert metabolite.concentrations == []

    def test_init_with_data(self):
        metabolite = HMDBMetabolite(
            hmdb_id="HMDB0000001",
            name="Test Metabolite",
            description="A test metabolite",
            pathways=["Glycolysis"],
            diseases=["Diabetes"],
        )
        assert metabolite.description == "A test metabolite"
        assert metabolite.pathways == ["Glycolysis"]
        assert metabolite.diseases == ["Diabetes"]


class TestSearchHMDBMetabolite:
    """Test search functionality."""

    def test_empty_name(self, capsys):
        result = search_hmdb_metabolite("")
        assert result is None
        captured = capsys.readouterr()
        assert "Warning: Empty metabolite name provided." in captured.err

    def test_none_name(self, capsys):
        result = search_hmdb_metabolite(None)
        assert result is None
        captured = capsys.readouterr()
        assert "Warning: Empty metabolite name provided." in captured.err

    def test_search_returns_none(self):
        # Current implementation returns None (placeholder)
        result = search_hmdb_metabolite("glucose")
        assert result is None


class TestGetXMLText:
    """Test the XML parsing helper function."""

    def test_get_xml_text_success(self):
        xml_string = "<root><name>Test Name</name></root>"
        root = ET.fromstring(xml_string)
        result = _get_xml_text(root, "name")
        assert result == "Test Name"

    def test_get_xml_text_missing_tag(self):
        xml_string = "<root><name>Test Name</name></root>"
        root = ET.fromstring(xml_string)
        result = _get_xml_text(root, "missing", "default")
        assert result == "default"

    def test_get_xml_text_empty_tag(self):
        xml_string = "<root><name></name></root>"
        root = ET.fromstring(xml_string)
        result = _get_xml_text(root, "name", "default")
        assert result == "default"

    def test_get_xml_text_none_element(self):
        result = _get_xml_text(None, "name", "default")
        assert result == "default"

    def test_get_xml_text_strips_whitespace(self):
        xml_string = "<root><name>  Test Name  </name></root>"
        root = ET.fromstring(xml_string)
        result = _get_xml_text(root, "name")
        assert result == "Test Name"


class TestGetHMDBMetaboliteData:
    """Test HMDB metabolite data retrieval."""

    def test_invalid_hmdb_id(self, capsys):
        result = get_hmdb_metabolite_data("invalid")
        assert result is None
        captured = capsys.readouterr()
        assert "Warning: Invalid HMDB ID provided" in captured.err

    def test_empty_hmdb_id(self, capsys):
        result = get_hmdb_metabolite_data("")
        assert result is None
        captured = capsys.readouterr()
        assert "Warning: Invalid HMDB ID provided" in captured.err

    @patch("apriomics.hmdb_utils.requests.get")
    def test_successful_data_retrieval(self, mock_get):
        # Mock successful HTTP response with sample XML
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <metabolite>
            <name>Glucose</name>
            <description>A simple sugar</description>
            <chemical_formula>C6H12O6</chemical_formula>
            <smiles>C(C1C(C(C(C(O1)O)O)O)O)O</smiles>
            <biological_properties>
                <pathways>
                    <pathway><name>Glycolysis</name></pathway>
                    <pathway><name>Gluconeogenesis</name></pathway>
                </pathways>
            </biological_properties>
            <diseases>
                <disease><name>Diabetes mellitus</name></disease>
            </diseases>
            <tissue_locations>
                <tissue><name>Blood</name></tissue>
                <tissue><name>Liver</name></tissue>
            </tissue_locations>
            <biospecimen_locations>
                <biospecimen><name>Urine</name></biospecimen>
            </biospecimen_locations>
            <normal_concentrations>
                <concentration>
                    <biospecimen>Blood</biospecimen>
                    <concentration_value>5.5</concentration_value>
                    <concentration_units>mM</concentration_units>
                    <subject_age>Adult</subject_age>
                    <subject_sex>Both</subject_sex>
                </concentration>
            </normal_concentrations>
        </metabolite>"""

        mock_response = Mock()
        mock_response.text = sample_xml
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_hmdb_metabolite_data("HMDB0000122")

        assert result is not None
        assert result.hmdb_id == "HMDB0000122"
        assert result.name == "Glucose"
        assert result.description == "A simple sugar"
        assert result.chemical_formula == "C6H12O6"
        assert result.smiles == "C(C1C(C(C(C(O1)O)O)O)O)O"
        assert "Glycolysis" in result.pathways
        assert "Gluconeogenesis" in result.pathways
        assert "Diabetes mellitus" in result.diseases
        assert "Blood" in result.tissues
        assert "Liver" in result.tissues
        assert "Urine" in result.biospecimens
        assert len(result.concentrations) == 1
        assert result.concentrations[0]["biospecimen"] == "Blood"
        assert result.concentrations[0]["value"] == "5.5"
        assert result.concentrations[0]["units"] == "mM"

    @patch("apriomics.hmdb_utils.requests.get")
    def test_request_timeout(self, mock_get, capsys):
        mock_get.side_effect = requests.exceptions.Timeout()

        result = get_hmdb_metabolite_data("HMDB0000122")
        assert result is None
        captured = capsys.readouterr()
        assert "Timeout occurred" in captured.err

    @patch("apriomics.hmdb_utils.requests.get")
    def test_request_error(self, mock_get, capsys):
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        result = get_hmdb_metabolite_data("HMDB0000122")
        assert result is None
        captured = capsys.readouterr()
        assert "Error fetching HMDB data" in captured.err

    @patch("apriomics.hmdb_utils.requests.get")
    def test_xml_parse_error(self, mock_get, capsys):
        mock_response = Mock()
        mock_response.text = "Invalid XML"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_hmdb_metabolite_data("HMDB0000122")
        assert result is None
        captured = capsys.readouterr()
        assert "Error parsing HMDB XML" in captured.err


class TestGetMetaboliteContext:
    """Test metabolite context generation."""

    def test_context_no_hmdb_id(self):
        with patch("apriomics.hmdb_utils.search_hmdb_metabolite", return_value=None):
            result = get_metabolite_context("glucose")
            assert result == "Metabolite: glucose (HMDB data not available)"

    def test_context_with_hmdb_id_no_data(self):
        with patch("apriomics.hmdb_utils.get_hmdb_metabolite_data", return_value=None):
            result = get_metabolite_context("glucose", "HMDB0000122")
            assert (
                result
                == "Metabolite: glucose (HMDB ID: HMDB0000122, data not accessible)"
            )

    def test_context_with_full_data(self):
        mock_metabolite = HMDBMetabolite(
            hmdb_id="HMDB0000122",
            name="Glucose",
            description="A simple sugar found in blood and other tissues",
            chemical_formula="C6H12O6",
            pathways=["Glycolysis", "Gluconeogenesis", "Pentose phosphate pathway"],
            diseases=["Diabetes mellitus", "Hypoglycemia"],
            tissues=["Blood", "Liver", "Muscle", "Brain", "Kidney"],
            concentrations=[
                {
                    "biospecimen": "Blood",
                    "value": "5.5",
                    "units": "mM",
                    "age": "Adult",
                    "sex": "Both",
                },
                {
                    "biospecimen": "Urine",
                    "value": "0.1",
                    "units": "mM",
                    "age": "Adult",
                    "sex": "Both",
                },
            ],
        )

        with patch(
            "apriomics.hmdb_utils.get_hmdb_metabolite_data",
            return_value=mock_metabolite,
        ):
            result = get_metabolite_context("glucose", "HMDB0000122")

            assert "Metabolite: Glucose (HMDB ID: HMDB0000122)" in result
            assert (
                "Description: A simple sugar found in blood and other tissues" in result
            )
            assert "Formula: C6H12O6" in result
            assert (
                "Pathways: Glycolysis, Gluconeogenesis, Pentose phosphate pathway"
                in result
            )
            assert "Associated diseases: Diabetes mellitus, Hypoglycemia" in result
            assert "Found in tissues: Blood, Liver, Muscle, Brain, Kidney" in result
            assert "Normal concentrations: Blood: 5.5 mM; Urine: 0.1 mM" in result

    def test_context_with_minimal_data(self):
        mock_metabolite = HMDBMetabolite(hmdb_id="HMDB0000122", name="Glucose")

        with patch(
            "apriomics.hmdb_utils.get_hmdb_metabolite_data",
            return_value=mock_metabolite,
        ):
            result = get_metabolite_context("glucose", "HMDB0000122")
            assert result == "Metabolite: Glucose (HMDB ID: HMDB0000122)"


class TestBatchGetMetaboliteContexts:
    """Test batch metabolite context retrieval."""

    def test_batch_without_mapping(self):
        metabolites = ["glucose", "pyruvate"]

        with patch("apriomics.hmdb_utils.get_metabolite_context") as mock_get_context:
            mock_get_context.side_effect = (
                lambda name, hmdb_id=None: f"Context for {name}"
            )

            result = batch_get_metabolite_contexts(metabolites)

            assert len(result) == 2
            assert result["glucose"] == "Context for glucose"
            assert result["pyruvate"] == "Context for pyruvate"

            # Verify that get_metabolite_context was called with None for hmdb_id
            assert mock_get_context.call_count == 2
            mock_get_context.assert_any_call("glucose", None)
            mock_get_context.assert_any_call("pyruvate", None)

    def test_batch_with_mapping(self):
        metabolites = ["glucose", "pyruvate"]
        mapping = {"glucose": "HMDB0000122", "pyruvate": "HMDB0000243"}

        with patch("apriomics.hmdb_utils.get_metabolite_context") as mock_get_context:
            mock_get_context.side_effect = (
                lambda name, hmdb_id=None: f"Context for {name} ({hmdb_id})"
            )

            result = batch_get_metabolite_contexts(metabolites, mapping)

            assert len(result) == 2
            assert result["glucose"] == "Context for glucose (HMDB0000122)"
            assert result["pyruvate"] == "Context for pyruvate (HMDB0000243)"

            # Verify that get_metabolite_context was called with correct hmdb_ids
            mock_get_context.assert_any_call("glucose", "HMDB0000122")
            mock_get_context.assert_any_call("pyruvate", "HMDB0000243")

    def test_batch_partial_mapping(self):
        metabolites = ["glucose", "unknown_metabolite"]
        mapping = {"glucose": "HMDB0000122"}

        with patch("apriomics.hmdb_utils.get_metabolite_context") as mock_get_context:
            mock_get_context.side_effect = (
                lambda name, hmdb_id=None: f"Context for {name} ({hmdb_id})"
            )

            result = batch_get_metabolite_contexts(metabolites, mapping)

            assert len(result) == 2
            assert result["glucose"] == "Context for glucose (HMDB0000122)"
            assert (
                result["unknown_metabolite"] == "Context for unknown_metabolite (None)"
            )


class TestExampleMetaboliteMappings:
    """Test the example metabolite mappings."""

    def test_example_mappings_exist(self):
        assert isinstance(EXAMPLE_METABOLITE_MAPPINGS, dict)
        assert len(EXAMPLE_METABOLITE_MAPPINGS) > 0

    def test_example_mappings_format(self):
        for name, hmdb_id in EXAMPLE_METABOLITE_MAPPINGS.items():
            assert isinstance(name, str)
            assert isinstance(hmdb_id, str)
            assert hmdb_id.startswith("HMDB")

    def test_glucose_mapping(self):
        assert "glucose" in EXAMPLE_METABOLITE_MAPPINGS
        assert EXAMPLE_METABOLITE_MAPPINGS["glucose"] == "HMDB0000122"


# Integration tests
class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_with_example_mapping(self):
        """Test the complete workflow using example mappings."""
        metabolite_name = "glucose"
        hmdb_id = EXAMPLE_METABOLITE_MAPPINGS.get(metabolite_name)

        # This will attempt real HMDB access, but should handle gracefully if it fails
        context = get_metabolite_context(metabolite_name, hmdb_id)

        assert isinstance(context, str)
        assert metabolite_name in context.lower()
        assert (
            "HMDB" in context
            or "not available" in context
            or "not accessible" in context
        )


# Fixtures for testing
@pytest.fixture
def sample_metabolite():
    """Fixture providing a sample HMDBMetabolite object."""
    return HMDBMetabolite(
        hmdb_id="HMDB0000122",
        name="Glucose",
        description="A simple sugar",
        chemical_formula="C6H12O6",
        smiles="C(C1C(C(C(C(O1)O)O)O)O)O",
        pathways=["Glycolysis", "Gluconeogenesis"],
        diseases=["Diabetes mellitus"],
        tissues=["Blood", "Liver"],
        biospecimens=["Urine"],
        concentrations=[
            {
                "biospecimen": "Blood",
                "value": "5.5",
                "units": "mM",
                "age": "Adult",
                "sex": "Both",
            }
        ],
    )


@pytest.fixture
def sample_xml():
    """Fixture providing sample HMDB XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <metabolite>
        <name>Test Metabolite</name>
        <description>A test metabolite description</description>
        <chemical_formula>C6H12O6</chemical_formula>
        <smiles>TestSMILES</smiles>
        <biological_properties>
            <pathways>
                <pathway><name>Test Pathway</name></pathway>
            </pathways>
        </biological_properties>
        <diseases>
            <disease><name>Test Disease</name></disease>
        </diseases>
        <tissue_locations>
            <tissue><name>Test Tissue</name></tissue>
        </tissue_locations>
        <biospecimen_locations>
            <biospecimen><name>Test Biospecimen</name></biospecimen>
        </biospecimen_locations>
        <normal_concentrations>
            <concentration>
                <biospecimen>Test Sample</biospecimen>
                <concentration_value>1.0</concentration_value>
                <concentration_units>mM</concentration_units>
                <subject_age>Adult</subject_age>
                <subject_sex>Both</subject_sex>
            </concentration>
        </normal_concentrations>
    </metabolite>"""


class TestBuildHMDBGraph:
    """Test the HMDB graph building functionality."""

    @patch("apriomics.hmdb_utils.get_hmdb_metabolite_data")
    def test_build_graph_with_mock_data(self, mock_get_data):
        # Define mock data
        mock_metabolites = {
            "HMDB001": HMDBMetabolite(
                hmdb_id="HMDB001", name="A", reaction_partners=["HMDB002"]
            ),
            "HMDB002": HMDBMetabolite(
                hmdb_id="HMDB002", name="B", reaction_partners=["HMDB001", "HMDB003"]
            ),
            "HMDB003": HMDBMetabolite(
                hmdb_id="HMDB003", name="C", reaction_partners=["HMDB002"]
            ),
            "HMDB004": HMDBMetabolite(
                hmdb_id="HMDB004", name="D", reaction_partners=[]
            ),
            "HMDB005": HMDBMetabolite(
                hmdb_id="HMDB005", name="E", reaction_partners=["HMDB006"]
            ),  # Partner not in list
        }

        def side_effect(hmdb_id):
            return mock_metabolites.get(hmdb_id)

        mock_get_data.side_effect = side_effect

        hmdb_ids_to_test = ["HMDB001", "HMDB002", "HMDB003", "HMDB004", "HMDB005"]

        result = build_hmdb_graph(hmdb_ids_to_test)

        # Expected edges are canonical (sorted) tuples
        expected_edges = {
            ("HMDB001", "HMDB002"),
            ("HMDB002", "HMDB003"),
        }

        # Convert result to a set of tuples for easy comparison
        result_set = set(map(tuple, result))

        assert result_set == expected_edges

    def test_build_graph_empty_list(self):
        result = build_hmdb_graph([])
        assert result == []

    @patch("apriomics.hmdb_utils.get_hmdb_metabolite_data")
    def test_build_graph_no_connections(self, mock_get_data):
        mock_metabolites = {
            "HMDB001": HMDBMetabolite(
                hmdb_id="HMDB001", name="A", reaction_partners=[]
            ),
            "HMDB002": HMDBMetabolite(
                hmdb_id="HMDB002", name="B", reaction_partners=[]
            ),
        }

        def side_effect(hmdb_id):
            return mock_metabolites.get(hmdb_id)

        mock_get_data.side_effect = side_effect

        result = build_hmdb_graph(["HMDB001", "HMDB002"])
        assert result == []
