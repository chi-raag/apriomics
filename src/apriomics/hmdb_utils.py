"""
Utilities for interacting with the HMDB (Human Metabolome Database).
"""

import requests
import sys
from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import xml.etree.ElementTree as ET
import json
from urllib.parse import urlparse # Added import

HMDB_SEARCH_URL = "https://hmdb.ca/unearth/q"
HMDB_METABOLITE_URL = "https://hmdb.ca/metabolites/{}.xml"

@dataclass
class HMDBMetabolite:
    """Data structure for HMDB metabolite information."""
    hmdb_id: str
    name: str
    description: Optional[str] = None
    chemical_formula: Optional[str] = None
    taxonomy: Optional[str] = None
    ontology: Optional[str] = None 
    smiles: Optional[str] = None
    pathways: Optional[List[str]] = None
    reaction_partners: Optional[List[str]] = None
    diseases: Optional[List[str]] = None
    tissues: Optional[List[str]] = None
    biospecimens: Optional[List[str]] = None
    concentrations: Optional[List[Dict[str, str]]] = None
    
    def __post_init__(self):
        if self.pathways is None:
            self.pathways = []
        if self.reaction_partners is None:
            self.reaction_partners = []
        if self.diseases is None:
            self.diseases = []
        if self.tissues is None:
            self.tissues = []
        if self.biospecimens is None:
            self.biospecimens = []
        if self.concentrations is None:
            self.concentrations = []

def search_hmdb_metabolite(metabolite_name: str) -> Optional[str]:
    """
    Search for a metabolite in HMDB by name and return the HMDB ID.
    
    Args:
        metabolite_name: The name of the metabolite to search for.
        
    Returns:
        The HMDB ID (e.g., "HMDB0000001") or None if not found.
    """
    if not metabolite_name:
        print("Warning: Empty metabolite name provided.", file=sys.stderr)
        return None
        
    try:
        params = {'query': metabolite_name}
        # Make a GET request to the HMDB search URL
        # HMDB's search for an exact name often redirects to the metabolite page
        response = requests.get(HMDB_SEARCH_URL, params=params, timeout=15, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # The final URL after redirects
        final_url = response.url
        
        # Parse the final URL to see if it's a direct metabolite page
        parsed_url = urlparse(final_url)
        path_parts = parsed_url.path.split('/')
        
        # Check if the path looks like /metabolites/HMDBXXXXXXX
        if len(path_parts) > 2 and path_parts[-2] == 'metabolites' and path_parts[-1].startswith('HMDB'):
            hmdb_id = path_parts[-1]
            print(f"Found HMDB ID: {hmdb_id} for '{metabolite_name}' from URL: {final_url}", file=sys.stderr)
            return hmdb_id
        else:
            # This means the search didn't redirect to a specific metabolite page
            # It might be a search results page or something else.
            print(f"Could not directly find HMDB ID for '{metabolite_name}'. Final URL: {final_url}", file=sys.stderr)
            return None
            
    except requests.exceptions.Timeout:
        print(f"Error: Timeout occurred while searching HMDB for '{metabolite_name}'.", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error searching HMDB for '{metabolite_name}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while searching HMDB for '{metabolite_name}': {e}", file=sys.stderr)
        return None

def get_hmdb_metabolite_data(hmdb_id: str) -> Optional[HMDBMetabolite]:
    """
    Retrieve detailed metabolite data from HMDB by ID.
    
    Args:
        hmdb_id: The HMDB ID (e.g., "HMDB0000001").
        
    Returns:
        HMDBMetabolite object with detailed information or None if not found.
    """
    if not hmdb_id or not hmdb_id.startswith('HMDB'):
        print(f"Warning: Invalid HMDB ID provided: '{hmdb_id}'. Expected format like 'HMDB0000001'.", file=sys.stderr)
        return None
        
    url = HMDB_METABOLITE_URL.format(hmdb_id)
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Extract basic information
        name = _get_xml_text(root, 'name', '')
        description = _get_xml_text(root, 'description', '')
        chemical_formula = _get_xml_text(root, 'chemical_formula', '')
        taxonomy = _get_xml_text(root, 'taxonomy', '')
        smiles = _get_xml_text(root, 'smiles', '')
        ontology = _get_xml_text(root, 'ontology', '')
        # Extract pathways and reaction partners
        pathways = []
        reaction_partners = set()  # Use a set to avoid duplicates
        pathways_elem = root.find('biological_properties/pathways')
        if pathways_elem is not None:
            for pathway in pathways_elem.findall('pathway'):
                pathway_name = _get_xml_text(pathway, 'name', '')
                if pathway_name:
                    pathways.append(pathway_name)
                
                # Extract reaction partners from the pathway
                metabolites_elem = pathway.find('metabolites')
                if metabolites_elem is not None:
                    for metabolite in metabolites_elem.findall('metabolite'):
                        partner_id = _get_xml_text(metabolite, 'hmdb_id', '')
                        if partner_id and partner_id != hmdb_id:
                            reaction_partners.add(partner_id)
        
        # Extract diseases
        diseases = []
        diseases_elem = root.find('diseases')
        if diseases_elem is not None:
            for disease in diseases_elem.findall('disease'):
                disease_name = _get_xml_text(disease, 'name', '')
                if disease_name:
                    diseases.append(disease_name)
        
        # Extract tissues
        tissues = []
        tissues_elem = root.find('tissue_locations')
        if tissues_elem is not None:
            for tissue in tissues_elem.findall('tissue'):
                tissue_name = _get_xml_text(tissue, 'name', '')
                if tissue_name:
                    tissues.append(tissue_name)
        
        # Extract biospecimens
        biospecimens = []
        biospecimens_elem = root.find('biospecimen_locations')
        if biospecimens_elem is not None:
            for biospecimen in biospecimens_elem.findall('biospecimen'):
                biospecimen_name = _get_xml_text(biospecimen, 'name', '')
                if biospecimen_name:
                    biospecimens.append(biospecimen_name)
        
        # Extract concentrations
        concentrations = []
        concentrations_elem = root.find('normal_concentrations')
        if concentrations_elem is not None:
            for conc in concentrations_elem.findall('concentration'):
                conc_data = {
                    'biospecimen': _get_xml_text(conc, 'biospecimen', ''),
                    'value': _get_xml_text(conc, 'concentration_value', ''),
                    'units': _get_xml_text(conc, 'concentration_units', ''),
                    'age': _get_xml_text(conc, 'subject_age', ''),
                    'sex': _get_xml_text(conc, 'subject_sex', '')
                }
                if conc_data['biospecimen'] and conc_data['value']:
                    concentrations.append(conc_data)
        
        return HMDBMetabolite(
            hmdb_id=hmdb_id,
            name=name,
            description=description,
            chemical_formula=chemical_formula,
            taxonomy=taxonomy,
            smiles=smiles,
            ontology=ontology,
            pathways=pathways,
            reaction_partners=list(reaction_partners),
            diseases=diseases,
            tissues=tissues,
            biospecimens=biospecimens,
            concentrations=concentrations
        )
        
    except requests.exceptions.Timeout:
        print(f"Error: Timeout occurred while fetching HMDB data for '{hmdb_id}'.", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HMDB data for '{hmdb_id}': {e}", file=sys.stderr)
        return None
    except ET.ParseError as e:
        print(f"Error parsing HMDB XML for '{hmdb_id}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing HMDB data for '{hmdb_id}': {e}", file=sys.stderr)
        return None

def _get_xml_text(element, tag: str, default: str = '') -> str:
    """Helper function to safely extract text from XML elements."""
    if element is None:
        return default
    found = element.find(tag)
    if found is not None and found.text:
        return found.text.strip()
    return default

def get_metabolite_context(metabolite_name: str, hmdb_id: Optional[str] = None) -> str:
    """
    Get formatted context string for a metabolite including HMDB information.
    
    Args:
        metabolite_name: Name of the metabolite.
        hmdb_id: Optional HMDB ID. If not provided, will attempt to search.
        
    Returns:
        Formatted context string suitable for LLM prompts.
    """
    if hmdb_id is None:
        hmdb_id = search_hmdb_metabolite(metabolite_name)
        
    if hmdb_id is None:
        return f"Metabolite: {metabolite_name} (HMDB data not available)"
    
    metabolite_data = get_hmdb_metabolite_data(hmdb_id)
    if metabolite_data is None:
        return f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id}, data not accessible)"
    
    context_parts = [
        f"Metabolite: {metabolite_data.name} (HMDB ID: {metabolite_data.hmdb_id})"
    ]
    
    if metabolite_data.description:
        context_parts.append(f"Description: {metabolite_data.description[:200]}...")
    
    if metabolite_data.chemical_formula:
        context_parts.append(f"Formula: {metabolite_data.chemical_formula}")
    
    if metabolite_data.pathways:
        pathways_str = ", ".join(metabolite_data.pathways[:5])  # Limit to first 5
        context_parts.append(f"Pathways: {pathways_str}")
    
    if metabolite_data.diseases:
        diseases_str = ", ".join(metabolite_data.diseases[:3])  # Limit to first 3
        context_parts.append(f"Associated diseases: {diseases_str}")
    
    if metabolite_data.tissues:
        tissues_str = ", ".join(metabolite_data.tissues[:5])  # Limit to first 5
        context_parts.append(f"Found in tissues: {tissues_str}")
    
    if metabolite_data.concentrations:
        # Show a few concentration examples
        conc_examples = []
        for conc in metabolite_data.concentrations[:2]:
            if conc['biospecimen'] and conc['value'] and conc['units']:
                conc_examples.append(f"{conc['biospecimen']}: {conc['value']} {conc['units']}")
        if conc_examples:
            context_parts.append(f"Normal concentrations: {'; '.join(conc_examples)}")
    
    return " | ".join(context_parts)

def batch_get_metabolite_contexts(metabolite_names: List[str], 
                                  hmdb_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Get context strings for multiple metabolites.
    
    Args:
        metabolite_names: List of metabolite names.
        hmdb_mapping: Optional dictionary mapping metabolite names to HMDB IDs.
        
    Returns:
        Dictionary mapping metabolite names to their context strings.
    """
    contexts = {}
    
    for name in metabolite_names:
        hmdb_id = hmdb_mapping.get(name) if hmdb_mapping else None
        contexts[name] = get_metabolite_context(name, hmdb_id)
        
    return contexts

# Example metabolite name to HMDB ID mappings (small subset for testing)
EXAMPLE_METABOLITE_MAPPINGS = {
    "glucose": "HMDB0000122",
    "pyruvate": "HMDB0000243",
    "lactate": "HMDB0000190",
    "alanine": "HMDB0000161",
    "glycine": "HMDB0000123",
    "serine": "HMDB0000187",
    "ATP": "HMDB0000538",
    "ADP": "HMDB0001341",
    "AMP": "HMDB0000045"
}

def build_hmdb_graph(hmdb_ids: List[str]) -> List[tuple[str, str]]:
    """
    Builds a metabolic network graph from a list of HMDB IDs.

    Args:
        hmdb_ids: A list of HMDB IDs to include in the graph.

    Returns:
        A list of tuples representing the edges of the graph, where each
        edge is a pair of HMDB IDs that are reaction partners.
        The edges are canonical (sorted) to ensure uniqueness.
    """
    edges = set()
    # Create a set for quick lookups of which metabolites are in our list
    metabolite_set = set(hmdb_ids)

    for hmdb_id in hmdb_ids:
        print(f"Fetching data for {hmdb_id}...", file=sys.stderr)
        metabolite_data = get_hmdb_metabolite_data(hmdb_id)

        if metabolite_data and metabolite_data.reaction_partners:
            for partner_id in metabolite_data.reaction_partners:
                # Only add edges between metabolites in the provided list
                if partner_id in metabolite_set:
                    # Create a canonical edge (sorted tuple) to avoid duplicates
                    # like (A, B) and (B, A), and self-loops
                    if hmdb_id != partner_id:
                        edge = tuple(sorted((hmdb_id, partner_id)))
                        edges.add(edge)
    
    return list(edges)
