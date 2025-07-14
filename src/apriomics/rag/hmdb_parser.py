"""
HMDB XML parser for efficient processing of large metabolite databases.

This module provides streaming XML parsing to handle the 6.49GB HMDB dump
without loading everything into memory at once.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path
import sys


@dataclass
class MetaboliteChunk:
    """A chunk of metabolite information optimized for embedding and retrieval."""
    
    hmdb_id: str
    chunk_type: str  # 'basic', 'pathways', 'diseases', 'tissues', 'concentrations', 'functions', 'biochemical_class', 'protein_interactions', 'cellular_processes'
    content: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure content is suitable for embedding."""
        if not self.content or not self.content.strip():
            raise ValueError(f"Empty content for chunk {self.hmdb_id}:{self.chunk_type}")


class HMDBParser:
    """
    Streaming parser for HMDB XML dumps.
    
    Efficiently processes large XML files by yielding metabolite chunks
    suitable for vector embedding and retrieval.
    """
    
    def __init__(self, xml_path: Path):
        """
        Initialize parser with XML file path.
        
        Args:
            xml_path: Path to HMDB XML dump file
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"HMDB XML file not found: {xml_path}")
        
        # HMDB XML namespace
        self.namespace = "{http://www.hmdb.ca}"
    
    def parse_metabolites(self, max_metabolites: Optional[int] = None) -> Iterator[MetaboliteChunk]:
        """
        Stream parse metabolites from XML and yield chunks.
        
        Args:
            max_metabolites: Optional limit for testing/development
            
        Yields:
            MetaboliteChunk: Processed metabolite information chunks
        """
        print(f"Starting to parse HMDB XML: {self.xml_path}", file=sys.stderr)
        
        # Use iterparse for memory-efficient streaming
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        
        # Get root element
        event, root = next(context)
        
        metabolite_count = 0
        processed_count = 0
        
        for event, elem in context:
            if event == 'end' and elem.tag == f'{self.namespace}metabolite':
                try:
                    # Process this metabolite
                    chunks = self._process_metabolite(elem)
                    for chunk in chunks:
                        yield chunk
                        processed_count += 1
                    
                    metabolite_count += 1
                    
                    if metabolite_count % 1000 == 0:
                        print(f"Processed {metabolite_count} metabolites, {processed_count} chunks", file=sys.stderr)
                    
                    # Clear element to free memory
                    elem.clear()
                    root.clear()
                    
                    # Check limit
                    if max_metabolites and metabolite_count >= max_metabolites:
                        break
                        
                except Exception as e:
                    print(f"Error processing metabolite {metabolite_count}: {e}", file=sys.stderr)
                    continue
        
        print(f"Finished parsing. Total: {metabolite_count} metabolites, {processed_count} chunks", file=sys.stderr)
    
    def _process_metabolite(self, metabolite_elem) -> List[MetaboliteChunk]:
        """
        Process a single metabolite element into multiple chunks.
        
        Args:
            metabolite_elem: XML element for a metabolite
            
        Returns:
            List of MetaboliteChunk objects
        """
        chunks = []
        
        # Extract basic information
        hmdb_id = self._get_text(metabolite_elem, 'accession', '')
        if not hmdb_id:
            return chunks  # Skip metabolites without ID
        
        name = self._get_text(metabolite_elem, 'name', '')
        description = self._get_text(metabolite_elem, 'description', '')
        chemical_formula = self._get_text(metabolite_elem, 'chemical_formula', '')
        smiles = self._get_text(metabolite_elem, 'smiles', '')
        
        # Basic information chunk
        basic_content = f"""Metabolite: {name} (HMDB ID: {hmdb_id})
Chemical Formula: {chemical_formula}
SMILES: {smiles}
Description: {description[:500]}...""".strip()
        
        if basic_content:
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='basic',
                content=basic_content,
                metadata={
                    'name': name,
                    'formula': chemical_formula,
                    'smiles': smiles
                }
            ))
        
        # Pathways chunk
        pathways = self._extract_pathways(metabolite_elem)
        if pathways:
            pathway_content = f"""Metabolite {name} ({hmdb_id}) is involved in the following metabolic pathways:
{'; '.join(pathways)}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='pathways',
                content=pathway_content,
                metadata={
                    'name': name,
                    'pathways': pathways
                }
            ))
        
        # Diseases chunk
        diseases = self._extract_diseases(metabolite_elem)
        if diseases:
            disease_content = f"""Metabolite {name} ({hmdb_id}) is associated with the following diseases and conditions:
{'; '.join(diseases)}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='diseases',
                content=disease_content,
                metadata={
                    'name': name,
                    'diseases': diseases
                }
            ))
        
        # Tissues/biospecimens chunk
        tissues = self._extract_tissues(metabolite_elem)
        biospecimens = self._extract_biospecimens(metabolite_elem)
        if tissues or biospecimens:
            location_content = f"""Metabolite {name} ({hmdb_id}) is found in:
Tissues: {'; '.join(tissues) if tissues else 'Not specified'}
Biospecimens: {'; '.join(biospecimens) if biospecimens else 'Not specified'}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='tissues',
                content=location_content,
                metadata={
                    'name': name,
                    'tissues': tissues,
                    'biospecimens': biospecimens
                }
            ))
        
        # Concentrations chunk
        concentrations = self._extract_concentrations(metabolite_elem)
        if concentrations:
            conc_examples = []
            for conc in concentrations[:5]:  # Limit to top 5 examples
                if conc.get('biospecimen') and conc.get('value') and conc.get('units'):
                    conc_examples.append(f"{conc['biospecimen']}: {conc['value']} {conc['units']}")
            
            if conc_examples:
                conc_content = f"""Normal concentrations of {name} ({hmdb_id}):
{'; '.join(conc_examples)}""".strip()
                
                chunks.append(MetaboliteChunk(
                    hmdb_id=hmdb_id,
                    chunk_type='concentrations',
                    content=conc_content,
                    metadata={
                        'name': name,
                        'concentrations': concentrations
                    }
                ))
        
        # Functions chunk (biological functions and roles)
        functions = self._extract_functions(metabolite_elem)
        if functions:
            func_content = f"""Biological functions of {name} ({hmdb_id}):
{'; '.join(functions)}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='functions',
                content=func_content,
                metadata={
                    'name': name,
                    'functions': functions
                }
            ))
        
        # Biochemical class chunk
        biochemical_class = self._extract_biochemical_class(metabolite_elem)
        if biochemical_class:
            class_content = f"""Biochemical classification of {name} ({hmdb_id}):
Class: {biochemical_class.get('class', 'Not specified')}
Subclass: {biochemical_class.get('subclass', 'Not specified')}
Molecular framework: {biochemical_class.get('molecular_framework', 'Not specified')}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='biochemical_class',
                content=class_content,
                metadata={
                    'name': name,
                    'biochemical_class': biochemical_class
                }
            ))
        
        # Protein interactions chunk
        protein_interactions = self._extract_protein_interactions(metabolite_elem)
        if protein_interactions:
            protein_content = f"""Protein interactions of {name} ({hmdb_id}):
{'; '.join(protein_interactions)}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='protein_interactions',
                content=protein_content,
                metadata={
                    'name': name,
                    'protein_interactions': protein_interactions
                }
            ))
        
        # Cellular processes chunk
        cellular_processes = self._extract_cellular_processes(metabolite_elem)
        if cellular_processes:
            cellular_content = f"""Cellular processes involving {name} ({hmdb_id}):
{'; '.join(cellular_processes)}""".strip()
            
            chunks.append(MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type='cellular_processes',
                content=cellular_content,
                metadata={
                    'name': name,
                    'cellular_processes': cellular_processes
                }
            ))
        
        return chunks
    
    def _get_text(self, element, tag: str, default: str = '') -> str:
        """Safely extract text from XML element."""
        if element is None:
            return default
        # Add namespace to tag if not already present
        if not tag.startswith('{'):
            tag = f'{self.namespace}{tag}'
        found = element.find(tag)
        if found is not None and found.text:
            return found.text.strip()
        return default
    
    def _extract_pathways(self, metabolite_elem) -> List[str]:
        """Extract pathway information."""
        pathways = []
        pathways_elem = metabolite_elem.find(f'{self.namespace}biological_properties/{self.namespace}pathways')
        if pathways_elem is not None:
            for pathway in pathways_elem.findall(f'{self.namespace}pathway'):
                pathway_name = self._get_text(pathway, 'name', '')
                if pathway_name:
                    pathways.append(pathway_name)
        return pathways
    
    def _extract_diseases(self, metabolite_elem) -> List[str]:
        """Extract disease associations."""
        diseases = []
        diseases_elem = metabolite_elem.find(f'{self.namespace}diseases')
        if diseases_elem is not None:
            for disease in diseases_elem.findall(f'{self.namespace}disease'):
                disease_name = self._get_text(disease, 'name', '')
                if disease_name:
                    diseases.append(disease_name)
        return diseases
    
    def _extract_tissues(self, metabolite_elem) -> List[str]:
        """Extract tissue locations."""
        tissues = []
        tissues_elem = metabolite_elem.find(f'{self.namespace}tissue_locations')
        if tissues_elem is not None:
            for tissue in tissues_elem.findall(f'{self.namespace}tissue'):
                tissue_name = self._get_text(tissue, 'name', '')
                if tissue_name:
                    tissues.append(tissue_name)
        return tissues
    
    def _extract_biospecimens(self, metabolite_elem) -> List[str]:
        """Extract biospecimen locations."""
        biospecimens = []
        biospecimens_elem = metabolite_elem.find(f'{self.namespace}biospecimen_locations')
        if biospecimens_elem is not None:
            for biospecimen in biospecimens_elem.findall(f'{self.namespace}biospecimen'):
                biospecimen_name = self._get_text(biospecimen, 'name', '')
                if biospecimen_name:
                    biospecimens.append(biospecimen_name)
        return biospecimens
    
    def _extract_concentrations(self, metabolite_elem) -> List[Dict[str, str]]:
        """Extract concentration data."""
        concentrations = []
        concentrations_elem = metabolite_elem.find(f'{self.namespace}normal_concentrations')
        if concentrations_elem is not None:
            for conc in concentrations_elem.findall(f'{self.namespace}concentration'):
                conc_data = {
                    'biospecimen': self._get_text(conc, 'biospecimen', ''),
                    'value': self._get_text(conc, 'concentration_value', ''),
                    'units': self._get_text(conc, 'concentration_units', ''),
                    'age': self._get_text(conc, 'subject_age', ''),
                    'sex': self._get_text(conc, 'subject_sex', '')
                }
                if conc_data['biospecimen'] and conc_data['value']:
                    concentrations.append(conc_data)
        return concentrations
    
    def _extract_functions(self, metabolite_elem) -> List[str]:
        """Extract biological functions and roles."""
        functions = []
        
        # Extract from biological_properties/general_function
        general_function = self._get_text(metabolite_elem, 'biological_properties/general_function', '')
        if general_function:
            functions.append(f"General function: {general_function}")
        
        # Extract from biological_properties/specific_function
        specific_function = self._get_text(metabolite_elem, 'biological_properties/specific_function', '')
        if specific_function:
            functions.append(f"Specific function: {specific_function}")
        
        # Extract from biological_properties/cellular_locations
        cellular_locations = metabolite_elem.find(f'{self.namespace}biological_properties/{self.namespace}cellular_locations')
        if cellular_locations is not None:
            for location in cellular_locations.findall(f'{self.namespace}cellular_location'):
                location_name = self._get_text(location, '', '')
                if location_name:
                    functions.append(f"Cellular location: {location_name}")
        
        return functions
    
    def _extract_biochemical_class(self, metabolite_elem) -> Dict[str, str]:
        """Extract biochemical classification data."""
        taxonomy_elem = metabolite_elem.find(f'{self.namespace}taxonomy')
        if taxonomy_elem is None:
            return {}
        
        classification = {
            'description': self._get_text(taxonomy_elem, 'description', ''),
            'direct_parent': self._get_text(taxonomy_elem, 'direct_parent', ''),
            'kingdom': self._get_text(taxonomy_elem, 'kingdom', ''),
            'super_class': self._get_text(taxonomy_elem, 'super_class', ''),
            'class': self._get_text(taxonomy_elem, 'class', ''),
            'subclass': self._get_text(taxonomy_elem, 'sub_class', ''),
            'molecular_framework': self._get_text(taxonomy_elem, 'molecular_framework', ''),
            'alternative_parents': []
        }
        
        # Extract alternative parents
        alt_parents_elem = taxonomy_elem.find(f'{self.namespace}alternative_parents')
        if alt_parents_elem is not None:
            for parent in alt_parents_elem.findall(f'{self.namespace}alternative_parent'):
                parent_name = self._get_text(parent, '', '')
                if parent_name:
                    classification['alternative_parents'].append(parent_name)
        
        return classification
    
    def _extract_protein_interactions(self, metabolite_elem) -> List[str]:
        """Extract protein associations (enzymes, transporters, binding proteins)."""
        proteins = []
        
        # Extract protein associations
        protein_associations = metabolite_elem.find(f'{self.namespace}protein_associations')
        if protein_associations is not None:
            for protein in protein_associations.findall(f'{self.namespace}protein'):
                protein_name = self._get_text(protein, 'name', '')
                protein_type = self._get_text(protein, 'protein_type', '')
                if protein_name and protein_type:
                    proteins.append(f"{protein_type}: {protein_name}")
                elif protein_name:
                    proteins.append(protein_name)
        
        return proteins
    
    def _extract_cellular_processes(self, metabolite_elem) -> List[str]:
        """Extract cellular processes and molecular functions."""
        processes = []
        
        # Extract from ontology
        ontology_elem = metabolite_elem.find(f'{self.namespace}ontology')
        if ontology_elem is not None:
            # Molecular functions
            for status in ontology_elem.findall(f'{self.namespace}status'):
                if status.text and 'molecular_function' in status.text.lower():
                    processes.append(f"Molecular function: {status.text}")
            
            # Biological processes
            for status in ontology_elem.findall(f'{self.namespace}status'):
                if status.text and 'biological_process' in status.text.lower():
                    processes.append(f"Biological process: {status.text}")
            
            # Cellular components
            for status in ontology_elem.findall(f'{self.namespace}status'):
                if status.text and 'cellular_component' in status.text.lower():
                    processes.append(f"Cellular component: {status.text}")
        
        return processes


def test_parser(xml_path: Path, max_metabolites: int = 10):
    """Test function to validate parser with small sample."""
    parser = HMDBParser(xml_path)
    
    print(f"Testing parser with max {max_metabolites} metabolites...")
    
    chunk_count = 0
    for chunk in parser.parse_metabolites(max_metabolites=max_metabolites):
        print(f"\nChunk {chunk_count}:")
        print(f"  HMDB ID: {chunk.hmdb_id}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Content: {chunk.content[:200]}...")
        print(f"  Metadata keys: {list(chunk.metadata.keys())}")
        
        chunk_count += 1
        
        if chunk_count >= 20:  # Limit output
            break
    
    print(f"\nProcessed {chunk_count} chunks from {max_metabolites} metabolites")


if __name__ == "__main__":
    # Test with small sample if run directly
    import sys
    if len(sys.argv) > 1:
        test_parser(Path(sys.argv[1]), max_metabolites=5)