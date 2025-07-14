"""
Focused HMDB parser optimized for LLM prior generation.

Only extracts biologically relevant information that helps LLMs
understand metabolite relevance for differential analysis.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path
import sys

from .hmdb_parser import MetaboliteChunk


class FocusedHMDBParser:
    """
    HMDB parser focused on biological relevance for LLM priors.

    Extracts only:
    - Basic identity (name, description)
    - Biological pathways
    - Disease associations
    - Tissue locations (simplified)

    Skips:
    - Chemical structure details (SMILES, InChI, formula)
    - Synonyms/alternative names
    - Detailed concentration data
    - Chemical taxonomy
    - Spectral data
    """

    def __init__(self, xml_path: Path):
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"HMDB XML file not found: {xml_path}")

        self.namespace = "{http://www.hmdb.ca}"

    def parse_metabolites(
        self, max_metabolites: Optional[int] = None
    ) -> Iterator[MetaboliteChunk]:
        """
        Stream parse metabolites focusing on biological relevance.

        Yields ~50% fewer chunks with more focused content.
        """
        print(f"Parsing HMDB with biological focus: {self.xml_path}", file=sys.stderr)

        context = ET.iterparse(self.xml_path, events=("start", "end"))
        context = iter(context)

        event, root = next(context)

        metabolite_count = 0
        processed_count = 0

        for event, elem in context:
            if event == "end" and elem.tag == f"{self.namespace}metabolite":
                try:
                    chunks = self._process_metabolite_focused(elem)
                    for chunk in chunks:
                        yield chunk
                        processed_count += 1

                    metabolite_count += 1

                    if metabolite_count % 1000 == 0:
                        print(
                            f"Processed {metabolite_count} metabolites, {processed_count} focused chunks",
                            file=sys.stderr,
                        )

                    elem.clear()
                    root.clear()

                    if max_metabolites and metabolite_count >= max_metabolites:
                        break

                except Exception as e:
                    print(
                        f"Error processing metabolite {metabolite_count}: {e}",
                        file=sys.stderr,
                    )
                    continue

        print(
            f"Focused parsing complete: {metabolite_count} metabolites, {processed_count} chunks",
            file=sys.stderr,
        )

    def _process_metabolite_focused(self, metabolite_elem) -> List[MetaboliteChunk]:
        """Process metabolite with focus on biological relevance."""
        chunks = []

        # Extract basic information
        hmdb_id = self._get_text(metabolite_elem, "accession", "")
        if not hmdb_id:
            return chunks

        name = self._get_text(metabolite_elem, "name", "")
        description = self._get_text(metabolite_elem, "description", "")

        # Focus on biological description, not chemical structure
        basic_content = f"Metabolite: {name} (HMDB ID: {hmdb_id})"
        if description:
            # Extract first sentence or 200 chars of description
            desc_short = description.split(".")[0][:200] if description else ""
            basic_content += f"\nBiological role: {desc_short}"

        chunks.append(
            MetaboliteChunk(
                hmdb_id=hmdb_id,
                chunk_type="identity",
                content=basic_content.strip(),
                metadata={"name": name},
            )
        )

        # Biological pathways (high value for LLM priors)
        pathways = self._extract_pathways(metabolite_elem)
        if pathways:
            # Focus on major pathways, limit to most relevant
            major_pathways = [p for p in pathways if self._is_major_pathway(p)][:10]

            if major_pathways:
                pathway_content = f"{name} is involved in these metabolic pathways: {'; '.join(major_pathways)}"

                chunks.append(
                    MetaboliteChunk(
                        hmdb_id=hmdb_id,
                        chunk_type="pathways",
                        content=pathway_content,
                        metadata={"name": name, "pathways": major_pathways},
                    )
                )

        # Disease associations (high value for LLM priors)
        diseases = self._extract_diseases(metabolite_elem)
        if diseases:
            # Focus on major diseases, remove very generic ones
            major_diseases = [d for d in diseases if self._is_major_disease(d)][:8]

            if major_diseases:
                disease_content = (
                    f"{name} is associated with: {'; '.join(major_diseases)}"
                )

                chunks.append(
                    MetaboliteChunk(
                        hmdb_id=hmdb_id,
                        chunk_type="diseases",
                        content=disease_content,
                        metadata={"name": name, "diseases": major_diseases},
                    )
                )

        # Simplified tissue/biospecimen info (moderate value)
        tissues = self._extract_tissues(metabolite_elem)
        biospecimens = self._extract_biospecimens(metabolite_elem)

        # Combine and simplify location information
        locations = []
        if tissues:
            major_tissues = [t for t in tissues if self._is_major_tissue(t)][:5]
            locations.extend(major_tissues)
        if biospecimens:
            major_biospecimens = [
                b for b in biospecimens if self._is_major_biospecimen(b)
            ][:3]
            locations.extend(major_biospecimens)

        if locations:
            location_content = f"{name} is found in: {'; '.join(set(locations))}"

            chunks.append(
                MetaboliteChunk(
                    hmdb_id=hmdb_id,
                    chunk_type="locations",
                    content=location_content,
                    metadata={"name": name, "locations": locations},
                )
            )

        # Functions chunk (focused on biological relevance)
        functions = self._extract_functions(metabolite_elem)
        if functions:
            # Filter for most relevant functions
            relevant_functions = [f for f in functions if self._is_relevant_function(f)][:5]
            
            if relevant_functions:
                func_content = f"{name} biological functions: {'; '.join(relevant_functions)}"
                
                chunks.append(MetaboliteChunk(
                    hmdb_id=hmdb_id,
                    chunk_type='functions',
                    content=func_content,
                    metadata={'name': name, 'functions': relevant_functions}
                ))
        
        # Biochemical class chunk (high value for LLM priors)
        biochemical_class = self._extract_biochemical_class(metabolite_elem)
        if biochemical_class and (biochemical_class.get('class') or biochemical_class.get('subclass')):
            class_info = []
            if biochemical_class.get('class'):
                class_info.append(f"Class: {biochemical_class['class']}")
            if biochemical_class.get('subclass'):
                class_info.append(f"Subclass: {biochemical_class['subclass']}")
            if biochemical_class.get('super_class'):
                class_info.append(f"Super class: {biochemical_class['super_class']}")
            
            if class_info:
                class_content = f"{name} biochemical classification: {'; '.join(class_info)}"
                
                chunks.append(MetaboliteChunk(
                    hmdb_id=hmdb_id,
                    chunk_type='biochemical_class',
                    content=class_content,
                    metadata={'name': name, 'biochemical_class': biochemical_class}
                ))
        
        # Protein interactions chunk (filtered for relevance)
        protein_interactions = self._extract_protein_interactions(metabolite_elem)
        if protein_interactions:
            # Focus on enzymes and transporters
            relevant_proteins = [p for p in protein_interactions if self._is_relevant_protein(p)][:8]
            
            if relevant_proteins:
                protein_content = f"{name} protein interactions: {'; '.join(relevant_proteins)}"
                
                chunks.append(MetaboliteChunk(
                    hmdb_id=hmdb_id,
                    chunk_type='protein_interactions',
                    content=protein_content,
                    metadata={'name': name, 'protein_interactions': relevant_proteins}
                ))

        return chunks

    def _is_major_pathway(self, pathway: str) -> bool:
        """Filter for biologically significant pathways."""
        pathway_lower = pathway.lower()

        # Skip very generic or chemical-focused pathways
        skip_terms = [
            "drug",
            "chemical",
            "food component",
            "vitamin",
            "cofactor",
            "gamma-glutamyl",
            "dipeptide",
            "general",
            "other",
        ]

        if any(term in pathway_lower for term in skip_terms):
            return False

        # Keep metabolic and disease pathways
        keep_terms = [
            "metabolism",
            "metabolic",
            "synthesis",
            "degradation",
            "cycle",
            "pathway",
            "disease",
            "disorder",
            "deficiency",
        ]

        return any(term in pathway_lower for term in keep_terms) or len(pathway) > 10

    def _is_major_disease(self, disease: str) -> bool:
        """Filter for clinically relevant diseases."""
        disease_lower = disease.lower()

        # Skip very generic terms
        skip_terms = [
            "administration",
            "treatment",
            "therapy",
            "supplementation",
            "general",
            "other",
            "unknown",
        ]

        if any(term in disease_lower for term in skip_terms):
            return False

        # Keep major disease categories
        return len(disease) > 5  # Filter out very short/generic terms

    def _is_major_tissue(self, tissue: str) -> bool:
        """Filter for major tissue types."""
        tissue_lower = tissue.lower()

        major_tissues = [
            "blood",
            "plasma",
            "serum",
            "liver",
            "kidney",
            "brain",
            "heart",
            "muscle",
            "lung",
            "intestine",
            "colon",
            "skin",
            "adipose",
            "bone",
            "pancreas",
        ]

        return any(major in tissue_lower for major in major_tissues)

    def _is_major_biospecimen(self, biospecimen: str) -> bool:
        """Filter for clinically relevant biospecimens."""
        biospecimen_lower = biospecimen.lower()

        major_biospecimens = [
            "blood",
            "plasma",
            "serum",
            "urine",
            "saliva",
            "cerebrospinal fluid",
            "tissue",
            "feces",
            "breath",
        ]

        return any(major in biospecimen_lower for major in major_biospecimens)
    
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
    
    def _is_relevant_function(self, function: str) -> bool:
        """Filter for biologically relevant functions."""
        function_lower = function.lower()
        
        # Skip very generic terms
        skip_terms = ['unknown', 'not available', 'general', 'other']
        if any(term in function_lower for term in skip_terms):
            return False
        
        # Keep specific biological functions
        keep_terms = [
            'enzyme', 'cofactor', 'substrate', 'inhibitor', 'activator',
            'signaling', 'transport', 'metabolism', 'synthesis', 'degradation',
            'regulation', 'binding', 'catalysis', 'membrane', 'cellular'
        ]
        
        return any(term in function_lower for term in keep_terms) or len(function) > 20
    
    def _is_relevant_protein(self, protein: str) -> bool:
        """Filter for relevant protein interactions."""
        protein_lower = protein.lower()
        
        # Focus on enzymes and transporters
        relevant_types = [
            'enzyme', 'transporter', 'carrier', 'channel', 'receptor',
            'kinase', 'phosphatase', 'dehydrogenase', 'synthase', 'hydrolase',
            'transferase', 'lyase', 'isomerase', 'ligase'
        ]
        
        return any(ptype in protein_lower for ptype in relevant_types)

    def _get_text(self, element, tag: str, default: str = "") -> str:
        """Safely extract text from XML element."""
        if element is None:
            return default
        if not tag.startswith("{"):
            tag = f"{self.namespace}{tag}"
        found = element.find(tag)
        if found is not None and found.text:
            return found.text.strip()
        return default

    def _extract_pathways(self, metabolite_elem) -> List[str]:
        """Extract pathway information."""
        pathways = []
        pathways_elem = metabolite_elem.find(
            f"{self.namespace}biological_properties/{self.namespace}pathways"
        )
        if pathways_elem is not None:
            for pathway in pathways_elem.findall(f"{self.namespace}pathway"):
                pathway_name = self._get_text(pathway, "name", "")
                if pathway_name:
                    pathways.append(pathway_name)
        return pathways

    def _extract_diseases(self, metabolite_elem) -> List[str]:
        """Extract disease associations."""
        diseases = []
        diseases_elem = metabolite_elem.find(f"{self.namespace}diseases")
        if diseases_elem is not None:
            for disease in diseases_elem.findall(f"{self.namespace}disease"):
                disease_name = self._get_text(disease, "name", "")
                if disease_name:
                    diseases.append(disease_name)
        return diseases

    def _extract_tissues(self, metabolite_elem) -> List[str]:
        """Extract tissue locations."""
        tissues = []
        tissues_elem = metabolite_elem.find(f"{self.namespace}tissue_locations")
        if tissues_elem is not None:
            for tissue in tissues_elem.findall(f"{self.namespace}tissue"):
                tissue_name = self._get_text(tissue, "name", "")
                if tissue_name:
                    tissues.append(tissue_name)
        return tissues

    def _extract_biospecimens(self, metabolite_elem) -> List[str]:
        """Extract biospecimen locations."""
        biospecimens = []
        biospecimens_elem = metabolite_elem.find(
            f"{self.namespace}biospecimen_locations"
        )
        if biospecimens_elem is not None:
            for biospecimen in biospecimens_elem.findall(
                f"{self.namespace}biospecimen"
            ):
                biospecimen_name = self._get_text(biospecimen, "name", "")
                if biospecimen_name:
                    biospecimens.append(biospecimen_name)
        return biospecimens
