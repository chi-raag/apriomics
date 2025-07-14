"""
Simple HMDB XML scraper for LLM context generation.

This module provides a lightweight alternative to the full RAG parser,
extracting only essential information needed for LLM context generation.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path
import sys


@dataclass
class MetaboliteContext:
    """Simple metabolite context for LLM input."""
    
    hmdb_id: str
    name: str
    description: str
    pathways: List[str]
    diseases: List[str]
    biochemical_class: str
    
    def to_context_string(self) -> str:
        """Convert to formatted string for LLM context."""
        context_parts = [
            f"Metabolite: {self.name} (HMDB ID: {self.hmdb_id})"
        ]
        
        if self.description:
            context_parts.append(f"Description: {self.description}")
        
        if self.biochemical_class:
            context_parts.append(f"Biochemical class: {self.biochemical_class}")
        
        if self.pathways:
            context_parts.append(f"Pathways: {'; '.join(self.pathways[:10])}")  # Limit to 10
        
        if self.diseases:
            context_parts.append(f"Disease associations: {'; '.join(self.diseases[:8])}")  # Limit to 8
        
        return "\n".join(context_parts)


class SimpleHMDBScraper:
    """
    Lightweight HMDB XML scraper for LLM context generation.
    
    Extracts only essential information: name, description, pathways, 
    diseases, and biochemical class.
    """
    
    def __init__(self, xml_path: Path):
        """Initialize scraper with XML file path."""
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"HMDB XML file not found: {xml_path}")
        
        self.namespace = "{http://www.hmdb.ca}"
    
    def extract_metabolite_contexts(self, max_metabolites: Optional[int] = None) -> Iterator[MetaboliteContext]:
        """
        Extract metabolite contexts from XML.
        
        Args:
            max_metabolites: Optional limit for testing
            
        Yields:
            MetaboliteContext: Simple context objects for LLM input
        """
        print(f"Extracting metabolite contexts from: {self.xml_path}", file=sys.stderr)
        
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        
        event, root = next(context)
        
        metabolite_count = 0
        
        for event, elem in context:
            if event == 'end' and elem.tag == f'{self.namespace}metabolite':
                try:
                    metabolite_context = self._extract_metabolite_context(elem)
                    if metabolite_context:  # Only yield if we got valid data
                        yield metabolite_context
                        metabolite_count += 1
                    
                    if metabolite_count % 1000 == 0:
                        print(f"Processed {metabolite_count} metabolites", file=sys.stderr)
                    
                    elem.clear()
                    root.clear()
                    
                    if max_metabolites and metabolite_count >= max_metabolites:
                        break
                        
                except Exception as e:
                    print(f"Error processing metabolite {metabolite_count}: {e}", file=sys.stderr)
                    continue
        
        print(f"Extracted {metabolite_count} metabolite contexts", file=sys.stderr)
    
    def _extract_metabolite_context(self, metabolite_elem) -> Optional[MetaboliteContext]:
        """Extract essential context from a single metabolite element."""
        
        # Get basic information
        hmdb_id = self._get_text(metabolite_elem, 'accession', '')
        if not hmdb_id:
            return None
        
        name = self._get_text(metabolite_elem, 'name', '')
        if not name:
            return None
        
        description = self._get_text(metabolite_elem, 'description', '')
        
        # Extract pathways
        pathways = self._extract_pathways(metabolite_elem)
        
        # Extract diseases
        diseases = self._extract_diseases(metabolite_elem)
        
        # Extract biochemical class
        biochemical_class = self._extract_biochemical_class(metabolite_elem)
        
        return MetaboliteContext(
            hmdb_id=hmdb_id,
            name=name,
            description=description,
            pathways=pathways,
            diseases=diseases,
            biochemical_class=biochemical_class
        )
    
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
        """Extract pathway names."""
        pathways = []
        pathways_elem = metabolite_elem.find(f'{self.namespace}biological_properties/{self.namespace}pathways')
        
        if pathways_elem is not None:
            for pathway in pathways_elem.findall(f'{self.namespace}pathway'):
                pathway_name = self._get_text(pathway, 'name', '')
                if pathway_name:
                    pathways.append(pathway_name)
        
        return pathways
    
    def _extract_diseases(self, metabolite_elem) -> List[str]:
        """Extract disease names."""
        diseases = []
        diseases_elem = metabolite_elem.find(f'{self.namespace}diseases')
        
        if diseases_elem is not None:
            for disease in diseases_elem.findall(f'{self.namespace}disease'):
                disease_name = self._get_text(disease, 'name', '')
                if disease_name:
                    diseases.append(disease_name)
        
        return diseases
    
    def _extract_biochemical_class(self, metabolite_elem) -> str:
        """Extract biochemical class information."""
        taxonomy_elem = metabolite_elem.find(f'{self.namespace}taxonomy')
        
        if taxonomy_elem is None:
            return ''
        
        # Try to get the most specific class available
        class_info = self._get_text(taxonomy_elem, 'class', '')
        if class_info:
            return class_info
        
        # Fall back to super class
        super_class = self._get_text(taxonomy_elem, 'super_class', '')
        if super_class:
            return super_class
        
        # Fall back to direct parent
        direct_parent = self._get_text(taxonomy_elem, 'direct_parent', '')
        return direct_parent
    
    def get_metabolite_by_name(self, metabolite_name: str) -> Optional[MetaboliteContext]:
        """
        Find a specific metabolite by name.
        
        Args:
            metabolite_name: Name of metabolite to find
            
        Returns:
            MetaboliteContext if found, None otherwise
        """
        for context in self.extract_metabolite_contexts():
            if context.name.lower() == metabolite_name.lower():
                return context
        return None
    
    def get_metabolite_by_hmdb_id(self, hmdb_id: str) -> Optional[MetaboliteContext]:
        """
        Find a specific metabolite by HMDB ID.
        
        Args:
            hmdb_id: HMDB ID to find (e.g., "HMDB0000122")
            
        Returns:
            MetaboliteContext if found, None otherwise
        """
        print(f"Searching for HMDB ID: {hmdb_id}", file=sys.stderr)
        
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        
        event, root = next(context)
        
        for event, elem in context:
            if event == 'end' and elem.tag == f'{self.namespace}metabolite':
                try:
                    # Quick check for HMDB ID first
                    found_id = self._get_text(elem, 'accession', '')
                    if found_id == hmdb_id:
                        print(f"✓ Found {hmdb_id}", file=sys.stderr)
                        metabolite_context = self._extract_metabolite_context(elem)
                        elem.clear()
                        root.clear()
                        return metabolite_context
                    
                    # Clear to save memory
                    elem.clear()
                    root.clear()
                        
                except Exception as e:
                    print(f"Error processing metabolite: {e}", file=sys.stderr)
                    continue
        
        print(f"✗ HMDB ID {hmdb_id} not found", file=sys.stderr)
        return None
    
    def get_metabolites_by_hmdb_ids(self, hmdb_ids: list) -> Dict[str, Optional[MetaboliteContext]]:
        """
        Find multiple metabolites by HMDB IDs efficiently.
        
        Args:
            hmdb_ids: List of HMDB IDs to find
            
        Returns:
            Dict mapping HMDB ID to MetaboliteContext (None if not found)
        """
        print(f"Searching for {len(hmdb_ids)} HMDB IDs", file=sys.stderr)
        
        results = {hmdb_id: None for hmdb_id in hmdb_ids}
        remaining_ids = set(hmdb_ids)
        
        context = ET.iterparse(self.xml_path, events=('start', 'end'))
        context = iter(context)
        
        event, root = next(context)
        processed = 0
        
        for event, elem in context:
            if event == 'end' and elem.tag == f'{self.namespace}metabolite':
                try:
                    # Quick check for HMDB ID
                    found_id = self._get_text(elem, 'accession', '')
                    
                    if found_id in remaining_ids:
                        print(f"✓ Found {found_id}", file=sys.stderr)
                        metabolite_context = self._extract_metabolite_context(elem)
                        results[found_id] = metabolite_context
                        remaining_ids.remove(found_id)
                        
                        # Early exit if we found everything
                        if not remaining_ids:
                            print("✓ Found all requested metabolites", file=sys.stderr)
                            break
                    
                    processed += 1
                    if processed % 5000 == 0:
                        print(f"Processed {processed} metabolites, {len(remaining_ids)} remaining", file=sys.stderr)
                    
                    # Clear to save memory
                    elem.clear()
                    root.clear()
                        
                except Exception as e:
                    print(f"Error processing metabolite: {e}", file=sys.stderr)
                    continue
        
        if remaining_ids:
            print(f"✗ Not found: {remaining_ids}", file=sys.stderr)
        
        return results


def test_scraper(xml_path: Path, test_metabolite: str = "glucose"):
    """Test the simple scraper with a specific metabolite."""
    scraper = SimpleHMDBScraper(xml_path)
    
    print(f"Testing simple scraper with metabolite: {test_metabolite}")
    print("=" * 60)
    
    # Test finding specific metabolite
    context = scraper.get_metabolite_by_name(test_metabolite)
    
    if context:
        print("✅ Found metabolite!")
        print(f"HMDB ID: {context.hmdb_id}")
        print(f"Name: {context.name}")
        print(f"Pathways: {len(context.pathways)}")
        print(f"Diseases: {len(context.diseases)}")
        print(f"Biochemical class: {context.biochemical_class}")
        
        print("\n" + "=" * 60)
        print("LLM CONTEXT STRING:")
        print("=" * 60)
        print(context.to_context_string())
        
    else:
        print(f"❌ Metabolite '{test_metabolite}' not found")
        
        # Show first few metabolites as examples
        print("\nFirst 5 metabolites found:")
        for i, ctx in enumerate(scraper.extract_metabolite_contexts(max_metabolites=5)):
            print(f"  {i+1}. {ctx.name} ({ctx.hmdb_id})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
        test_metabolite = sys.argv[2] if len(sys.argv) > 2 else "glucose"
        test_scraper(xml_path, test_metabolite)
    else:
        print("Usage: python simple_hmdb_scraper.py <xml_path> [metabolite_name]")