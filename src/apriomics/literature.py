from google import genai
from typing import List, Optional
import re
import dspy

# ------------------------------------------------------------------
# DSPy Signature for pathway cleaning
# ------------------------------------------------------------------
class PathwayCleaning(dspy.Signature):
    """Clean and normalize metabolic pathway names"""
    pathways = dspy.InputField(desc="Raw list of pathway names to clean")
    cleaned_pathways = dspy.OutputField(desc="Cleaned pathway names, one per line")

# ------------------------------------------------------------------
# DSPy-based pathway cleaner
# ------------------------------------------------------------------
class DSPyPathwayCleaner:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # Note: DSPy will need to be configured with the appropriate LM
        self.predictor = dspy.Predict(PathwayCleaning)
    
    def clean_pathway_list(self, pathway_list: List[str]) -> List[str]:
        """
        Uses DSPy to clean a list of pathway names.
        
        Args:
            pathway_list: The initial list of potentially messy pathway names.
            
        Returns:
            A cleaned list of pathway names, or an empty list if cleaning fails.
        """
        if not pathway_list:
            return []
        
        list_as_string = "\n".join(pathway_list)
        
        try:
            result = self.predictor(pathways=list_as_string)
            cleaned_list_text = result.cleaned_pathways.strip()
            
            # Parse the result
            cleaned_list = [line.strip() for line in cleaned_list_text.split('\n') if line.strip()]
            
            # Sanity check
            if not cleaned_list or not all(len(p) > 3 for p in cleaned_list):
                print(f"Warning: DSPy cleaning returned unexpected result: {cleaned_list_text}")
                return self._fallback_cleaning(pathway_list)
            
            print(f"Cleaned list ({len(cleaned_list)} pathways): {cleaned_list}")
            return cleaned_list
            
        except Exception as e:
            print(f"Error during DSPy pathway cleaning: {e}")
            return self._fallback_cleaning(pathway_list)
    
    def _fallback_cleaning(self, pathway_list: List[str]) -> List[str]:
        """Fallback programmatic cleaning when DSPy fails"""
        print("Applying basic programmatic cleaning as fallback.")
        processed = set()
        fallback_list = []
        for item in pathway_list:
            # Basic normalization
            norm_item = re.sub(r'\s*\(.*\)\s*', '', item).strip() # Remove (...) content
            norm_item = norm_item.strip('/').strip()
            norm_item = norm_item.title()
            # Simple split
            parts = re.split(r'[;/]', norm_item)
            for part in parts:
                part = part.strip()
                if part and part.lower() not in processed:
                    # Avoid very generic terms programmatically too
                    if part.lower() not in ['chemical', 'drug', 'dipeptide', 'lysolipid', 'essential fatty acid', 'gamma-glutamyl', 'monoacylglycerol', 'dipeptide derivative'] and not part.startswith('Drug -'):
                         fallback_list.append(part)
                         processed.add(part.lower())
        return fallback_list

# ------------------------------------------------------------------
# Legacy function (kept for backward compatibility)
# ------------------------------------------------------------------
def clean_pathway_list_with_llm(
    pathway_list: List[str],
    client: Optional[genai.Client] = None,
    model_name: str = "gemini-2.0-flash",
    use_dspy: bool = True
) -> List[str]:
    """
    Uses an LLM to clean a list of pathway names.
    
    Args:
        pathway_list: The initial list of potentially messy pathway names.
        client: An initialized google.generativeai Client (deprecated when use_dspy=True).
        model_name: The name of the Gemini model to use.
        use_dspy: Whether to use DSPy (recommended) or legacy approach.
        
    Returns:
        A cleaned list of pathway names, or an empty list if cleaning fails.
    """
    if use_dspy:
        cleaner = DSPyPathwayCleaner(model_name=model_name)
        return cleaner.clean_pathway_list(pathway_list)
    
    # Legacy implementation
    if not pathway_list:
        return []
    
    if client is None:
        raise ValueError("client is required when use_dspy=False")

    list_as_string = "\n".join(pathway_list)

    prompt = f"""
    Analyze the following list of metabolic pathway names. Perform the following cleaning steps:
    1. Normalize capitalization to Title Case for all pathway names.
    2. Remove exact duplicate pathway names (case-insensitive comparison after normalization).
    3. Identify entries that seem to combine multiple distinct pathways (often separated by ';', '/', or similar conjunctions like 'and'). Split these into separate, valid pathway names on new lines.
    4. Remove entries that are too generic or broad to be useful as specific pathway search terms on PubMed (e.g., 'Chemical', 'Drug', 'Food Component/Plant', 'Dipeptide', 'Sterol/Steroid', 'Lysolipid', 'Essential fatty acid', 'gamma-glutamyl', 'Monoacylglycerol', 'Dipeptide Derivative'). Also remove generic drug categories like 'Drug - Cardiovascular'.
    5. Remove parenthetical qualifiers or alternative names like '(also BCAA metabolism)' or '(Acyl Carnitine)', keeping only the primary pathway name.
    6. Remove trailing slashes or punctuation.

    Return *only* the final, cleaned list of pathway names, with each distinct pathway on a new line. Do not include any explanations, comments, or numbering.

    Original List:
    --------------
    {list_as_string}
    --------------

    Cleaned List:
    """

    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        cleaned_list_text = response.text.strip()

        # Basic parsing: split by newline and remove empty strings
        cleaned_list = [line.strip() for line in cleaned_list_text.split('\n') if line.strip()]

        # Further sanity check: Ensure the results look like pathway names
        if not cleaned_list or not all(len(p) > 3 for p in cleaned_list):
             print(f"Warning: LLM cleaning returned unexpected result: {cleaned_list_text}")
             # Fallback: Basic programmatic cleaning (deduplication and title case)
             print("Applying basic programmatic cleaning as fallback.")
             processed = set()
             fallback_list = []
             for item in pathway_list:
                # Basic normalization
                norm_item = re.sub(r'\s*\(.*\)\s*', '', item).strip() # Remove (...) content
                norm_item = norm_item.strip('/').strip()
                norm_item = norm_item.title()
                # Simple split
                parts = re.split(r'[;/]', norm_item)
                for part in parts:
                    part = part.strip()
                    if part and part.lower() not in processed:
                        # Avoid very generic terms programmatically too
                        if part.lower() not in ['chemical', 'drug', 'dipeptide', 'lysolipid', 'essential fatty acid', 'gamma-glutamyl', 'monoacylglycerol', 'dipeptide derivative'] and not part.startswith('Drug -'):
                             fallback_list.append(part)
                             processed.add(part.lower())
             return fallback_list

        print(f"Cleaned list ({len(cleaned_list)} pathways): {cleaned_list}")
        return cleaned_list

    except Exception as e:
        print(f"Error during LLM pathway cleaning: {e}")
        # Handle potential API errors
        try:
            print(f"Prompt Feedback: {response.prompt_feedback}")
        except Exception:
            pass
        return [] # Return empty list on error 