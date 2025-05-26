import os
import sys
import pandas as pd
import numpy as np
from .utils.smiles import get_smiles_from_names
from .utils.fingerprints import generate_map4_fingerprints, create_similarity_matrix as create_similarity_matrix_util
from .utils.fingerprints import calculate_fingerprints_batch
from .hmdb_utils import batch_get_metabolite_contexts, EXAMPLE_METABOLITE_MAPPINGS
from typing import Dict, List, Tuple, Any, Optional, Union

# Data structure to pass between functions
class PriorData:
    def __init__(self, 
                 dimensions: int = 1024,
                 smiles_data = None,
                 fingerprints_data = None,
                 similarity_matrix = None,
                 metabolite_names = None,
                 hmdb_contexts = None):
        self.dimensions = dimensions
        self.smiles_data = smiles_data
        self.fingerprints_data = fingerprints_data
        self.similarity_matrix = similarity_matrix
        self.metabolite_names = metabolite_names
        self.hmdb_contexts = hmdb_contexts

def load_metabolites_from_excel(file_paths: Union[str, List[str]]) -> List[str]:
    """
    Load metabolite names from Excel files
    
    Parameters:
    -----------
    file_paths : str or list
        Path(s) to Excel file(s) containing metabolite data
        
    Returns:
    --------
    list
        List of metabolite names
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    metabolites = []
    for file in file_paths:
        df = pd.read_excel(file)
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        # Rename first column to 'metabolite' if it's not already
        if df.columns[0] != 'metabolite':
            df = df.rename(columns={df.columns[0]: 'metabolite'})
        metabolites.extend(df['metabolite'].tolist())
    
    # Remove duplicates and None values
    metabolites = [m for m in metabolites if m is not None]
    metabolites = list(dict.fromkeys(metabolites))
    
    return metabolites

def get_hmdb_contexts(priors: PriorData, metabolites: List[str], 
                      hmdb_mapping: Optional[Dict[str, str]] = None) -> PriorData:
    """
    Retrieve HMDB context information for metabolites.
    
    Parameters:
    -----------
    priors : PriorData
        Data container
    metabolites : list
        List of metabolite names
    hmdb_mapping : dict, optional
        Mapping of metabolite names to HMDB IDs. If not provided, will use example mappings.
        
    Returns:
    --------
    PriorData
        Updated data container with HMDB contexts
    """
    if hmdb_mapping is None:
        hmdb_mapping = EXAMPLE_METABOLITE_MAPPINGS
    
    hmdb_contexts = batch_get_metabolite_contexts(metabolites, hmdb_mapping)
    
    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=priors.smiles_data,
        fingerprints_data=priors.fingerprints_data,
        similarity_matrix=priors.similarity_matrix,
        metabolite_names=priors.metabolite_names,
        hmdb_contexts=hmdb_contexts
    )

def get_smiles(priors: PriorData, metabolites: List[str], max_workers: int = 4) -> PriorData:
    """
    Retrieve SMILES for a list of metabolite names
    
    Parameters:
    -----------
    priors : PriorData
        Data container
    metabolites : list
        List of metabolite names
    max_workers : int
        Number of parallel workers for API requests
        
    Returns:
    --------
    PriorData
        Updated data container with SMILES
    """
    smiles_data = get_smiles_from_names(metabolites, max_workers)
    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=smiles_data,
        fingerprints_data=priors.fingerprints_data,
        similarity_matrix=priors.similarity_matrix,
        metabolite_names=priors.metabolite_names,
        hmdb_contexts=priors.hmdb_contexts
    )

def generate_fingerprints(priors: PriorData) -> PriorData:
    """
    Generate fingerprints from SMILES data
    
    Parameters:
    -----------
    priors : PriorData
        Data container with SMILES data
        
    Returns:
    --------
    PriorData
        Updated data container with fingerprints
    """
    if priors.smiles_data is None:
        raise ValueError("No SMILES data available. Run get_smiles() first.")
        
    # Filter out entries without SMILES
    smiles_non_na = priors.smiles_data[priors.smiles_data['smiles'].notna()].copy()
    
    fingerprints_data = generate_map4_fingerprints(
        smiles_non_na, 
        dimensions=priors.dimensions
    )
    
    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=priors.smiles_data,
        fingerprints_data=fingerprints_data,
        similarity_matrix=priors.similarity_matrix,
        metabolite_names=priors.metabolite_names,
        hmdb_contexts=priors.hmdb_contexts
    )

def create_similarity_matrix(priors: PriorData) -> PriorData:
    """
    Create similarity matrix from fingerprints
    
    Parameters:
    -----------
    priors : PriorData
        Data container with fingerprint data
        
    Returns:
    --------
    PriorData
        Updated data container with similarity matrix
    """
    if priors.fingerprints_data is None:
        raise ValueError("No fingerprint data available. Run generate_fingerprints() first.")
        
    similarity_matrix, metabolite_names = create_similarity_matrix_util(priors.fingerprints_data)
    
    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=priors.smiles_data,
        fingerprints_data=priors.fingerprints_data,
        similarity_matrix=similarity_matrix,
        metabolite_names=metabolite_names,
        hmdb_contexts=priors.hmdb_contexts
    )

def get_kernel(priors: PriorData, scale: float = 1.0) -> np.ndarray:
    """
    Get the similarity kernel for use in Gaussian Process models
    
    Parameters:
    -----------
    priors : PriorData
        Data container with similarity matrix
    scale : float
        Scaling factor for the kernel
        
    Returns:
    --------
    numpy.ndarray
        Scaled similarity matrix
    """
    if priors.similarity_matrix is None:
        raise ValueError("No similarity matrix available. Run create_similarity_matrix() first.")
        
    # Apply scaling and add small diagonal term for numerical stability
    kernel = scale * priors.similarity_matrix
    kernel = kernel + 1e-6 * np.eye(kernel.shape[0])
    
    return kernel

def get_metabolite_context_for_llm(priors: PriorData, condition: str = "") -> str:
    """
    Generate combined context string for LLM from HMDB data and chemical similarity.
    
    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts and similarity data
    condition : str
        Optional study condition to include in context
        
    Returns:
    --------
    str
        Formatted context string for LLM prompts
    """
    context_parts = []
    
    if condition:
        context_parts.append(f"Study condition: {condition}")
    
    if priors.hmdb_contexts:
        context_parts.append("Metabolite information from HMDB:")
        for metabolite, context in priors.hmdb_contexts.items():
            context_parts.append(f"- {context}")
    
    if priors.similarity_matrix is not None and priors.metabolite_names:
        context_parts.append(f"\nChemical similarity data available for {len(priors.metabolite_names)} metabolites.")
        
        # Add information about highly similar metabolite pairs
        similarity_threshold = 0.8
        similar_pairs = []
        n_metabolites = len(priors.metabolite_names)
        
        for i in range(n_metabolites):
            for j in range(i + 1, n_metabolites):
                similarity = priors.similarity_matrix[i, j]
                if similarity > similarity_threshold:
                    similar_pairs.append((
                        priors.metabolite_names[i], 
                        priors.metabolite_names[j], 
                        similarity
                    ))
        
        if similar_pairs:
            context_parts.append(f"Highly similar metabolite pairs (similarity > {similarity_threshold}):")
            for met1, met2, sim in similar_pairs[:5]:  # Limit to top 5
                context_parts.append(f"- {met1} ↔ {met2} (similarity: {sim:.2f})")
    
    return "\n".join(context_parts)

def get_llm_differential_priors(priors: PriorData, condition: str, 
                               llm_scorer=None, batch_size: int = 10) -> Dict[str, float]:
    """
    Generate LLM-informed priors for differential expression analysis.
    
    This function uses HMDB context and LLM expertise to generate importance scores
    for metabolites in the context of a specific biological condition.
    
    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts
    condition : str
        Study condition or experimental design (e.g., "diabetes vs control")
    llm_scorer : object, optional
        LLM scorer object (e.g., GeminiScorer from llm-lasso.py). If None, returns uniform priors.
    batch_size : int
        Number of metabolites to process in each LLM batch
        
    Returns:
    --------
    dict
        Dictionary mapping metabolite names to importance scores (0-1)
    """
    if priors.hmdb_contexts is None:
        raise ValueError("No HMDB contexts available. Run get_hmdb_contexts() first.")
    
    metabolites = list(priors.hmdb_contexts.keys())
    
    # If no LLM scorer provided, return uniform priors
    if llm_scorer is None:
        print("Warning: No LLM scorer provided. Returning uniform priors.", file=sys.stderr)
        return {met: 0.5 for met in metabolites}
    
    # Process metabolites in batches using LLM scorer
    differential_priors = {}
    
    for i in range(0, len(metabolites), batch_size):
        batch_metabolites = metabolites[i:i + batch_size]
        batch_contexts = {met: priors.hmdb_contexts[met] for met in batch_metabolites}
        
        try:
            # Use the LLM scorer to get importance scores
            llm_scores = llm_scorer.score_batch(condition, batch_contexts)
            
            # Extract importance scores
            for score_obj in llm_scores:
                if hasattr(score_obj, 'metabolite') and hasattr(score_obj, 'score'):
                    differential_priors[score_obj.metabolite] = float(score_obj.score)
                else:
                    # Fallback if score object structure is different
                    print(f"Warning: Unexpected LLM score format for batch starting at {i}", file=sys.stderr)
                    
        except Exception as e:
            print(f"Error processing LLM batch starting at {i}: {e}", file=sys.stderr)
            # Assign default scores for this batch
            for met in batch_metabolites:
                differential_priors[met] = 0.5
    
    # Ensure all metabolites have scores
    for met in metabolites:
        if met not in differential_priors:
            differential_priors[met] = 0.5
    
    return differential_priors

def get_network_priors(priors: PriorData, threshold: float = 0.7) -> Dict[Tuple[str, str], float]:
    """
    Generate network priors from chemical similarity matrix.
    
    This function creates prior beliefs about metabolite-metabolite interactions
    based on chemical similarity for network analysis.
    
    Parameters:
    -----------
    priors : PriorData
        Data container with similarity matrix
    threshold : float
        Minimum similarity threshold for including edges
        
    Returns:
    --------
    dict
        Dictionary mapping metabolite pairs to similarity scores
    """
    if priors.similarity_matrix is None or priors.metabolite_names is None:
        raise ValueError("No similarity matrix available. Run create_similarity_matrix() first.")
    
    network_priors = {}
    n_metabolites = len(priors.metabolite_names)
    
    for i in range(n_metabolites):
        for j in range(i + 1, n_metabolites):
            similarity = priors.similarity_matrix[i, j]
            if similarity >= threshold:
                met1 = priors.metabolite_names[i]
                met2 = priors.metabolite_names[j]
                network_priors[(met1, met2)] = float(similarity)
    
    return network_priors

def save_results(priors: PriorData, output_dir: str = '.', 
                 differential_priors: Optional[Dict[str, float]] = None,
                 network_priors: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, str]:
    """
    Save all results to files
    
    Parameters:
    -----------
    priors : PriorData
        Data container with results
    output_dir : str
        Directory to save files in
    differential_priors : dict, optional
        LLM-generated differential priors to save
    network_priors : dict, optional
        Chemical similarity-based network priors to save
        
    Returns:
    --------
    dict
        Dictionary of file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = {}
    
    if priors.smiles_data is not None:
        smiles_path = os.path.join(output_dir, 'metabolite_smiles.csv')
        priors.smiles_data.to_csv(smiles_path, index=False)
        files['smiles'] = smiles_path
        
    if priors.fingerprints_data is not None:
        fp_path = os.path.join(output_dir, 'metabolite_fingerprints.csv')
        # Save fingerprints as strings for CSV compatibility
        fp_df = priors.fingerprints_data.copy()
        fp_df['map4_fingerprint'] = fp_df['map4_fingerprint'].apply(
            lambda x: ','.join(map(str, x)) if x is not None else None
        )
        fp_df.to_csv(fp_path, index=False)
        files['fingerprints'] = fp_path
        
    if priors.similarity_matrix is not None and priors.metabolite_names is not None:
        sim_path = os.path.join(output_dir, 'metabolite_similarity_matrix.csv')
        sim_df = pd.DataFrame(
            priors.similarity_matrix, 
            index=priors.metabolite_names, 
            columns=priors.metabolite_names
        )
        sim_df.to_csv(sim_path)
        files['similarity_matrix'] = sim_path
    
    if priors.hmdb_contexts is not None:
        hmdb_path = os.path.join(output_dir, 'hmdb_contexts.csv')
        hmdb_df = pd.DataFrame([
            {'metabolite': metabolite, 'hmdb_context': context}
            for metabolite, context in priors.hmdb_contexts.items()
        ])
        hmdb_df.to_csv(hmdb_path, index=False)
        files['hmdb_contexts'] = hmdb_path
    
    # Save differential priors if provided
    if differential_priors is not None:
        diff_path = os.path.join(output_dir, 'differential_priors.csv')
        diff_df = pd.DataFrame([
            {'metabolite': metabolite, 'importance_score': score}
            for metabolite, score in differential_priors.items()
        ])
        diff_df.to_csv(diff_path, index=False)
        files['differential_priors'] = diff_path
    
    # Save network priors if provided
    if network_priors is not None:
        network_path = os.path.join(output_dir, 'network_priors.csv')
        network_df = pd.DataFrame([
            {'metabolite_1': pair[0], 'metabolite_2': pair[1], 'similarity_score': score}
            for pair, score in network_priors.items()
        ])
        network_df.to_csv(network_path, index=False)
        files['network_priors'] = network_path
        
    return files

def run_pipeline(dimensions: int = 1024, 
                metabolites: Optional[List[str]] = None, 
                excel_files: Optional[Union[str, List[str]]] = None, 
                max_workers: int = 4, 
                output_dir: Optional[str] = None,
                include_hmdb: bool = True,
                hmdb_mapping: Optional[Dict[str, str]] = None) -> PriorData:
    """
    Run the complete pipeline
    
    Parameters:
    -----------
    dimensions : int
        Number of dimensions for the MAP4 fingerprints
    metabolites : list, optional
        List of metabolite names. If None, must provide excel_files.
    excel_files : str or list, optional
        Path(s) to Excel file(s) containing metabolite data.
        If None, must provide metabolites.
    max_workers : int
        Number of parallel workers for API requests
    output_dir : str, optional
        Directory to save results. If None, results are not saved.
    include_hmdb : bool
        Whether to include HMDB context data
    hmdb_mapping : dict, optional
        Mapping of metabolite names to HMDB IDs
        
    Returns:
    --------
    PriorData
        Data container with all results
    """
    if metabolites is None and excel_files is None:
        raise ValueError("Must provide either metabolites list or excel_files path(s)")
        
    if metabolites is None:
        metabolites = load_metabolites_from_excel(excel_files)
    
    # Initialize empty data container
    priors = PriorData(dimensions=dimensions)
    
    # Apply each function in sequence
    priors = get_smiles(priors, metabolites, max_workers)
    priors = generate_fingerprints(priors)
    priors = create_similarity_matrix(priors)
    
    # Add HMDB contexts if requested
    if include_hmdb:
        priors = get_hmdb_contexts(priors, metabolites, hmdb_mapping)
    
    if output_dir is not None:
        save_results(priors, output_dir)
        
    return priors

# For full functional style without the PriorData class, pipe/compose functions could be used
def pipe(data, *functions):
    """Function composition - right to left"""
    result = data
    for func in functions:
        result = func(result)
    return result

# Example of more pure functional usage without PriorData class (alternative approach)
# This would require rewriting the functions to accept and return plain dictionaries
# instead of the PriorData class 