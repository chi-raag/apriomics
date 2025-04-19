import os
import pandas as pd
import numpy as np
from .utils.smiles import get_smiles_from_names
from .utils.fingerprints import generate_map4_fingerprints, create_similarity_matrix as create_similarity_matrix_util
from .utils.fingerprints import calculate_fingerprints_batch
from typing import Dict, List, Tuple, Any, Optional, Union

# Data structure to pass between functions
class PriorData:
    def __init__(self, 
                 dimensions: int = 1024,
                 smiles_data = None,
                 fingerprints_data = None,
                 similarity_matrix = None,
                 metabolite_names = None):
        self.dimensions = dimensions
        self.smiles_data = smiles_data
        self.fingerprints_data = fingerprints_data
        self.similarity_matrix = similarity_matrix
        self.metabolite_names = metabolite_names

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
        metabolite_names=priors.metabolite_names
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
        metabolite_names=priors.metabolite_names
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
        metabolite_names=metabolite_names
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

def save_results(priors: PriorData, output_dir: str = '.') -> Dict[str, str]:
    """
    Save all results to files
    
    Parameters:
    -----------
    priors : PriorData
        Data container with results
    output_dir : str
        Directory to save files in
        
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
        
    return files

def run_pipeline(dimensions: int = 1024, 
                metabolites: Optional[List[str]] = None, 
                excel_files: Optional[Union[str, List[str]]] = None, 
                max_workers: int = 4, 
                output_dir: Optional[str] = None) -> PriorData:
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