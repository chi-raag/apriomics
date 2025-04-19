import pandas as pd
import numpy as np
from rdkit import Chem
import map4
from mhfp.encoder import MHFPEncoder

def generate_map4_fingerprints(smiles_df, dimensions=1024):
    """
    Generate MAP4 fingerprints for a DataFrame containing SMILES
    
    Parameters:
    -----------
    smiles_df : pandas.DataFrame
        DataFrame with a 'smiles' column
    dimensions : int
        Size of the fingerprint (default 1024)
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added fingerprint column
    """
    # Initialize MAP4 calculator
    map4_calc = map4.MAP4(dimensions=dimensions)
    
    # Function to calculate fingerprint for a single SMILES
    def calculate_fp(smiles):
        if pd.isna(smiles):
            return None
        try:
            # First validate SMILES with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Calculate MAP4 fingerprint
            return map4_calc.calculate(mol)
        except Exception as e:
            print(f"Error calculating fingerprint for {smiles}: {e}")
            return None
    
    # Calculate fingerprints for all SMILES
    smiles_df['map4_fingerprint'] = smiles_df['smiles'].apply(calculate_fp)
    
    return smiles_df

def calculate_fingerprints_batch(mols, dimensions=1024):
    """
    Calculate MAP4 fingerprints for a batch of molecules
    
    Parameters:
    -----------
    mols : list
        List of RDKit molecules
    dimensions : int
        Size of the fingerprint (default 1024)
        
    Returns:
    --------
    list
        List of MAP4 fingerprints
    """
    # Initialize MAP4 calculator
    map4_calc = map4.MAP4(dimensions=dimensions)
    
    # Calculate fingerprints for all molecules in batch
    return map4_calc.calculate_many(mols)

def create_similarity_matrix(fingerprints_df):
    """
    Create a similarity matrix from MAP4 fingerprints using tmap distance
    
    Parameters:
    -----------
    fingerprints_df : pandas.DataFrame
        DataFrame with 'metabolite' and 'map4_fingerprint' columns
        
    Returns:
    --------
    tuple
        (similarity_matrix, metabolite_names)
    """
    # Filter out rows with missing fingerprints
    valid_fps = fingerprints_df.dropna(subset=['map4_fingerprint']).copy()
    metabolite_names = valid_fps['metabolite'].tolist()
    fingerprints = valid_fps['map4_fingerprint'].tolist()
    
    # Create similarity matrix
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    # Initialize the tmap encoder
    encoder = MHFPEncoder()
    
    for i in range(n):
        for j in range(n):
            # Calculate distance using tmap
            distance = encoder.distance(fingerprints[i], fingerprints[j])
            # Convert distance to similarity (1 - normalized distance)
            # As distance approaches 0, similarity approaches 1
            similarity_matrix[i, j] = 1.0 - distance
    
    return similarity_matrix, metabolite_names 