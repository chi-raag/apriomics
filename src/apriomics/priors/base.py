import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# Data structure to pass between functions
class PriorData:
    def __init__(
        self,
        dimensions: int = 1024,
        smiles_data=None,
        fingerprints_data=None,
        similarity_matrix=None,
        metabolite_names=None,
        hmdb_contexts=None,
    ):
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
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        # Rename first column to 'metabolite' if it's not already
        if df.columns[0] != "metabolite":
            df = df.rename(columns={df.columns[0]: "metabolite"})
        metabolites.extend(df["metabolite"].tolist())

    # Remove duplicates and None values
    metabolites = [m for m in metabolites if m is not None]
    metabolites = list(dict.fromkeys(metabolites))

    return metabolites


def load_mtbls1_data(file_path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Loads and processes the MTBLS1 dataset from the specified TSV file.

    Args:
        file_path: The path to the MTBLS1 data file
                   (m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv).

    Returns:
        A tuple containing:
        - A list of metabolite names.
        - A list of sample names.
        - A numpy array of the abundance data (metabolites x samples).
    """
    try:
        df = pd.read_csv(file_path, sep="\t")
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}", file=sys.stderr)
        raise

    # Extract metabolite names
    metabolite_names = df["metabolite_identification"].tolist()

    # Identify sample columns (they typically start with a study-specific prefix)
    sample_columns = [
        col
        for col in df.columns
        if "ADG" in col
        or "smallmolecule_abundance" not in col
        and col
        not in [
            "database_identifier",
            "chemical_formula",
            "smiles",
            "inchi",
            "metabolite_identification",
            "chemical_shift",
            "multiplicity",
            "taxid",
            "species",
            "database",
            "database_version",
            "reliability",
            "uri",
            "search_engine",
            "search_engine_score",
        ]
    ]

    # Extract abundance data
    abundance_data = df[sample_columns].values

    # The data is samples x metabolites, so we transpose it
    abundance_data = abundance_data.T

    return metabolite_names, sample_columns, abundance_data


def get_hmdb_contexts(
    priors: PriorData,
    metabolites: List[str],
    hmdb_mapping: Optional[Dict[str, str]] = None,
) -> PriorData:
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
        hmdb_contexts=priors.hmdb_contexts,
    )


def get_smiles(
    priors: PriorData, metabolites: List[str], max_workers: int = 4
) -> PriorData:
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
        hmdb_contexts=priors.hmdb_contexts,
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
    smiles_non_na = priors.smiles_data[priors.smiles_data["smiles"].notna()].copy()

    fingerprints_data = generate_map4_fingerprints(
        smiles_non_na, dimensions=priors.dimensions
    )

    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=priors.smiles_data,
        fingerprints_data=fingerprints_data,
        similarity_matrix=priors.similarity_matrix,
        metabolite_names=priors.metabolite_names,
        hmdb_contexts=priors.hmdb_contexts,
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
        raise ValueError(
            "No fingerprint data available. Run generate_fingerprints() first."
        )

    similarity_matrix, metabolite_names = create_similarity_matrix_util(
        priors.fingerprints_data
    )

    return PriorData(
        dimensions=priors.dimensions,
        smiles_data=priors.smiles_data,
        fingerprints_data=priors.fingerprints_data,
        similarity_matrix=priors.similarity_matrix,
        metabolite_names=metabolite_names,
        hmdb_contexts=priors.hmdb_contexts,
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
        raise ValueError(
            "No similarity matrix available. Run create_similarity_matrix() first."
        )

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
        context_parts.append(
            f"\nChemical similarity data available for {len(priors.metabolite_names)} metabolites."
        )

        # Add information about highly similar metabolite pairs
        similarity_threshold = 0.8
        similar_pairs = []
        n_metabolites = len(priors.metabolite_names)

        for i in range(n_metabolites):
            for j in range(i + 1, n_metabolites):
                similarity = priors.similarity_matrix[i, j]
                if similarity > similarity_threshold:
                    similar_pairs.append(
                        (
                            priors.metabolite_names[i],
                            priors.metabolite_names[j],
                            similarity,
                        )
                    )

        if similar_pairs:
            context_parts.append(
                f"Highly similar metabolite pairs (similarity > {similarity_threshold}):"
            )
            for met1, met2, sim in similar_pairs[:5]:  # Limit to top 5
                context_parts.append(f"- {met1} ↔ {met2} (similarity: {sim:.2f})")

    return "\n".join(context_parts)


def get_llm_differential_priors(
    priors: PriorData,
    condition: str,
    llm_scorer=None,
    hmdb_retriever=None,
    use_hmdb_context: bool = True,
    prior_strength: str = "conservative",
) -> Dict[str, float]:
    """
    Generate LLM-informed priors for differential expression analysis.

    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts
    condition : str
        Study condition or experimental design (e.g., "diabetes vs control")
    llm_scorer : object, optional
        LLM scorer object. If None, creates DSPyGeminiScorer.
    batch_size : int
        Number of metabolites to process in each LLM batch
    hmdb_retriever : HMDBRetriever, optional
        RAG-based HMDB retriever for enhanced context. If provided, replaces hmdb_contexts.

    Returns:
    --------
    dict
        Dictionary mapping metabolite names to comprehensive prior information:
        - 'relevance': float (0-1) - importance score for the condition
        - 'direction': str - expected regulation direction ('increase', 'decrease', 'minimal', 'unclear')
        - 'rationale': str - explanation for the assessment
        - 'expected_log2fc': float - expected log2-fold-change for Bayesian priors
        - 'prior_sd': float - standard deviation for the log2fc prior
        - 'magnitude': str - effect size category ('minimal', 'small', 'moderate', 'large')
        - 'confidence': str - assessment confidence ('high', 'moderate', 'low')
    """
    import sys

    # Determine metabolites and get contexts
    if use_hmdb_context:
        if hmdb_retriever is not None:
            # Use RAG retriever - get metabolites from existing data or extract from priors
            if hasattr(priors, "metabolite_names") and priors.metabolite_names:
                metabolites = priors.metabolite_names
            elif priors.hmdb_contexts:
                metabolites = list(priors.hmdb_contexts.keys())
            else:
                raise ValueError(
                    "No metabolite names available. Provide metabolites in PriorData or hmdb_contexts."
                )

            # Get enhanced contexts using RAG
            print("Using HMDB RAG retriever for enhanced contexts.", file=sys.stderr)
            batch_contexts = hmdb_retriever.get_metabolite_contexts_batch(
                metabolites, condition=condition
            )
        else:
            # Use traditional approach
            if priors.hmdb_contexts is None:
                raise ValueError(
                    "No HMDB contexts available. Run get_hmdb_contexts() first or provide hmdb_retriever."
                )

            metabolites = list(priors.hmdb_contexts.keys())
            batch_contexts = priors.hmdb_contexts
    else:
        if not priors.metabolite_names:
            raise ValueError(
                "No metabolite names available. Run get_smiles() and generate_fingerprints() first."
            )
        metabolites = priors.metabolite_names
        batch_contexts = {name: name for name in metabolites}

    if llm_scorer is None:
        if os.getenv("GOOGLE_API_KEY"):
            try:
                import google.generativeai as genai
                import json
                import re

                # Configure Gemini directly with deterministic settings
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(
                    "gemini-2.5-flash-lite-preview-06-17",  # Use non-thinking model for reliable JSON parsing
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,  # Deterministic results like the paper
                        top_p=1.0,
                        top_k=1,
                    ),
                )

                class DirectGeminiScorer:
                    def __init__(self, model):
                        self.model = model

                    def analyze_metabolite(
                        self, condition: str, metabolite: str, context: str
                    ):
                        """Analyze a single metabolite for focused, scientifically rigorous analysis."""

                        prompt = f"""
You are a metabolomics expert analyzing how metabolites change in disease conditions. Your task is to provide a scientifically rigorous assessment based on the provided information.

**Study Context:**
{condition}

**Metabolite Information:**
{context}

**Analysis Task:**
1.  **Summarize Study:** Briefly summarize the key aspects of the study, including the condition, sample type, and patient characteristics.
2.  **Identify Relevant Biology:** Based on the metabolite's background, identify the most relevant pathways, diseases, or functions related to the study context.
3.  **Formulate Hypothesis:** Synthesize the information from steps 1 and 2 to form a hypothesis about the metabolite's expected change (increase, decrease, or unchanged) and the likely magnitude of this change (small, moderate, or large). Explain your reasoning.

**Prediction:**
Based on your analysis, provide your final prediction in the following JSON format:
{{
    "prediction": "<increase|decrease|unchanged>",
    "magnitude": "<small|moderate|large>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<A brief summary of your hypothesis.>"
}}
"""

                        try:
                            response = self.model.generate_content(prompt)
                            response_text = response.text.strip()

                            # Clean up the response to extract JSON
                            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                            else:
                                json_str = response_text.strip()

                            # Remove markdown code blocks if present
                            json_str = re.sub(r"```json\s*", "", json_str)
                            json_str = re.sub(r"```\s*$", "", json_str)

                            # Parse JSON response
                            response_data = json.loads(json_str)

                            # Extract values with defaults
                            prediction = response_data.get(
                                "prediction", "unchanged"
                            ).lower()
                            magnitude = response_data.get("magnitude", "small").lower()
                            confidence = float(response_data.get("confidence", 0.5))
                            reasoning = response_data.get(
                                "reasoning", "No reasoning provided"
                            )

                            # --- Conservative Prior Generation ---
                            def conservative_directional_prior(prediction, confidence):
                                """Original conservative prior generation."""
                                if prediction == "increase":
                                    prior_mean = 0.15 * confidence
                                elif prediction == "decrease":
                                    prior_mean = -0.15 * confidence
                                else:
                                    prior_mean = 0.0
                                prior_sd = 1.0 - (0.6 * confidence)
                                return prior_mean, prior_sd

                            # --- Moderate Prior Generation ---
                            def moderate_directional_prior(prediction, magnitude, confidence):
                                """New, less conservative prior generation."""
                                # Map magnitude to effect size
                                if prediction == "unchanged":
                                    prior_mean = 0.0
                                else:
                                    sign = 1 if prediction == "increase" else -1
                                    if magnitude == "small":
                                        prior_mean = sign * 0.2
                                    elif magnitude == "moderate":
                                        prior_mean = sign * 0.5
                                    elif magnitude == "large":
                                        prior_mean = sign * 1.0
                                    else: # default to small
                                        prior_mean = sign * 0.2

                                # Map confidence to uncertainty
                                if confidence < 0.4:
                                    prior_sd = 2.0  # Low confidence
                                elif confidence < 0.75:
                                    prior_sd = 1.0  # Moderate confidence
                                else:
                                    prior_sd = 0.5  # High confidence
                                
                                return prior_mean, prior_sd

                            # --- Hierarchical Prior Generation ---
                            def hierarchical_prior(prediction, magnitude):
                                """Generates group assignments for hierarchical model."""
                                if prediction == "unchanged":
                                    return "unchanged"
                                return f"{magnitude}_{prediction}"

                            # Choose prior generation based on strength
                            if prior_strength == "conservative":
                                expected_log2fc, prior_sd = conservative_directional_prior(
                                    prediction, confidence
                                )
                                group = None
                            elif prior_strength == "moderate":
                                expected_log2fc, prior_sd = moderate_directional_prior(
                                    prediction, magnitude, confidence
                                )
                                group = None
                            elif prior_strength == "hierarchical":
                                group = hierarchical_prior(prediction, magnitude)
                                expected_log2fc, prior_sd = None, None # Not used in this mode
                            else:
                                raise ValueError(f"Unknown prior_strength: {prior_strength}")


                            # Create result object
                            return {
                                "metabolite": metabolite,
                                "prediction": prediction,
                                "magnitude": magnitude,
                                "confidence": confidence,
                                "reasoning": reasoning,
                                "expected_log2fc": expected_log2fc,
                                "prior_sd": prior_sd,
                                "group": group,
                            }

                        except Exception as e:
                            print(f"Error scoring {metabolite}: {e}", file=sys.stderr)
                            # Default result for failed analysis
                            return {
                                "metabolite": metabolite,
                                "prediction": "unchanged",
                                "confidence": 0.0,
                                "reasoning": f"Error in LLM processing: {str(e)[:100]}",
                                "expected_log2fc": 0.0,
                                "prior_sd": 1.5,
                                "group": "unchanged",
                            }

                llm_scorer = DirectGeminiScorer(model)
            except Exception as e:
                raise RuntimeError(
                    f"Could not create Gemini scorer: {e}. Ensure GOOGLE_API_KEY is set and google-generativeai is installed."
                ) from e
        else:
            raise ValueError("GOOGLE_API_KEY environment variable is required.")

    # Process metabolites individually (paid tier - no rate limiting needed)
    differential_priors = {}

    print(f"Processing {len(metabolites)} metabolites individually", file=sys.stderr)

    for i, metabolite in enumerate(metabolites, 1):
        print(
            f"Processing metabolite {i}/{len(metabolites)}: {metabolite}",
            file=sys.stderr,
        )

        if metabolite not in batch_contexts:
            print(f"Warning: No context available for {metabolite}", file=sys.stderr)
            differential_priors[metabolite] = {
                "relevance": 0.5,
                "direction": "unclear",
                "rationale": "No context available",
                "expected_log2fc": 0.0,
                "prior_sd": 0.8,
                "magnitude": "moderate",
                "confidence": "low",
                "group": "unchanged",
            }
            continue

        try:
            # Process single metabolite
            result = llm_scorer.analyze_metabolite(
                condition, metabolite, batch_contexts[metabolite]
            )

            differential_priors[metabolite] = result

        except Exception as e:
            print(f"Error processing {metabolite}: {e}", file=sys.stderr)
            differential_priors[metabolite] = {
                "relevance": 0.5,
                "direction": "unclear",
                "rationale": f"Error in LLM processing: {str(e)[:100]}",
                "expected_log2fc": 0.0,
                "prior_sd": 0.8,
                "magnitude": "moderate",
                "confidence": "low",
                "group": "unchanged",
            }

    return differential_priors


def get_network_priors(
    priors: PriorData, threshold: float = 0.7
) -> Dict[Tuple[str, str], float]:
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
        raise ValueError(
            "No similarity matrix available. Run create_similarity_matrix() first."
        )

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


def save_results(
    priors: PriorData,
    output_dir: str = ".",
    differential_priors: Optional[Dict[str, float]] = None,
    network_priors: Optional[Dict[Tuple[str, str], float]] = None,
) -> Dict[str, str]:
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
        smiles_path = os.path.join(output_dir, "metabolite_smiles.csv")
        priors.smiles_data.to_csv(smiles_path, index=False)
        files["smiles"] = smiles_path

    if priors.fingerprints_data is not None:
        fp_path = os.path.join(output_dir, "metabolite_fingerprints.csv")
        # Save fingerprints as strings for CSV compatibility
        fp_df = priors.fingerprints_data.copy()
        fp_df["map4_fingerprint"] = fp_df["map4_fingerprint"].apply(
            lambda x: ",".join(map(str, x)) if x is not None else None
        )
        fp_df.to_csv(fp_path, index=False)
        files["fingerprints"] = fp_path

    if priors.similarity_matrix is not None and priors.metabolite_names is not None:
        sim_path = os.path.join(output_dir, "metabolite_similarity_matrix.csv")
        sim_df = pd.DataFrame(
            priors.similarity_matrix,
            index=priors.metabolite_names,
            columns=priors.metabolite_names,
        )
        sim_df.to_csv(sim_path)
        files["similarity_matrix"] = sim_path

    if priors.hmdb_contexts is not None:
        hmdb_path = os.path.join(output_dir, "hmdb_contexts.csv")
        hmdb_df = pd.DataFrame(
            [
                {"metabolite": metabolite, "hmdb_context": context}
                for metabolite, context in priors.hmdb_contexts.items()
            ]
        )
        hmdb_df.to_csv(hmdb_path, index=False)
        files["hmdb_contexts"] = hmdb_path

    # Save differential priors if provided
    if differential_priors is not None:
        diff_path = os.path.join(output_dir, "differential_priors.csv")
        diff_df = pd.DataFrame(
            [
                {"metabolite": metabolite, "importance_score": score}
                for metabolite, score in differential_priors.items()
            ]
        )
        diff_df.to_csv(diff_path, index=False)
        files["differential_priors"] = diff_path

    # Save network priors if provided
    if network_priors is not None:
        network_path = os.path.join(output_dir, "network_priors.csv")
        network_df = pd.DataFrame(
            [
                {
                    "metabolite_1": pair[0],
                    "metabolite_2": pair[1],
                    "similarity_score": score,
                }
                for pair, score in network_priors.items()
            ]
        )
        network_df.to_csv(network_path, index=False)
        files["network_priors"] = network_path

    return files


def run_pipeline(
    dimensions: int = 1024,
    metabolites: Optional[List[str]] = None,
    excel_files: Optional[Union[str, List[str]]] = None,
    max_workers: int = 4,
    output_dir: Optional[str] = None,
    include_hmdb: bool = True,
    hmdb_mapping: Optional[Dict[str, str]] = None,
) -> PriorData:
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
