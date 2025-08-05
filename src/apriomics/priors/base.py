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


def get_llm_qualitative_predictions(
    priors: PriorData,
    condition: str,
    llm_scorer=None,
    hmdb_retriever=None,
    use_hmdb_context: bool = True,
    model_name: str = "gemini-2.5-flash-lite-preview-06-17",
    temperature: float = 0.0,
) -> Dict[str, Dict]:
    """
    Generate qualitative LLM predictions for differential expression analysis.

    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts
    condition : str
        Study condition or experimental design (e.g., "diabetes vs control")
    llm_scorer : object, optional
        LLM scorer object.
    hmdb_retriever : HMDBRetriever, optional
        RAG-based HMDB retriever for enhanced context. If provided, replaces hmdb_contexts.
    use_hmdb_context : bool
        Whether to use HMDB context information
    model_name : str
        LLM model to use

    Returns:
    --------
    dict
        Dictionary mapping metabolite names to qualitative predictions:
        - 'prediction': str - expected regulation direction ('increase', 'decrease', 'unchanged')
        - 'magnitude': str - effect size category ('small', 'moderate', 'large')
        - 'confidence': float (0-1) - assessment confidence
        - 'reasoning': str - explanation for the assessment
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
        # Check if it's an OpenAI model
        if model_name.startswith(("gpt-", "o1-", "o3-", "o4-")):
            if os.getenv("OPENAI_API_KEY"):
                try:
                    import openai
                    import json
                    import re

                    # Configure OpenAI client
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    # Check if it's a deep research model
                    is_deep_research = "deep-research" in model_name

                    class DirectOpenAIDeepResearchScorer:
                        def __init__(self, client, model_name):
                            self.client = client
                            self.model_name = model_name

                        def analyze_metabolite(
                            self, condition: str, metabolite: str, context: str
                        ):
                            """Analyze a single metabolite using OpenAI Deep Research models."""

                            system_message = """
You are an expert metabolomics researcher with deep knowledge of biochemical pathways, disease mechanisms, and metabolic regulation. Your expertise spans diabetes metabolism, renal physiology, and urinary biomarkers.

Conduct thorough research to predict how the given metabolite's urinary concentration will change in the study condition. Use web search to find the latest literature and evidence to support your analysis.
"""

                            user_query = f"""
<study_context>
{condition}
</study_context>

<metabolite>
{metabolite}
</metabolite>

Research and predict how this metabolite's urinary concentration will change in the study condition.

<magnitude_calibration>
For metabolomics studies, effect sizes typically range:
- **Large**: >50% change (log2FC > 0.7) - major pathway disruption, central metabolites
- **Moderate**: 20-50% change (log2FC 0.3-0.7) - significant but not dramatic pathway effects  
- **Small**: 5-20% change (log2FC 0.1-0.3) - subtle regulatory changes, downstream effects
</magnitude_calibration>

<confidence_calibration>
Rate your certainty about the **direction** of change:
- **0.9-1.0**: Well-established in diabetes literature, clear single mechanism
- **0.7-0.9**: Strong biological rationale, some supporting evidence
- **0.5-0.7**: Plausible mechanism, but limited evidence or competing pathways
- **0.3-0.5**: Weak evidence, highly speculative, or strong conflicting mechanisms
- **0.1-0.3**: No clear mechanism, pure speculation
</confidence_calibration>

Provide your final answer in JSON format only.
{{
    "prediction": "<increase|decrease|unchanged>",
    "magnitude": "<small|moderate|large>",
    "confidence": <0.1-1.0>,
    "reasoning": "<Concise summary of your biological rationale and key evidence>"
}}
"""

                            try:
                                response = self.client.responses.create(
                                    model=self.model_name,
                                    input=[
                                        {
                                            "role": "developer",
                                            "content": [
                                                {
                                                    "type": "input_text",
                                                    "text": system_message,
                                                }
                                            ],
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "input_text",
                                                    "text": user_query,
                                                }
                                            ],
                                        },
                                    ],
                                    reasoning={"summary": "auto"},
                                    tools=[{"type": "web_search_preview"}],
                                )

                                # Extract the final message content from responses API
                                # Find the message item in the output (skip reasoning items)
                                message_item = None
                                for item in response.output:
                                    if hasattr(item, "type") and item.type == "message":
                                        message_item = item
                                        break

                                if (
                                    message_item
                                    and hasattr(message_item, "content")
                                    and len(message_item.content) > 0
                                ):
                                    response_text = message_item.content[0].text
                                else:
                                    # Fallback - try first item if it has content
                                    if len(response.output) > 0 and hasattr(
                                        response.output[0], "content"
                                    ):
                                        response_text = (
                                            response.output[0].content[0].text
                                        )
                                    else:
                                        response_text = (
                                            "No message content found in response"
                                        )

                                # Clean up the response to extract JSON
                                json_match = re.search(
                                    r"\{.*\}", response_text, re.DOTALL
                                )
                                if json_match:
                                    json_str = json_match.group(0)
                                else:
                                    json_str = response_text.strip()

                                # Remove markdown code blocks if present
                                json_str = re.sub(r"```json\s*", "", json_str)
                                json_str = re.sub(r"```\s*$", "", json_str)

                                # Parse JSON response with better error handling
                                try:
                                    response_data = json.loads(json_str)
                                except json.JSONDecodeError as json_err:
                                    print(
                                        f"JSON parsing failed for {metabolite}: {json_err}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Raw response (first 500 chars): {response_text[:500]}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Cleaned JSON string: {json_str[:500]}",
                                        file=sys.stderr,
                                    )
                                    raise json_err

                                # Extract values with defaults
                                prediction = response_data.get(
                                    "prediction", "unchanged"
                                ).lower()
                                magnitude = response_data.get(
                                    "magnitude", "small"
                                ).lower()
                                confidence = float(response_data.get("confidence", 0.5))
                                reasoning = response_data.get(
                                    "reasoning", "No reasoning provided"
                                )

                                # Return only qualitative predictions
                                return {
                                    "prediction": prediction,
                                    "magnitude": magnitude,
                                    "confidence": confidence,
                                    "reasoning": reasoning,
                                }

                            except Exception as e:
                                print(
                                    f"Error scoring {metabolite}: {e}", file=sys.stderr
                                )
                                # Default result for failed analysis
                                return {
                                    "prediction": "unchanged",
                                    "magnitude": "small",
                                    "confidence": 0.0,
                                    "reasoning": f"Error in LLM processing: {str(e)[:100]}",
                                }

                    class DirectOpenAIScorer:
                        def __init__(self, client, model_name, temperature):
                            self.client = client
                            self.model_name = model_name
                            self.temperature = temperature

                        def analyze_metabolite(
                            self, condition: str, metabolite: str, context: str
                        ):
                            """Analyze a single metabolite using OpenAI models."""

                            prompt = f"""
<role>
You are an expert metabolomics researcher with deep knowledge of biochemical pathways, disease mechanisms, and metabolic regulation. Your expertise spans diabetes metabolism, renal physiology, and urinary biomarkers.
</role>

<task>
Predict how the given metabolite's urinary concentration will change in the study condition. Focus on biological mechanisms rather than statistical associations.
</task>

<study_context>
{condition}
</study_context>

<metabolite_information>
{context}
</metabolite_information>

<magnitude_calibration>
For metabolomics studies, effect sizes typically range:
- **Large**: >50% change (log2FC > 0.7) - major pathway disruption, central metabolites
- **Moderate**: 20-50% change (log2FC 0.3-0.7) - significant but not dramatic pathway effects  
- **Small**: 5-20% change (log2FC 0.1-0.3) - subtle regulatory changes, downstream effects

Examples from diabetes literature:
- Large: Glucose, HbA1c, ketone bodies (central energy metabolism)
- Moderate: BCAA metabolites, inflammatory markers (secondary pathways)
- Small: Minor amino acid derivatives, trace pathway outputs
</magnitude_calibration>

<direction_examples>
In diabetes, metabolites commonly:
**Increase**: Glucose spillover products, BCAA catabolites, oxidative stress markers, inflammatory compounds
**Decrease**: Antioxidants (glutathione), some vitamins, efficiently cleared waste products, beneficial gut metabolites
**Consider both directions** - diabetes involves depletion as well as accumulation.
</direction_examples>

<analysis_framework>
Think step by step:

1. **Pathway Context**: What biochemical pathways produce/consume this metabolite?
2. **Disease Mechanism**: How does diabetes specifically affect these pathways?
3. **Directional Logic**: Would diabetes increase production, decrease clearance, enhance consumption, or alter regulation?
4. **Magnitude Reasoning**: Is this metabolite central to diabetes pathophysiology or peripheral? How dramatic would the change be?
5. **Evidence Assessment**: How strong is the biological rationale? Are there conflicting mechanisms?
</analysis_framework>

<confidence_calibration>
Rate your certainty about the **direction** of change:
- **0.9-1.0**: Well-established in diabetes literature, clear single mechanism
- **0.7-0.9**: Strong biological rationale, some supporting evidence
- **0.5-0.7**: Plausible mechanism, but limited evidence or competing pathways
- **0.3-0.5**: Weak evidence, highly speculative, or strong conflicting mechanisms
- **0.1-0.3**: No clear mechanism, pure speculation
</confidence_calibration>

<examples>
<example_1>
Metabolite: 3-hydroxybutyrate
Reasoning: Central ketone body produced during fatty acid oxidation. Diabetes enhances lipolysis and ketogenesis due to insulin deficiency/resistance. Well-documented diabetes biomarker.
Prediction: increase, magnitude: large, confidence: 0.95
</example_1>

<example_2>
Metabolite: Glutathione
Reasoning: Major antioxidant that becomes depleted under oxidative stress. Diabetes increases reactive oxygen species while potentially reducing glutathione synthesis. Supported by literature.
Prediction: decrease, magnitude: moderate, confidence: 0.8
</example_2>

<example_3>
Metabolite: Minor tryptophan metabolite
Reasoning: Downstream product of tryptophan metabolism. Diabetes may slightly alter gut microbiome affecting tryptophan processing, but mechanism unclear and effect likely subtle.
Prediction: increase, magnitude: small, confidence: 0.4
</example_3>
</examples>

<output_format>
Provide your final answer in JSON format only.
{{
    "prediction": "<increase|decrease|unchanged>",
    "magnitude": "<small|moderate|large>",
    "confidence": <0.1-1.0>,
    "reasoning": "<Concise summary of your biological rationale and key evidence>"
}}
</output_format>

<critical_reminders>
- Predict BOTH increases AND decreases - diabetes involves depletion as well as accumulation
- Use the full confidence range (0.1-1.0) - be appropriately uncertain for unclear cases
- Use "large" magnitude for central diabetes-related metabolites (aim for 10-20% of predictions)
- Consider mechanism strength, not just direction plausibility
</critical_reminders>
"""

                            try:
                                # Handle different parameter names for different models
                                if self.model_name.startswith(("o1-", "o3-")):
                                    # o3 models use max_completion_tokens and do not support temperature
                                    # Increase token limit for verbose o3 models
                                    response = self.client.chat.completions.create(
                                        model=self.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_completion_tokens=4096,
                                    )
                                else:
                                    # Standard models use max_tokens
                                    response = self.client.chat.completions.create(
                                        model=self.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=self.temperature,
                                        max_tokens=1000,
                                    )

                                response_text = response.choices[
                                    0
                                ].message.content.strip()

                                # Clean up the response to extract JSON
                                json_match = re.search(
                                    r"\{.*\}", response_text, re.DOTALL
                                )
                                if json_match:
                                    json_str = json_match.group(0)
                                else:
                                    json_str = response_text.strip()

                                # Remove markdown code blocks if present
                                json_str = re.sub(r"```json\s*", "", json_str)
                                json_str = re.sub(r"```\s*$", "", json_str)

                                # Parse JSON response with better error handling
                                try:
                                    response_data = json.loads(json_str)
                                except json.JSONDecodeError as json_err:
                                    print(
                                        f"JSON parsing failed for {metabolite}: {json_err}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Raw response (first 500 chars): {response_text[:500]}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Cleaned JSON string: {json_str[:500]}",
                                        file=sys.stderr,
                                    )
                                    raise json_err

                                # Extract values with defaults
                                prediction = response_data.get(
                                    "prediction", "unchanged"
                                ).lower()
                                magnitude = response_data.get(
                                    "magnitude", "small"
                                ).lower()
                                confidence = float(response_data.get("confidence", 0.5))
                                reasoning = response_data.get(
                                    "reasoning", "No reasoning provided"
                                )

                                # Return only qualitative predictions
                                return {
                                    "prediction": prediction,
                                    "magnitude": magnitude,
                                    "confidence": confidence,
                                    "reasoning": reasoning,
                                }

                            except Exception as e:
                                print(
                                    f"Error scoring {metabolite}: {e}", file=sys.stderr
                                )
                                # Default result for failed analysis
                                return {
                                    "prediction": "unchanged",
                                    "magnitude": "small",
                                    "confidence": 0.0,
                                    "reasoning": f"Error in LLM processing: {str(e)[:100]}",
                                }

                    # Use appropriate scorer based on model type
                    if is_deep_research:
                        llm_scorer = DirectOpenAIDeepResearchScorer(client, model_name)
                    else:
                        llm_scorer = DirectOpenAIScorer(client, model_name, temperature)

                except Exception as e:
                    raise RuntimeError(
                        f"Could not create OpenAI scorer: {e}. Ensure OPENAI_API_KEY is set and openai is installed."
                    ) from e
            else:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI models."
                )

        else:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI models."
            )

    # Process metabolites individually (paid tier - no rate limiting needed)
    qualitative_predictions = {}

    print(f"Processing {len(metabolites)} metabolites individually", file=sys.stderr)

    for i, metabolite in enumerate(metabolites, 1):
        print(
            f"Processing metabolite {i}/{len(metabolites)}: {metabolite}",
            file=sys.stderr,
        )

        if metabolite not in batch_contexts:
            print(f"Warning: No context available for {metabolite}", file=sys.stderr)
            qualitative_predictions[metabolite] = {
                "prediction": "unchanged",
                "magnitude": "small",
                "confidence": 0.0,
                "reasoning": "No context available",
            }
            continue

        try:
            # Process single metabolite
            result = llm_scorer.analyze_metabolite(
                condition, metabolite, batch_contexts[metabolite]
            )

            qualitative_predictions[metabolite] = result

        except Exception as e:
            print(f"Error processing {metabolite}: {e}", file=sys.stderr)
            qualitative_predictions[metabolite] = {
                "prediction": "unchanged",
                "magnitude": "small",
                "confidence": 0.0,
                "reasoning": f"Error in LLM processing: {str(e)[:100]}",
            }

    return qualitative_predictions


def get_llm_quantitative_priors(
    priors: PriorData,
    condition: str,
    llm_scorer=None,
    hmdb_retriever=None,
    use_hmdb_context: bool = True,
    model_name: str = "gpt-4o-2024-08-06",
    temperature: float = 0.0,
) -> Dict[str, Dict]:
    """
    Generate direct quantitative LLM priors for differential expression analysis.

    Instead of qualitative predictions, directly asks the LLM for numerical estimates
    of log fold change and uncertainty.

    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts
    condition : str
        Study condition or experimental design (e.g., "diabetes vs control")
    llm_scorer : object, optional
        LLM scorer object. If None, creates appropriate scorer.
    hmdb_retriever : HMDBRetriever, optional
        RAG-based HMDB retriever for enhanced context.
    use_hmdb_context : bool
        Whether to use HMDB context information
    model_name : str
        LLM model to use
    temperature : float
        LLM temperature setting

    Returns:
    --------
    dict
        Dictionary mapping metabolite names to quantitative predictions:
        - 'expected_lnfc': float - expected natural log fold change
        - 'prior_sd': float - standard deviation of the prior distribution
        - 'confidence': float (0-1) - assessment confidence
        - 'reasoning': str - explanation for the assessment
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
        # Check if it's an OpenAI model
        if model_name.startswith(("gpt-", "o1-", "o3-", "o4-")):
            if os.getenv("OPENAI_API_KEY"):
                try:
                    import openai
                    import json
                    import re

                    # Configure OpenAI client
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                    class DirectOpenAIQuantitativeScorer:
                        def __init__(self, client, model_name, temperature):
                            self.client = client
                            self.model_name = model_name
                            self.temperature = temperature

                        def analyze_metabolite(
                            self, condition: str, metabolite: str, context: str
                        ):
                            """Analyze a single metabolite for direct quantitative prediction."""

                            prompt = f"""
<role>
You are an expert metabolomics researcher with deep knowledge of biochemical pathways, disease mechanisms, and metabolic regulation. Your expertise spans diabetes metabolism, renal physiology, and urinary biomarkers.
</role>

<task>
Provide a direct numerical estimate of the expected natural log fold change (ln(case/control)) for this metabolite's urinary concentration in the study condition. Also estimate your uncertainty about this prediction.
</task>

<study_context>
{condition}
</study_context>

<metabolite_information>
{context}
</metabolite_information>

<quantitative_calibration>
Natural log fold change (ln FC) interpretation:
- **ln FC = 0**: No change (case = control)
- **ln FC = +0.1**: ~10% increase (small effect)
- **ln FC = +0.3**: ~35% increase (moderate effect) 
- **ln FC = +0.5**: ~65% increase (large effect)
- **ln FC = +0.7**: ~100% increase (very large effect)
- **ln FC = -0.1**: ~10% decrease (small effect)
- **ln FC = -0.3**: ~25% decrease (moderate effect)
- **ln FC = -0.5**: ~40% decrease (large effect)

Typical metabolomics effect sizes:
- Most metabolites: ln FC between -0.2 and +0.2
- Moderately affected: ln FC between -0.5 and +0.5  
- Strongly affected: ln FC beyond ±0.5 (rare, <5% of metabolites)
</quantitative_calibration>

<uncertainty_calibration>
Prior standard deviation (SD) represents your uncertainty:
- **SD = 0.1**: Very confident, tight prior (±0.2 covers ~95% of belief)
- **SD = 0.3**: Moderately confident (±0.6 covers ~95% of belief)
- **SD = 0.5**: Somewhat uncertain (±1.0 covers ~95% of belief)
- **SD = 0.8**: High uncertainty, weak prior (±1.6 covers ~95% of belief)

Choose SD based on:
- Literature evidence strength
- Mechanism clarity
- Pathway centrality to disease
- Potential confounding factors
</uncertainty_calibration>

<analysis_framework>
Think step by step:

1. **Pathway Analysis**: What biochemical pathways involve this metabolite?
2. **Disease Mechanism**: How does diabetes specifically affect these pathways?
3. **Quantitative Reasoning**: Based on biological mechanisms, estimate the magnitude and direction of change
4. **Literature Calibration**: Are there studies showing similar metabolites' fold changes?
5. **Uncertainty Assessment**: How confident are you in this estimate given available evidence?

KEY: Provide your best numerical estimate, not just direction. Use mechanism strength to determine both effect size and uncertainty.
</analysis_framework>

<examples>
<example_1>
Metabolite: 3-hydroxybutyrate
Reasoning: Central ketone body in diabetes. Well-established 2-5x increase in diabetic patients due to enhanced lipolysis and ketogenesis. Strong mechanistic evidence.
Expected ln FC: +1.2 (corresponds to ~3x increase)
Prior SD: 0.3 (moderately tight due to strong evidence)
Confidence: 0.9
</example_1>

<example_2>
Metabolite: Glutathione  
Reasoning: Antioxidant depleted under oxidative stress. Diabetes literature shows moderate decreases (20-40%). Mechanism clear but effect size variable.
Expected ln FC: -0.4 (corresponds to ~33% decrease)
Prior SD: 0.4 (moderate uncertainty due to variable effect sizes)
Confidence: 0.7
</example_2>

<example_3>
Metabolite: Histidine
Reasoning: Essential amino acid from diet. No direct diabetes mechanism. Might have subtle changes due to dietary/absorption differences but unclear direction and magnitude.
Expected ln FC: 0.0 (no compelling evidence for change)
Prior SD: 0.6 (high uncertainty, weak prior)
Confidence: 0.3
</example_3>
</examples>

<output_format>
Provide your final answer in JSON format only.
{{
    "expected_lnfc": <numerical value>,
    "prior_sd": <numerical value>,
    "confidence": <0.0-1.0>,
    "reasoning": "<Concise summary of your quantitative reasoning and key evidence>"
}}
</output_format>

<critical_reminders>
- Provide specific numerical estimates, not just directions
- Use mechanism strength to determine both effect size AND uncertainty
- Most metabolites have small effects (ln FC between -0.2 and +0.2)
- Higher uncertainty (larger SD) for weaker evidence
- Consider both increases AND decreases based on biological mechanisms
</critical_reminders>
"""

                            try:
                                # Handle different parameter names for different models
                                if self.model_name.startswith(("o1-", "o3-")):
                                    # o3 models use max_completion_tokens and do not support temperature
                                    # Increase token limit for verbose o3 models
                                    response = self.client.chat.completions.create(
                                        model=self.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_completion_tokens=4096,
                                    )
                                else:
                                    # Standard models use max_tokens
                                    response = self.client.chat.completions.create(
                                        model=self.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=self.temperature,
                                        max_tokens=4096,
                                    )

                                response_text = response.choices[
                                    0
                                ].message.content.strip()

                                # Clean up the response to extract JSON
                                json_match = re.search(
                                    r"\{.*\}", response_text, re.DOTALL
                                )
                                if json_match:
                                    json_str = json_match.group(0)
                                else:
                                    json_str = response_text.strip()

                                # Remove markdown code blocks if present
                                json_str = re.sub(r"```json\s*", "", json_str)
                                json_str = re.sub(r"```\s*$", "", json_str)

                                # Parse JSON response with better error handling
                                try:
                                    response_data = json.loads(json_str)
                                except json.JSONDecodeError as json_err:
                                    print(
                                        f"JSON parsing failed for {metabolite}: {json_err}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Raw response (first 500 chars): {response_text[:500]}",
                                        file=sys.stderr,
                                    )
                                    print(
                                        f"Cleaned JSON string: {json_str[:500]}",
                                        file=sys.stderr,
                                    )
                                    raise json_err

                                # Extract values with defaults
                                expected_lnfc = float(
                                    response_data.get("expected_lnfc", 0.0)
                                )
                                prior_sd = float(response_data.get("prior_sd", 0.6))
                                confidence = float(response_data.get("confidence", 0.5))
                                reasoning = response_data.get(
                                    "reasoning", "No reasoning provided"
                                )

                                # Validate ranges
                                expected_lnfc = max(
                                    min(expected_lnfc, 3.0), -3.0
                                )  # Cap at reasonable range
                                prior_sd = max(
                                    min(prior_sd, 2.0), 0.05
                                )  # Reasonable SD range
                                confidence = max(min(confidence, 1.0), 0.0)  # 0-1 range

                                return {
                                    "expected_lnfc": expected_lnfc,
                                    "prior_sd": prior_sd,
                                    "confidence": confidence,
                                    "reasoning": reasoning,
                                }

                            except Exception as e:
                                print(
                                    f"Error scoring {metabolite}: {e}", file=sys.stderr
                                )
                                # Default result for failed analysis
                                return {
                                    "expected_lnfc": 0.0,
                                    "prior_sd": 0.8,
                                    "confidence": 0.0,
                                    "reasoning": f"Error in LLM processing: {str(e)[:100]}",
                                }

                    llm_scorer = DirectOpenAIQuantitativeScorer(
                        client, model_name, temperature
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Could not create OpenAI scorer: {e}. Ensure OPENAI_API_KEY is set and openai is installed."
                    ) from e
            else:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI models."
                )

        else:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI models."
            )
    # Process metabolites individually
    quantitative_predictions = {}

    print(
        f"Processing {len(metabolites)} metabolites for quantitative predictions",
        file=sys.stderr,
    )

    for i, metabolite in enumerate(metabolites, 1):
        print(
            f"Processing metabolite {i}/{len(metabolites)}: {metabolite}",
            file=sys.stderr,
        )

        if metabolite not in batch_contexts:
            print(f"Warning: No context available for {metabolite}", file=sys.stderr)
            quantitative_predictions[metabolite] = {
                "expected_lnfc": 0.0,
                "prior_sd": 0.8,
                "confidence": 0.0,
                "reasoning": "No context available",
            }
            continue

        try:
            # Process single metabolite
            result = llm_scorer.analyze_metabolite(
                condition, metabolite, batch_contexts[metabolite]
            )

            quantitative_predictions[metabolite] = result

        except Exception as e:
            print(f"Error processing {metabolite}: {e}", file=sys.stderr)
            quantitative_predictions[metabolite] = {
                "expected_lnfc": 0.0,
                "prior_sd": 0.8,
                "confidence": 0.0,
                "reasoning": f"Error in LLM processing: {str(e)[:100]}",
            }

    return quantitative_predictions


def map_qualitative_to_numerical_priors(
    qualitative_predictions: Dict[str, Dict], prior_strength: str = "conservative"
) -> Dict[str, Dict]:
    """
    Map qualitative LLM predictions to numerical priors based on strength.

    Parameters:
    -----------
    qualitative_predictions : dict
        Dictionary mapping metabolite names to qualitative predictions
    prior_strength : str
        Prior strength ("conservative", "moderate", or "strong")

    Returns:
    --------
    dict
        Dictionary mapping metabolite names to numerical prior information
    """

    def conservative_directional_prior(prediction, magnitude, confidence):
        """Conservative prior: magnitude drives mean, confidence drives uncertainty."""
        # Magnitude-based effect sizes (conservative scaling)
        magnitude_effects = {"small": 0.08, "moderate": 0.15, "large": 0.25}

        base_effect = magnitude_effects.get(magnitude, 0.10)  # Default to moderate

        if prediction == "increase":
            prior_mean = base_effect
        elif prediction == "decrease":
            prior_mean = -base_effect
        else:
            prior_mean = 0.0

        # Confidence-based uncertainty (conservative: wider overall)
        if confidence > 0.8:
            prior_sd = 0.5  # High confidence → moderate uncertainty
        elif confidence > 0.6:
            prior_sd = 0.7  # Medium confidence → higher uncertainty
        else:
            prior_sd = 0.9  # Low confidence → high uncertainty

        return prior_mean, prior_sd

    def moderate_directional_prior(prediction, magnitude, confidence):
        """Moderate prior: magnitude drives mean, confidence drives uncertainty."""
        # Magnitude-based effect sizes (moderate scaling)
        magnitude_effects = {"small": 0.12, "moderate": 0.22, "large": 0.35}

        base_effect = magnitude_effects.get(magnitude, 0.15)  # Default to moderate

        if prediction == "increase":
            prior_mean = base_effect
        elif prediction == "decrease":
            prior_mean = -base_effect
        else:
            prior_mean = 0.0

        # Confidence-based uncertainty (moderate: tighter than conservative)
        if confidence > 0.8:
            prior_sd = 0.3  # High confidence → tight uncertainty
        elif confidence > 0.6:
            prior_sd = 0.5  # Medium confidence → moderate uncertainty
        else:
            prior_sd = 0.7  # Low confidence → wider uncertainty

        return prior_mean, prior_sd

    def strong_directional_prior(prediction, confidence):
        """Strong prior: direction + confidence, larger effect sizes with tight uncertainty."""
        base_effect = 0.15  # Medium effect size (still reasonable for metabolomics)

        if prediction == "increase":
            prior_mean = base_effect * confidence
        elif prediction == "decrease":
            prior_mean = -base_effect * confidence
        else:
            prior_mean = 0.0

        # Strong: tight uncertainty, especially for high confidence
        if confidence < 0.4:
            prior_sd = 0.6  # Some uncertainty
        elif confidence < 0.7:
            prior_sd = 0.3  # Low uncertainty
        else:
            prior_sd = 0.2  # Very tight

        return prior_mean, prior_sd

    numerical_priors = {}

    for metabolite, qual_pred in qualitative_predictions.items():
        prediction = qual_pred["prediction"]
        magnitude = qual_pred["magnitude"]
        confidence = qual_pred["confidence"]
        reasoning = qual_pred["reasoning"]

        # Choose mapping based on strength (now use magnitude + confidence)
        if prior_strength == "conservative":
            expected_lnfc, prior_sd = conservative_directional_prior(
                prediction, magnitude, confidence
            )
        elif prior_strength == "moderate":
            expected_lnfc, prior_sd = moderate_directional_prior(
                prediction, magnitude, confidence
            )
        elif prior_strength == "strong":
            expected_lnfc, prior_sd = strong_directional_prior(prediction, confidence)
        else:
            raise ValueError(
                f"Unknown prior_strength: {prior_strength}. Must be 'conservative', 'moderate', or 'strong'."
            )

        numerical_priors[metabolite] = {
            "prediction": prediction,
            "magnitude": magnitude,
            "confidence": confidence,
            "reasoning": reasoning,
            "expected_lnfc": expected_lnfc,
            "prior_sd": prior_sd,
        }

    return numerical_priors


def get_llm_differential_priors(
    priors: PriorData,
    condition: str,
    llm_scorer=None,
    hmdb_retriever=None,
    use_hmdb_context: bool = True,
    prior_strength: str = "conservative",
    model_name: str = "gemini-2.5-flash-lite-preview-06-17",
    qualitative_predictions: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Generate LLM-informed numerical priors for differential expression analysis.

    If qualitative_predictions are provided, uses them directly and applies numerical mapping.
    Otherwise, generates qualitative predictions first then applies mapping.

    Parameters:
    -----------
    priors : PriorData
        Data container with HMDB contexts
    condition : str
        Study condition or experimental design
    llm_scorer : object, optional
        LLM scorer object
    hmdb_retriever : HMDBRetriever, optional
        RAG-based HMDB retriever for enhanced context
    use_hmdb_context : bool
        Whether to use HMDB context information
    prior_strength : str
        Prior strength ("conservative", "moderate", or "strong")
    model_name : str
        LLM model to use
    qualitative_predictions : dict, optional
        Pre-generated qualitative predictions to reuse

    Returns:
    --------
    dict
        Dictionary mapping metabolite names to numerical prior information
    """
    if qualitative_predictions is None:
        # Generate qualitative predictions
        qualitative_predictions = get_llm_qualitative_predictions(
            priors, condition, llm_scorer, hmdb_retriever, use_hmdb_context, model_name
        )

    # Map to numerical priors
    return map_qualitative_to_numerical_priors(qualitative_predictions, prior_strength)


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
