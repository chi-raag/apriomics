"""
Benchmark: Recovery of ground truth natural log fold change (lnFC) using different LLM prior methods.

This script evaluates how well LLM priors help recover true lnFC estimates
from the full MTBLS1 dataset when given limited subsampled data.

Compares:
1. Bayesian + LLM priors (no HMDB context)
2. Bayesian + LLM priors (with HMDB context)

Ground truth: empirical lnFC from full dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
import pickle
from apriomics.priors import PriorData, get_llm_qualitative_predictions
from chembridge.databases.hmdb import HMDBClient
from chembridge import map_metabolites
import logging
from tqdm import tqdm

warnings.filterwarnings("ignore")
# Suppress PyMC verbose output - only show progress bars
logging.getLogger("pymc").setLevel(logging.WARNING)
logging.getLogger("pytensor").setLevel(logging.WARNING)


def load_mtbls1_data(cache_dir="/Users/chiraag/Projects/gwu/lab/apriomics/output"):
    """Load MTBLS1 dataset with caching for HMDB mapping."""

    data_path = "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv"
    metadata_path = (
        "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/s_MTBLS1.txt"
    )
    cache_path = os.path.join(cache_dir, "mtbls1_hmdb_mapped_data.pkl")

    # Check if files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    os.makedirs(cache_dir, exist_ok=True)

    # Try to load cached mapped data
    if os.path.exists(cache_path):
        print(f"Loading cached HMDB-mapped data from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            print("✅ Using cached HMDB mapping")
            filtered_data = cached_data["filtered_data"]
        except Exception as e:
            print(f"⚠️  Error loading cache: {e} - regenerating")
            filtered_data = None
    else:
        filtered_data = None

    # Always load metadata (needed for sample alignment)
    print("Loading metadata...")
    try:
        metadata = pd.read_csv(metadata_path, sep="\t")
        print(f"✅ Loaded metadata: {len(metadata)} samples")
    except Exception as e:
        raise RuntimeError(f"Error loading metadata: {e}")

    if filtered_data is None:
        print("Loading data files...")
        try:
            data = pd.read_csv(data_path, sep="\t")
            print(f"✅ Loaded data: {len(data)} metabolites")
        except Exception as e:
            raise RuntimeError(f"Error loading data files: {e}")

        print("Mapping metabolite names to HMDB IDs...")
        try:
            # Map metabolite names (like in qmd)
            mapped_data = map_metabolites(
                data["database_identifier"].tolist(), source="chebi", target="hmdb"
            )
            # Use the correct .df() method from MappingResults
            data["hmdb_id"] = mapped_data.df()["hmdb_id"]
            filtered_data = data[data["hmdb_id"].notna()]
            print(
                f"✅ HMDB mapping complete: {len(filtered_data)} metabolites with HMDB IDs"
            )

            # Save to cache
            try:
                cache_data = {"filtered_data": filtered_data}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"✅ Saved HMDB mapping to cache: {cache_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save mapping to cache: {e}")

        except Exception as e:
            raise RuntimeError(f"Error during HMDB mapping: {e}")

    print(f"After HMDB mapping: {len(filtered_data)} metabolites")

    # Get metabolite info
    sample_data = filtered_data[["metabolite_identification", "hmdb_id"]]
    metabolite_names = sample_data["metabolite_identification"].tolist()

    # Extract abundance data and pivot (exactly like qmd)
    abundance_data = (
        filtered_data.filter(regex="ADG")
        .join(filtered_data[["metabolite_identification"]])
        .groupby("metabolite_identification")
        .mean()
        # NOTE: Do NOT use .loc[metabolite_names] here if metabolite_names has duplicates
        # The groupby().mean() already ensures we have one row per unique metabolite
    )
    abundance_data = abundance_data.T  # Transpose so metabolites are columns

    # Get sample names from abundance data to align with metadata
    sample_names = abundance_data.index.tolist()

    # Update metabolite_names to match abundance_data columns (unique names only)
    metabolite_names = abundance_data.columns.tolist()

    # Filter metadata to only include samples that are in abundance_data
    metadata_filtered = metadata[metadata["Sample Name"].isin(sample_names)]

    # Reorder metadata to match abundance_data sample order
    metadata_filtered = (
        metadata_filtered.set_index("Sample Name").reindex(sample_names).reset_index()
    )

    # Get group labels from filtered and aligned metadata
    group_labels = (
        metadata_filtered["Factor Value[Metabolic syndrome]"]
        .apply(lambda x: 1 if "diabetes mellitus" in str(x) else 0)
        .values
    )

    print(
        f"Dataset: {abundance_data.shape[0]} samples, {abundance_data.shape[1]} metabolites"
    )
    print(
        f"Sample alignment: abundance_data samples = {len(sample_names)}, metadata samples = {len(metadata_filtered)}"
    )
    print(
        f"Class distribution: Control={np.sum(group_labels == 0)}, Case={np.sum(group_labels == 1)}"
    )

    # Verify alignment
    if len(group_labels) != len(abundance_data):
        raise RuntimeError(
            f"Shape mismatch: abundance_data has {len(abundance_data)} samples but group_labels has {len(group_labels)}"
        )

    # Also update sample_data to match unique metabolites
    sample_data = sample_data.drop_duplicates(
        subset=["metabolite_identification"]
    ).reset_index(drop=True)
    sample_data = sample_data[
        sample_data["metabolite_identification"].isin(metabolite_names)
    ]

    return abundance_data, group_labels, metabolite_names, sample_data


def calculate_ground_truth_lnfc(abundance_data, group_labels):
    """Calculate empirical natural log fold change from full dataset - this is our ground truth."""
    control_indices = group_labels == 0
    case_indices = group_labels == 1

    control_means = abundance_data[control_indices].mean()
    case_means = abundance_data[case_indices].mean()

    # Calculate natural log fold change (case vs control) to match Bayesian models
    # Note: Using natural log to match np.log() used in Bayesian models
    # Add small constant to avoid log(0) - same as Bayesian models
    # Add small constant to avoid log(0) - use 1% of minimum non-zero value
    min_nonzero_control = (
        control_means[control_means > 0].min() if np.any(control_means > 0) else 1e-6
    )
    min_nonzero_case = (
        case_means[case_means > 0].min() if np.any(case_means > 0) else 1e-6
    )
    epsilon = 0.01 * min(min_nonzero_control, min_nonzero_case)
    lnfc = np.log((case_means + epsilon) / (control_means + epsilon))
    return lnfc


def fit_uninformative_bayesian_baseline(abundance_subsample, group_labels_subsample):
    """Fit Bayesian model with uninformative priors - the fair baseline comparison."""

    metabolite_names = abundance_subsample.columns.tolist()
    n_metabolites = len(metabolite_names)

    # Add small constant to avoid issues with zero abundances
    min_nonzero = abundance_subsample.values[abundance_subsample.values > 0].min()
    epsilon = 0.01 * min_nonzero if min_nonzero > 0 else 1e-6
    abundance_clean = abundance_subsample.values + epsilon

    with pm.Model():
        # More conservative priors to avoid numerical instability
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=2.0,  # Narrower to prevent exp() explosion
            shape=n_metabolites,
        )

        # Priors for baseline log-abundances (control group)
        alpha = pm.Normal(
            "alpha",
            mu=np.log(abundance_clean.mean(axis=0)),  # Log of abundance means
            sigma=1.0,  # Tighter to improve convergence
            shape=n_metabolites,
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=0.5, shape=n_metabolites
        )

        # Log-link GLM with clipping to prevent overflow
        log_mu = alpha + beta * group_labels_subsample[:, None]
        log_mu_clipped = pt.clip(log_mu, -10, 10)  # Prevent extreme values
        mu = pt.exp(log_mu_clipped)

        _ = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_clean,
        )

        # Sample from posterior (more samples for better convergence)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
                target_accept=0.9,
                cores=4,
                return_inferencedata=True,
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )

    # Get posterior means for beta (MAP estimates)
    beta_posterior_means = idata["posterior"]["beta"].mean(dim=["chain", "draw"]).values

    return beta_posterior_means, metabolite_names


def fit_llm_informed_hierarchical_model(
    abundance_subsample, group_labels_subsample, llm_priors
):
    """
    Fit a robust hierarchical Bayesian model using LLM predictions to create informed groups.

    This model uses a Student's T-distribution for robustness and implements a
    fallback mechanism for groups with too few members to model hierarchically.
    """
    min_group_size = 5  # Minimum members for a group to be modeled hierarchically

    # --- 1. Filter and Sort Metabolites ---
    available_metabolites = sorted(
        list(set(abundance_subsample.columns) & set(llm_priors.keys()))
    )
    abundance_subsample_filtered = abundance_subsample[available_metabolites]

    # --- 2. Group Metabolites and Identify Small/Large Groups ---
    llm_groups = {"increase": [], "decrease": [], "unchanged": []}
    for m in available_metabolites:
        direction = llm_priors[m].get("prediction", "unchanged")
        if "increase" in direction:
            llm_groups["increase"].append(m)
        elif "decrease" in direction:
            llm_groups["decrease"].append(m)
        else:
            llm_groups["unchanged"].append(m)

    large_groups = {k: v for k, v in llm_groups.items() if len(v) >= min_group_size}
    small_group_mets = [
        m for k, v in llm_groups.items() if len(v) < min_group_size for m in v
    ]
    hierarchical_mets = [m for v in large_groups.values() for m in v]

    # Reorder for PyMC model: hierarchical first, then fallback
    ordered_metabolites = hierarchical_mets + small_group_mets
    abundance_clean = abundance_subsample_filtered[ordered_metabolites].values + 1e-6
    n_hierarchical = len(hierarchical_mets)
    n_fallback = len(small_group_mets)

    # --- 3. Set up Indices and Priors for Hierarchical Portion ---
    group_names = sorted(large_groups.keys())
    group_assignments = [
        next(g_name for g_name, mets in large_groups.items() if met in mets)
        for met in hierarchical_mets
    ]
    group_idx = np.array([group_names.index(g) for g in group_assignments])

    # Weak directional guidance for groups - let data determine magnitudes
    group_mean_priors = {
        "increase": 0.1,  # Weak positive bias
        "decrease": -0.1,  # Weak negative bias
        "unchanged": 0.0,  # Neutral
    }
    group_mean_mus = np.array([group_mean_priors[g] for g in group_names])

    with pm.Model() as model:
        # === Part 1: Minimal Hierarchical Structure for LARGE Groups ===
        # Use LLM groups for intelligent initialization, not strong hierarchical shrinkage

        # Very weak group-level structure - mainly for initialization
        group_means = pm.Normal(
            "group_means",
            mu=group_mean_mus,
            sigma=3.0,
            shape=len(group_names),  # Very wide
        )

        # Individual metabolite effects with group-informed initialization but weak pooling
        beta_hierarchical = pm.Normal(
            "beta_hierarchical",
            mu=group_means[group_idx],
            sigma=2.0,  # Wide individual variation - minimal shrinkage
            shape=n_hierarchical,
        )

        # === Part 2: Fallback Priors for SMALL Groups ===
        beta_fallback = pm.Normal("beta_fallback", mu=0, sigma=1.5, shape=n_fallback)

        # === Combine Betas ===
        beta = pt.concatenate([beta_hierarchical, beta_fallback], axis=0)

        # === Common Model Components (Likelihood, etc.) ===
        alpha = pm.Normal(
            "alpha",
            mu=np.log(abundance_clean.mean(axis=0)),
            sigma=1.0,
            shape=len(ordered_metabolites),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=0.5, shape=len(ordered_metabolites)
        )
        log_mu = alpha + beta * group_labels_subsample[:, None]
        mu = pt.exp(pt.clip(log_mu, -10, 10))

        _ = pm.Normal("y_obs", mu=mu, sigma=metabolite_sigmas, observed=abundance_clean)

        # === Sample from Posterior ===
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
                target_accept=0.9,
                cores=4,
                return_inferencedata=True,
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )

    # --- 4. Extract and Reconstruct Results ---
    posterior = idata.posterior
    beta_hier_means = posterior["beta_hierarchical"].mean(dim=["chain", "draw"]).values

    # Handle case where there are no fallback metabolites
    if n_fallback > 0:
        beta_fall_means = posterior["beta_fallback"].mean(dim=["chain", "draw"]).values
        # Combine the results
        beta_posterior_means = np.concatenate([beta_hier_means, beta_fall_means])
    else:
        beta_posterior_means = beta_hier_means

    # The results are already in the correct order due to how we structured the inputs
    return beta_posterior_means, ordered_metabolites


def fit_hierarchical_empirical_bayes(abundance_subsample, group_labels_subsample):
    """Fit hierarchical empirical Bayes model with data-driven shrinkage."""

    metabolite_names = abundance_subsample.columns.tolist()
    n_metabolites = len(metabolite_names)

    # Add small constant to avoid issues with zero abundances
    min_nonzero = abundance_subsample.values[abundance_subsample.values > 0].min()
    epsilon = 0.01 * min_nonzero if min_nonzero > 0 else 1e-6
    abundance_clean = abundance_subsample.values + epsilon

    with pm.Model():
        # Hierarchical priors - let data determine shrinkage
        # Global mean and variance for log fold changes
        mu_global = pm.Normal("mu_global", mu=0, sigma=2)
        tau_global = pm.HalfNormal(
            "tau_global", sigma=1
        )  # Between-metabolite variation

        # Individual metabolite effects (hierarchical)
        beta = pm.Normal("beta", mu=mu_global, sigma=tau_global, shape=n_metabolites)

        # Baseline log-abundances (control group)
        alpha = pm.Normal(
            "alpha",
            mu=np.log(abundance_clean.mean(axis=0)),
            sigma=1.0,
            shape=n_metabolites,
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=0.5, shape=n_metabolites
        )

        # Log-link GLM with clipping to prevent overflow
        log_mu = alpha + beta * group_labels_subsample[:, None]
        log_mu_clipped = pt.clip(log_mu, -10, 10)
        mu = pt.exp(log_mu_clipped)

        _ = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_clean,
        )

        # Sample from posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
                target_accept=0.9,
                cores=4,
                return_inferencedata=True,
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )

    # Get posterior means for beta (empirical Bayes estimates)
    beta_posterior_means = idata["posterior"]["beta"].mean(dim=["chain", "draw"]).values

    # Also get shrinkage information

    mu_global_mean = idata["posterior"]["mu_global"].mean().values
    tau_global_mean = idata["posterior"]["tau_global"].mean().values

    return (
        beta_posterior_means,
        metabolite_names,
        {
            "global_mean": mu_global_mean,
            "global_tau": tau_global_mean,
            "shrinkage_applied": True,
        },
    )


def create_oracle_priors(ground_truth_lnfc, metabolite_names, noise_level=0.1):
    """Create oracle priors based on ground truth with optional noise."""

    oracle_priors = {}

    for metabolite in metabolite_names:
        if metabolite in ground_truth_lnfc.index:
            true_lnfc = ground_truth_lnfc[metabolite]

            # Add small amount of noise to simulate imperfect oracle
            if noise_level > 0:
                noisy_lnfc = true_lnfc + np.random.normal(0, noise_level)
            else:
                noisy_lnfc = true_lnfc

            # Set oracle prior with realistic biological uncertainty around truth
            oracle_priors[metabolite] = {
                "expected_lnfc": noisy_lnfc,
                "prior_sd": 0.25,  # Realistic uncertainty for "perfect" biological knowledge
            }

    return oracle_priors


def fit_oracle_bayesian_baseline(
    abundance_subsample, group_labels_subsample, oracle_priors
):
    """Fit Bayesian model with oracle priors - the theoretical upper bound."""

    # metabolite_names = abundance_subsample.columns.tolist()

    # Filter to only metabolites we have oracle priors for
    available_metabolites = list(oracle_priors.keys())
    intersection = set(abundance_subsample.columns) & set(available_metabolites)

    abundance_subsample_filtered = abundance_subsample[list(intersection)]
    available_metabolites = list(intersection)

    # Add small constant to avoid issues with zero abundances
    min_nonzero = abundance_subsample_filtered.values[
        abundance_subsample_filtered.values > 0
    ].min()
    epsilon = 0.01 * min_nonzero if min_nonzero > 0 else 1e-6
    abundance_clean = abundance_subsample_filtered.values + epsilon

    # Extract oracle prior parameters
    oracle_priors_mean = np.array(
        [oracle_priors[m]["expected_lnfc"] for m in available_metabolites]
    )
    oracle_priors_sd = np.array(
        [oracle_priors[m]["prior_sd"] for m in available_metabolites]
    )

    with pm.Model():
        # Use oracle priors for beta (log fold change effects)
        beta = pm.Normal(
            "beta",
            mu=oracle_priors_mean,
            sigma=oracle_priors_sd,
            shape=len(available_metabolites),
        )

        # Priors for baseline log-abundances (control group)
        alpha = pm.Normal(
            "alpha",
            mu=np.log(abundance_clean.mean(axis=0)),
            sigma=1.0,
            shape=len(available_metabolites),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=0.5, shape=len(available_metabolites)
        )

        # Log-link GLM with clipping to prevent overflow
        log_mu = alpha + beta * group_labels_subsample[:, None]
        log_mu_clipped = pt.clip(log_mu, -10, 10)
        mu = pt.exp(log_mu_clipped)

        _ = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_clean,
        )

        # Sample from posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
                target_accept=0.9,
                cores=4,
                return_inferencedata=True,
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )

    # Get posterior means for beta (MAP estimates)
    beta_posterior_means = idata["posterior"]["beta"].mean(dim=["chain", "draw"]).values

    return beta_posterior_means, available_metabolites


def generate_hmdb_contexts(sample_data):
    """Generate streamlined HMDB contexts with description + available structured data."""
    hmdb_client = HMDBClient()
    hmdb_contexts = {}

    print(
        "Fetching HMDB contexts (streamlined: description + available structured data)..."
    )
    successful_fetches = 0

    for idx, row in sample_data.iterrows():
        metabolite_name = row["metabolite_identification"]
        hmdb_id = row["hmdb_id"]
        print(f"Fetching context for {metabolite_name}")

        try:
            # Get metabolite data from chembridge
            metabolite_data = hmdb_client.get_metabolite_info(hmdb_id)
            if metabolite_data:
                successful_fetches += 1
                # Streamlined context: description + available structured data
                context_parts = [f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"]

                # Add description (rich narrative context)
                if metabolite_data.get("description"):
                    context_parts.append(
                        f"Description: {metabolite_data['description']}"
                    )

                # Add any available structured data (pathways, biological functions, etc.)
                # Check what structured fields are available in this dataset
                structured_fields = {
                    "pathways": "Pathways",
                    "biological_functions": "Functions",
                    "diseases": "Disease associations",
                    "tissue_locations": "Tissue locations",
                }

                for field_key, field_label in structured_fields.items():
                    if metabolite_data.get(field_key):
                        values = metabolite_data[field_key]
                        if isinstance(values, list) and values:
                            # Limit to top 3-5 items to keep context manageable
                            values_str = ", ".join(values[:3])
                            context_parts.append(f"{field_label}: {values_str}")

                hmdb_contexts[metabolite_name] = " | ".join(context_parts)
            else:
                # Fallback: just metabolite name and ID
                hmdb_contexts[metabolite_name] = (
                    f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"
                )

        except Exception as e:
            print(f"Error fetching data for {hmdb_id}: {e}")
            hmdb_contexts[metabolite_name] = (
                f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"
            )

    print(
        f"Successfully fetched HMDB data for {successful_fetches}/{len(sample_data)} metabolites"
    )
    return hmdb_contexts


def load_or_generate_qualitative_predictions(
    metabolite_names,
    sample_data,
    use_hmdb_context=True,
    model_name="gpt-4o-mini-2024-07-18",
    temperature=1.0,
    cache_dir="/Users/chiraag/Projects/gwu/lab/apriomics/output",
):
    """Load cached qualitative predictions if available, otherwise generate and cache them."""

    # Create cache filename based on model/context/temperature parameters
    context_suffix = "_with_context" if use_hmdb_context else "_no_context"
    model_suffix = f"_{model_name.replace('-', '_').replace('.', '_')}"
    temp_suffix = f"_temp{temperature:.1f}".replace(".", "")
    cache_filename = f"qualitative_predictions{context_suffix}{model_suffix}{temp_suffix}_{len(metabolite_names)}metabolites.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    os.makedirs(cache_dir, exist_ok=True)

    # Try to load from cache first
    if os.path.exists(cache_path):
        print(f"Loading cached qualitative predictions from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_predictions = pickle.load(f)

            # Verify the cached predictions contain all our metabolites
            if set(metabolite_names).issubset(set(cached_predictions.keys())):
                print(
                    "✅ Cache contains all needed metabolites - using cached predictions"
                )
                # Return only the metabolites we need
                filtered_predictions = {
                    name: cached_predictions[name]
                    for name in metabolite_names
                    if name in cached_predictions
                }
                return filtered_predictions
            else:
                missing = set(metabolite_names) - set(cached_predictions.keys())
                print(f"⚠️  Cache missing metabolites: {missing} - regenerating")
        except Exception as e:
            print(f"⚠️  Error loading cache: {e} - regenerating")

    # Generate fresh qualitative predictions
    print(
        f"Generating fresh qualitative predictions ({'with' if use_hmdb_context else 'without'} HMDB context, {model_name})..."
    )

    condition = """
Study: Type 2 diabetes mellitus vs healthy control
Context: Type 2 diabetes mellitus is the result of a combination of impaired insulin secretion with reduced insulin sensitivity of target tissues. In this study, NMR-based metabolomic analysis in conjunction with uni- and multivariate statistics was applied to examine the urinary metabolic changes in Human type 2 diabetes mellitus patients compared to the control group. The human population were un medicated diabetic patients who have good daily dietary control over their blood glucose concentrations by following the guidelines on diet issued by the American Diabetes Association.
Sample type: Urine samples analyzed by NMR spectroscopy
Patient population: Unmedicated Type 2 diabetes patients with good dietary control vs healthy controls
Expected changes: Look for metabolites altered in diabetes pathophysiology, particularly those related to glucose metabolism, insulin sensitivity, and urinary excretion patterns.
"""

    if use_hmdb_context:
        # Generate HMDB contexts
        hmdb_contexts = generate_hmdb_contexts(sample_data)
        priors_data = PriorData(hmdb_contexts=hmdb_contexts)
    else:
        # Just use metabolite names
        priors_data = PriorData(metabolite_names=metabolite_names)

    # Generate qualitative predictions
    qualitative_predictions = get_llm_qualitative_predictions(
        priors=priors_data,
        condition=condition,
        use_hmdb_context=use_hmdb_context,
        model_name=model_name,
        temperature=temperature,
    )

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(qualitative_predictions, f)
        print(f"✅ Saved qualitative predictions to cache: {cache_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save to cache: {e}")

    return qualitative_predictions


def apply_prior_strength_mapping(qualitative_predictions, prior_strength):
    """Apply numerical mapping to qualitative predictions based on strength."""
    from apriomics.priors.base import map_qualitative_to_numerical_priors

    return map_qualitative_to_numerical_priors(qualitative_predictions, prior_strength)


def fit_bayesian_model_subsample(
    abundance_subsample, group_labels_subsample, differential_priors
):
    """Fit Bayesian model on subsample with LLM priors (same as qmd)."""

    # Filter abundance_subsample to only include metabolites we have priors for
    available_metabolites = list(differential_priors.keys())

    # Check intersection
    intersection = set(abundance_subsample.columns) & set(available_metabolites)

    abundance_subsample_filtered = abundance_subsample[list(intersection)]
    available_metabolites = list(
        intersection
    )  # Update to only use actually available ones

    # Add small constant to avoid issues with zero abundances
    min_nonzero = abundance_subsample_filtered.values[
        abundance_subsample_filtered.values > 0
    ].min()
    epsilon = 0.01 * min_nonzero if min_nonzero > 0 else 1e-6
    abundance_clean = abundance_subsample_filtered.values + epsilon

    # Extract LLM prior parameters (only for available metabolites)
    llm_priors_mean = np.array(
        [differential_priors[m]["expected_lnfc"] for m in available_metabolites]
    )
    llm_priors_sd = np.array(
        [differential_priors[m]["prior_sd"] for m in available_metabolites]
    )

    with pm.Model():
        # Use LLM priors for beta (log fold change effects)
        beta = pm.Normal(
            "beta",
            mu=llm_priors_mean,
            sigma=llm_priors_sd,
            shape=len(available_metabolites),
        )

        # Priors for baseline log-abundances (control group)
        alpha = pm.Normal(
            "alpha",
            mu=np.log(abundance_clean.mean(axis=0)),  # Log of abundance means
            sigma=1.0,  # Tighter to improve convergence
            shape=len(available_metabolites),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=0.5, shape=len(available_metabolites)
        )

        # Log-link GLM with clipping to prevent overflow
        log_mu = alpha + beta * group_labels_subsample[:, None]
        log_mu_clipped = pt.clip(log_mu, -10, 10)  # Prevent extreme values
        mu = pt.exp(log_mu_clipped)

        # Likelihood in linear space
        _ = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_clean,
        )

        # Sample from posterior (reduced for speed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
                target_accept=0.9,
                cores=4,
                return_inferencedata=True,
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )

    # Get posterior means for beta
    beta_group = idata["posterior"]
    beta_posterior_means = beta_group["beta"].mean(dim=["chain", "draw"]).values

    return beta_posterior_means, available_metabolites


def subsample_balanced(abundance_data, group_labels, n_per_group, random_state=None):
    """Create balanced subsample."""
    control_indices = np.where(group_labels == 0)[0]
    case_indices = np.where(group_labels == 1)[0]

    np.random.seed(random_state)

    # Sample n_per_group from each class
    control_sample = np.random.choice(control_indices, size=n_per_group, replace=False)
    case_sample = np.random.choice(case_indices, size=n_per_group, replace=False)

    # Combine samples
    subsample_indices = np.concatenate([control_sample, case_sample])
    subsample_labels = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])

    return abundance_data.iloc[subsample_indices], subsample_labels


def run_benchmark(
    abundance_data,
    group_labels,
    metabolite_names,
    sample_data,
    ground_truth_lnfc,
    sample_sizes=[5, 10, 15, 20],
    n_replicates=20,
):
    """Run the main benchmark experiment comparing multiple methods."""

    # Load or generate qualitative predictions once per model/context combo (EFFICIENT!)
    print("--- Loading/generating qualitative predictions ---")

    # Generate qualitative predictions only once per model/context combo
    qualitative_predictions = {
        # Gemini Flash 2.0
        "flash_no_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=False,
            model_name="gemini-2.0-flash",
            temperature=0.0,
        ),
        "flash_with_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=True,
            model_name="gemini-2.0-flash",
            temperature=0.0,
        ),
        "pro_no_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=False,
            model_name="gemini-2.5-pro",
            temperature=0.0,
        ),
        # OpenAI models
        "4o_no_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=False,
            model_name="gpt-4o-2024-08-06",
            temperature=0.0,
        ),
        "4o_with_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=True,
            model_name="gpt-4o-2024-08-06",
            temperature=0.0,
        ),
        "o3-mini_no_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=False,
            model_name="o3-mini-2025-01-31",
            temperature=0.0,
        ),
        "o3-mini_with_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=True,
            model_name="o3-mini-2025-01-31",
            temperature=0.0,
        ),
        "o3_no_context": load_or_generate_qualitative_predictions(
            metabolite_names,
            sample_data,
            use_hmdb_context=False,
            model_name="o3-2025-04-16",
            temperature=0.0,
        ),
    }

    # Create oracle priors based on ground truth (perfect knowledge upper bound)
    print("Creating oracle priors from ground truth...")
    oracle_priors = create_oracle_priors(
        ground_truth_lnfc, metabolite_names, noise_level=0.00
    )

    # Apply conservative strength mappings to all qualitative predictions
    print("--- Applying conservative strength mappings to qualitative predictions ---")
    priors = {
        # Gemini Flash 2.0
        "flash_no_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["flash_no_context"], "conservative"
        ),
        "flash_with_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["flash_with_context"], "conservative"
        ),
        "pro_no_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["pro_no_context"], "conservative"
        ),
        # OpenAI models
        "4o_no_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["4o_no_context"], "conservative"
        ),
        "4o_with_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["4o_with_context"], "conservative"
        ),
        "o3-mini_no_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["o3-mini_no_context"], "conservative"
        ),
        "o3-mini_with_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["o3-mini_with_context"], "conservative"
        ),
        "o3_no_context_conservative": apply_prior_strength_mapping(
            qualitative_predictions["o3_no_context"], "conservative"
        ),
    }
    print("--- All priors loaded ---")

    common_metabolites = list(
        set.intersection(*[set(p.keys()) for p in priors.values()])
    )
    ground_truth_filtered = ground_truth_lnfc[common_metabolites]
    print(f"Using {len(common_metabolites)} metabolites common to all prior methods")

    results = []
    detailed_results = []  # To store per-metabolite results

    total_iterations = len(sample_sizes) * n_replicates
    pbar = tqdm(
        total=total_iterations,
        desc="Running benchmark",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for n_per_group in sample_sizes:
        for replicate in range(n_replicates):
            pbar.set_description(
                f"Sample size {n_per_group}, replicate {replicate + 1}/{n_replicates}"
            )

            abundance_subsample, group_labels_subsample = subsample_balanced(
                abundance_data, group_labels, n_per_group, random_state=replicate
            )

            try:
                # Uninformative Bayesian baseline (fair comparison)
                beta_uninformative, mets_uninformative = (
                    fit_uninformative_bayesian_baseline(
                        abundance_subsample, group_labels_subsample
                    )
                )
                df_uninformative = (
                    pd.DataFrame(
                        {"metabolite": mets_uninformative, "beta": beta_uninformative}
                    )
                    .set_index("metabolite")
                    .reindex(common_metabolites)
                    .dropna()
                )
                corr_uninformative, _ = pearsonr(
                    ground_truth_filtered.reindex(df_uninformative.index),
                    df_uninformative["beta"],
                )
                rmse_uninformative = np.sqrt(
                    mean_squared_error(
                        ground_truth_filtered.reindex(df_uninformative.index),
                        df_uninformative["beta"],
                    )
                )
                results.append(
                    {
                        "sample_size": n_per_group,
                        "replicate": replicate,
                        "method": "uninformative_bayesian",
                        "correlation": corr_uninformative,
                        "rmse": rmse_uninformative,
                    }
                )

                # Store detailed results for bias-variance calculation
                for metabolite in df_uninformative.index:
                    detailed_results.append(
                        {
                            "sample_size": n_per_group,
                            "replicate": replicate,
                            "method": "uninformative_bayesian",
                            "metabolite": metabolite,
                            "estimate": df_uninformative.loc[metabolite, "beta"],
                            "ground_truth": ground_truth_filtered.loc[metabolite],
                        }
                    )

                # Hierarchical Bayesian with LLM-informed groups
                beta_hierarchical, mets_hierarchical = (
                    fit_llm_informed_hierarchical_model(
                        abundance_subsample,
                        group_labels_subsample,
                        priors["flash_no_context_conservative"],
                    )
                )
                df_hierarchical = (
                    pd.DataFrame(
                        {"metabolite": mets_hierarchical, "beta": beta_hierarchical}
                    )
                    .set_index("metabolite")
                    .reindex(common_metabolites)
                    .dropna()
                )
                corr_hierarchical, _ = pearsonr(
                    ground_truth_filtered.reindex(df_hierarchical.index),
                    df_hierarchical["beta"],
                )
                rmse_hierarchical = np.sqrt(
                    mean_squared_error(
                        ground_truth_filtered.reindex(df_hierarchical.index),
                        df_hierarchical["beta"],
                    )
                )
                results.append(
                    {
                        "sample_size": n_per_group,
                        "replicate": replicate,
                        "method": "llm_informed_hierarchical",
                        "correlation": corr_hierarchical,
                        "rmse": rmse_hierarchical,
                    }
                )

                # Store detailed results for bias-variance calculation
                for metabolite in df_hierarchical.index:
                    detailed_results.append(
                        {
                            "sample_size": n_per_group,
                            "replicate": replicate,
                            "method": "llm_informed_hierarchical",
                            "metabolite": metabolite,
                            "estimate": df_hierarchical.loc[metabolite, "beta"],
                            "ground_truth": ground_truth_filtered.loc[metabolite],
                        }
                    )

                # Oracle Bayesian (perfect knowledge upper bound)
                beta_oracle, mets_oracle = fit_oracle_bayesian_baseline(
                    abundance_subsample, group_labels_subsample, oracle_priors
                )
                df_oracle = (
                    pd.DataFrame({"metabolite": mets_oracle, "beta": beta_oracle})
                    .set_index("metabolite")
                    .reindex(common_metabolites)
                    .dropna()
                )
                corr_oracle, _ = pearsonr(
                    ground_truth_filtered.reindex(df_oracle.index), df_oracle["beta"]
                )
                rmse_oracle = np.sqrt(
                    mean_squared_error(
                        ground_truth_filtered.reindex(df_oracle.index),
                        df_oracle["beta"],
                    )
                )
                results.append(
                    {
                        "sample_size": n_per_group,
                        "replicate": replicate,
                        "method": "oracle_bayesian",
                        "correlation": corr_oracle,
                        "rmse": rmse_oracle,
                    }
                )

                # Store detailed results for bias-variance calculation
                for metabolite in df_oracle.index:
                    detailed_results.append(
                        {
                            "sample_size": n_per_group,
                            "replicate": replicate,
                            "method": "oracle_bayesian",
                            "metabolite": metabolite,
                            "estimate": df_oracle.loc[metabolite, "beta"],
                            "ground_truth": ground_truth_filtered.loc[metabolite],
                        }
                    )

                # Test streamlined LLM methods with conservative mapping
                llm_methods = [
                    (
                        "flash_no_context_conservative",
                        "Flash 2.0 (No Context, Conservative)",
                    ),
                    (
                        "flash_with_context_conservative",
                        "Flash 2.0 (With Context, Conservative)",
                    ),
                    (
                        "pro_no_context_conservative",
                        "Pro 2.5 (No Context, Conservative)",
                    ),
                    ("4o_no_context_conservative", "GPT-4o (No Context, Conservative)"),
                    (
                        "4o_with_context_conservative",
                        "GPT-4o (With Context, Conservative)",
                    ),
                    (
                        "o3-mini_no_context_conservative",
                        "O3 Mini (No Context, Conservative)",
                    ),
                    (
                        "o3-mini_with_context_conservative",
                        "O3 Mini (With Context, Conservative)",
                    ),
                    ("o3_no_context_conservative", "O3 (No Context, Conservative)"),
                ]

                for method_key, method_name in llm_methods:
                    prior_set = priors[method_key]
                    beta_vals, mets = fit_bayesian_model_subsample(
                        abundance_subsample, group_labels_subsample, prior_set
                    )
                    df_method = (
                        pd.DataFrame({"metabolite": mets, "beta": beta_vals})
                        .set_index("metabolite")
                        .reindex(common_metabolites)
                        .dropna()
                    )
                    corr_method, _ = pearsonr(
                        ground_truth_filtered.reindex(df_method.index),
                        df_method["beta"],
                    )
                    rmse_method = np.sqrt(
                        mean_squared_error(
                            ground_truth_filtered.reindex(df_method.index),
                            df_method["beta"],
                        )
                    )
                    results.append(
                        {
                            "sample_size": n_per_group,
                            "replicate": replicate,
                            "method": method_key,
                            "correlation": corr_method,
                            "rmse": rmse_method,
                        }
                    )

                    # Store detailed results for bias-variance calculation
                    for metabolite in df_method.index:
                        detailed_results.append(
                            {
                                "sample_size": n_per_group,
                                "replicate": replicate,
                                "method": method_key,
                                "metabolite": metabolite,
                                "estimate": df_method.loc[metabolite, "beta"],
                                "ground_truth": ground_truth_filtered.loc[metabolite],
                            }
                        )

            except Exception as e:
                pbar.write(
                    f"  FATAL ERROR in replicate {replicate}: {type(e).__name__}: {e}"
                )
                import traceback

                pbar.write(f"  Traceback: {traceback.format_exc()}")
                raise e
            finally:
                pbar.update(1)

    pbar.close()

    # Calculate bias and variance per method
    detailed_df = pd.DataFrame(detailed_results)
    bias_variance_results = calculate_bias_variance_per_method(detailed_df)

    return pd.DataFrame(results), bias_variance_results


def calculate_bias_variance_per_method(detailed_df):
    """
    Calculate bias and variance per method from detailed per-metabolite results.

    Args:
        detailed_df: DataFrame with columns [sample_size, replicate, method, metabolite, estimate, ground_truth]

    Returns:
        DataFrame with columns [sample_size, method, overall_bias, overall_variance, mse]
    """
    bias_variance_results = []

    # Group by sample_size and method
    for (sample_size, method), group in detailed_df.groupby(["sample_size", "method"]):
        # For each metabolite, calculate bias and variance across replicates
        metabolite_bias_variance = []

        for metabolite, met_group in group.groupby("metabolite"):
            estimates = met_group["estimate"].values
            ground_truth = met_group["ground_truth"].iloc[0]  # Same for all replicates

            # Bias = E[estimate] - truth
            mean_estimate = np.mean(estimates)
            bias = mean_estimate - ground_truth

            # Variance = E[(estimate - E[estimate])^2]
            variance = np.var(estimates, ddof=1) if len(estimates) > 1 else 0.0

            # MSE = bias^2 + variance
            mse = bias**2 + variance

            metabolite_bias_variance.append(
                {
                    "metabolite": metabolite,
                    "bias": bias,
                    "variance": variance,
                    "mse": mse,
                    "abs_bias": abs(bias),
                }
            )

        # Overall metrics across metabolites
        met_df = pd.DataFrame(metabolite_bias_variance)
        overall_bias = met_df["abs_bias"].mean()  # Mean absolute bias
        overall_variance = met_df["variance"].mean()  # Mean variance
        overall_mse = met_df["mse"].mean()  # Mean MSE

        bias_variance_results.append(
            {
                "sample_size": sample_size,
                "method": method,
                "overall_bias": overall_bias,
                "overall_variance": overall_variance,
                "overall_mse": overall_mse,
                "n_metabolites": len(met_df),
            }
        )

    return pd.DataFrame(bias_variance_results)


def test_full_dataset_performance(
    abundance_data, group_labels, metabolite_names, sample_data, ground_truth_lnfc
):
    """Test performance on full dataset - uninformative Bayesian should perform best here."""
    print("\n" + "=" * 60)
    print("FULL DATASET TEST (Expected: Uninformative Bayesian ≈ Empirical)")
    print("=" * 60)

    # Test uninformative Bayesian on full dataset
    print("Testing uninformative Bayesian on full dataset...")
    try:
        beta_uninformative, mets_uninformative = fit_uninformative_bayesian_baseline(
            abundance_data, group_labels
        )

        # Calculate correlation with ground truth
        common_metabolites = list(
            set(mets_uninformative) & set(ground_truth_lnfc.index)
        )

        uninformative_filtered = pd.Series(
            beta_uninformative, index=mets_uninformative
        )[common_metabolites]
        ground_truth_filtered = ground_truth_lnfc[common_metabolites]

        corr_uninformative, _ = pearsonr(ground_truth_filtered, uninformative_filtered)
        rmse_uninformative = np.sqrt(
            mean_squared_error(ground_truth_filtered, uninformative_filtered)
        )

        print(
            f"Uninformative Bayesian (full data): r={corr_uninformative:.3f}, RMSE={rmse_uninformative:.6f}"
        )

        # Compare to empirical (should be nearly identical)
        print(
            f"Difference from empirical: Δr={1.000 - corr_uninformative:.3f}, ΔRMSE={rmse_uninformative:.6f}"  # pyright: ignore[reportOperatorIssue]
        )

        if corr_uninformative > 0.99 and rmse_uninformative < 0.01:  # type: ignore
            print("✅ PASS: Uninformative Bayesian ≈ Empirical on full data")
        else:
            print(
                "❌ CONCERN: Large difference suggests prior still too informative or model issue"
            )

    except Exception as e:
        print(f"❌ ERROR in full dataset test: {e}")

    print("=" * 60)


def analyze_benchmark_results(results_df, ground_truth_lnfc, method_keys):
    """Analyze and visualize benchmark results."""

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    summary = (
        results_df.groupby(["sample_size", "method"])
        .agg({"correlation": ["mean", "std"], "rmse": ["mean", "std"]})
        .round(3)
    )
    print("\nCorrelation with Ground Truth:")
    print(summary["correlation"])
    print("\nRMSE from Ground Truth:")
    print(summary["rmse"])

    print(f"\n{'STATISTICAL COMPARISONS':^60}")
    print("-" * 60)
    from scipy.stats import ttest_ind

    for sample_size in results_df["sample_size"].unique():
        print(f"\nSample size {sample_size} per group:")
        subset = results_df[results_df["sample_size"] == sample_size]
        # Create groups for all methods
        baseline_group = subset[subset["method"] == "uninformative_bayesian"]

        # Compare each LLM method vs uninformative Bayesian baseline
        for method_key in method_keys:
            method_group = subset[subset["method"] == method_key]
            if len(method_group) > 0 and len(baseline_group) > 0:
                corr_stat, corr_pval = ttest_ind(
                    method_group["correlation"], baseline_group["correlation"]
                )
                rmse_stat, rmse_pval = ttest_ind(
                    baseline_group["rmse"], method_group["rmse"]
                )  # Lower RMSE is better
                method_display = method_key.replace("_", " ").title()
                print(f"  {method_display} vs Uninformative Bayesian:")
                print(
                    f"    Correlation improvement: t={corr_stat:.3f}, p={corr_pval:.4f}"
                )
                print(f"    RMSE improvement: t={rmse_stat:.3f}, p={rmse_pval:.4f}")

    return summary


def create_benchmark_visualizations(results_df, ground_truth_lnfc):
    """Create visualizations for the benchmark results."""

    plt.style.use("default")
    sns.set_palette("colorblind")
    # Create readable method names for visualization
    method_names = {
        "uninformative_bayesian": "Uninformative Bayesian",
        "llm_informed_hierarchical": "LLM-Informed Hierarchical",
        "oracle_bayesian": "Oracle Bayesian (Upper Bound)",
        "flash_no_context_conservative": "Flash 2.0 (No Context, Conservative)",
        "flash_with_context_conservative": "Flash 2.0 (With Context, Conservative)",
        "pro_no_context_conservative": "Pro 2.5 (No Context, Conservative)",
        "4o_no_context_conservative": "GPT-4o (No Context, Conservative)",
        "4o_with_context_conservative": "GPT-4o (With Context, Conservative)",
        "o3-mini_no_context_conservative": "O3 Mini (No Context, Conservative)",
        "o3-mini_with_context_conservative": "O3 Mini (With Context, Conservative)",
        "o3_no_context_conservative": "O3 (No Context, Conservative)",
    }
    results_df["Method"] = results_df["method"].replace(method_names)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    sns.boxplot(
        data=results_df, x="sample_size", y="correlation", hue="Method", ax=axes[0]
    )
    axes[0].set_title("Correlation with Ground Truth vs Sample Size")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Correlation with Ground Truth")
    axes[0].legend(title="Method")
    axes[0].grid(True, alpha=0.3)

    sns.boxplot(data=results_df, x="sample_size", y="rmse", hue="Method", ax=axes[1])
    axes[1].set_title("RMSE from Ground Truth vs Sample Size")
    axes[1].set_xlabel("Samples per Group")
    axes[1].set_ylabel("RMSE from Ground Truth")
    axes[1].get_legend().remove()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/benchmark_prior_recovery.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main benchmark execution."""

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set! This benchmark requires LLM access.")
        return

    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()

    print(f"Using {len(metabolite_names)} metabolites for benchmark")

    print("Calculating ground truth lnFC from full dataset...")
    ground_truth_lnfc = calculate_ground_truth_lnfc(abundance_data, group_labels)

    # Test uninformative Bayesian on full dataset (should ≈ empirical)
    test_full_dataset_performance(
        abundance_data, group_labels, metabolite_names, sample_data, ground_truth_lnfc
    )

    print("Running benchmark experiment...")
    results_df, bias_variance_df = run_benchmark(
        abundance_data,
        group_labels,
        metabolite_names,
        sample_data,
        ground_truth_lnfc,
        sample_sizes=[5, 10, 15, 20],
        n_replicates=10,
    )

    print("Analyzing results...")
    if len(results_df) == 0:
        print("ERROR: No results generated! All replicates failed.")
        print("This likely indicates issues with:")
        print("- PyMC sampling convergence")
        print("- LLM API timeouts/errors")
        print("- Data processing errors")
        return None, None

    print(f"Successfully generated {len(results_df)} results")

    # Get the method keys that were actually used (exclude baseline)
    llm_method_keys = [
        k for k in results_df["method"].unique() if k != "uninformative_bayesian"
    ]
    summary = analyze_benchmark_results(results_df, ground_truth_lnfc, llm_method_keys)

    print("Creating visualizations...")
    create_benchmark_visualizations(results_df, ground_truth_lnfc)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(f"{output_dir}/benchmark_prior_recovery_results.csv", index=False)
    summary.to_csv(f"{output_dir}/benchmark_prior_recovery_summary.csv")
    bias_variance_df.to_csv(
        f"{output_dir}/benchmark_bias_variance_results.csv", index=False
    )

    # Display bias-variance summary
    print("\n" + "=" * 80)
    print("BIAS-VARIANCE DECOMPOSITION BY METHOD")
    print("=" * 80)

    # Create readable method names
    method_names = {
        "uninformative_bayesian": "Uninformative Bayesian",
        "llm_informed_hierarchical": "LLM-Informed Hierarchical",
        "oracle_bayesian": "Oracle Bayesian (Upper Bound)",
        "flash_no_context_conservative": "Flash 2.0 (No Context, Conservative)",
        "flash_with_context_conservative": "Flash 2.0 (With Context, Conservative)",
        "pro_no_context_conservative": "Pro 2.5 (No Context, Conservative)",
        "4o_no_context_conservative": "GPT-4o (No Context, Conservative)",
        "4o_with_context_conservative": "GPT-4o (With Context, Conservative)",
        "o3-mini_no_context_conservative": "O3 Mini (No Context, Conservative)",
        "o3-mini_with_context_conservative": "O3 Mini (With Context, Conservative)",
        "o3_no_context_conservative": "O3 (No Context, Conservative)",
    }

    bias_variance_df["Method"] = (
        bias_variance_df["method"].map(method_names).fillna(bias_variance_df["method"])
    )

    # Show bias-variance results grouped by sample size
    for sample_size in sorted(bias_variance_df["sample_size"].unique()):
        print(f"\nSample Size: {sample_size}")
        print("-" * 40)
        subset = bias_variance_df[bias_variance_df["sample_size"] == sample_size].copy()
        subset = subset.sort_values("overall_mse")

        for _, row in subset.iterrows():
            print(
                f"{row['Method']:35s}: Bias={row['overall_bias']:.4f}, Var={row['overall_variance']:.4f}, MSE={row['overall_mse']:.4f}"
            )

    print(f"\nResults saved to {output_dir}/")
    print("- benchmark_prior_recovery_results.csv")
    print("- benchmark_prior_recovery_summary.csv")
    print("- benchmark_bias_variance_results.csv")
    print("Benchmark complete!")

    return results_df, summary


if __name__ == "__main__":
    result = main()
    if result is not None:
        results_df, summary = result
