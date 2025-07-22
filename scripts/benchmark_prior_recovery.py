"""
Benchmark: Recovery of ground truth log2FC using different LLM prior methods.

This script evaluates how well LLM priors help recover true log2FC estimates
from the full MTBLS1 dataset when given limited subsampled data.

Compares:
1. Bayesian + LLM priors (no HMDB context)
2. Bayesian + LLM priors (with HMDB context)

Ground truth: empirical log2FC from full dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pymc as pm
import arviz as az
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
import pickle

warnings.filterwarnings("ignore")

from apriomics.priors import PriorData, get_llm_differential_priors
from chembridge.databases.hmdb import HMDBClient
from chembridge import map_metabolites


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

    # DEBUG: Print detailed shape information
    print(f"\n=== DEBUG SHAPES ===")
    print(f"abundance_data.shape: {abundance_data.shape}")
    print(f"group_labels.shape: {group_labels.shape}")
    print(f"sample_names length: {len(sample_names)}")
    print(f"metadata_filtered length: {len(metadata_filtered)}")
    print(f"First 5 sample names from abundance_data: {sample_names[:5]}")
    print(
        f"First 5 sample names from metadata_filtered: {metadata_filtered['Sample Name'].tolist()[:5]}"
    )
    print(f"===================\n")

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


def calculate_ground_truth_log2fc(abundance_data, group_labels):
    """Calculate empirical log2FC from full dataset - this is our ground truth."""
    control_indices = group_labels == 0
    case_indices = group_labels == 1

    control_means = abundance_data[control_indices].mean()
    case_means = abundance_data[case_indices].mean()

    # Calculate log2 fold change (case vs control)
    log2fc = np.log2(case_means / control_means)
    return log2fc


def generate_hmdb_contexts(sample_data):
    """Generate HMDB contexts like in the qmd file."""
    hmdb_client = HMDBClient()
    hmdb_contexts = {}

    print("Fetching HMDB contexts...")
    for idx, row in sample_data.iterrows():
        metabolite_name = row["metabolite_identification"]
        hmdb_id = row["hmdb_id"]
        print(f"Fetching context for {metabolite_name}")

        try:
            # Get metabolite data from chembridge
            metabolite_data = hmdb_client.get_metabolite_info(hmdb_id)
            if metabolite_data:
                # Format rich context for LLM (same as qmd)
                context_parts = [f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"]

                # Add description
                if metabolite_data.get("description"):
                    context_parts.append(
                        f"Description: {metabolite_data['description']}"
                    )

                # Add pathways
                if metabolite_data.get("pathways"):
                    pathways_str = ", ".join(metabolite_data["pathways"][:5])
                    context_parts.append(f"Pathways: {pathways_str}")

                # Add diseases
                if metabolite_data.get("diseases"):
                    diseases_str = ", ".join(metabolite_data["diseases"][:5])
                    context_parts.append(f"Associated diseases: {diseases_str}")

                # Add biological functions
                if metabolite_data.get("biological_functions"):
                    functions_str = ", ".join(
                        metabolite_data["biological_functions"][:3]
                    )
                    context_parts.append(f"Biological functions: {functions_str}")

                # Add tissue locations
                if metabolite_data.get("tissue_locations"):
                    tissues_str = ", ".join(metabolite_data["tissue_locations"][:5])
                    context_parts.append(f"Found in tissues: {tissues_str}")

                # Add biofluid locations
                if metabolite_data.get("biofluid_locations"):
                    biofluids_str = ", ".join(metabolite_data["biofluid_locations"][:5])
                    context_parts.append(f"Found in biofluids: {biofluids_str}")

                hmdb_contexts[metabolite_name] = " | ".join(context_parts)
            else:
                hmdb_contexts[metabolite_name] = (
                    f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"
                )

        except Exception as e:
            print(f"Error fetching data for {hmdb_id}: {e}")
            hmdb_contexts[metabolite_name] = (
                f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"
            )

    return hmdb_contexts


def load_or_generate_llm_priors(
    metabolite_names,
    sample_data,
    use_hmdb_context=True,
    prior_strength="conservative",
    model_name="gemini-2.0-flash-exp",
    cache_dir="/Users/chiraag/Projects/gwu/lab/apriomics/output",
):
    """Load cached LLM priors if available, otherwise generate and cache them."""

    # Create cache filename based on parameters
    context_suffix = "_with_context" if use_hmdb_context else "_no_context"
    cache_filename = f"llm_priors_{prior_strength}{context_suffix}_{len(metabolite_names)}metabolites.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    os.makedirs(cache_dir, exist_ok=True)

    # Try to load from cache first
    if os.path.exists(cache_path):
        print(f"Loading cached LLM priors from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_priors = pickle.load(f)

            # Verify the cached priors contain all our metabolites (subset is ok)
            if set(metabolite_names).issubset(set(cached_priors.keys())):
                print("✅ Cache contains all needed metabolites - using cached priors")
                # Return only the metabolites we need
                filtered_priors = {
                    name: cached_priors[name]
                    for name in metabolite_names
                    if name in cached_priors
                }
                return filtered_priors
            else:
                missing = set(metabolite_names) - set(cached_priors.keys())
                print(f"⚠️  Cache missing metabolites: {missing} - regenerating")
        except Exception as e:
            print(f"⚠️  Error loading cache: {e} - regenerating")

    # Generate fresh LLM priors
    print(
        f"Generating fresh LLM priors ({'with' if use_hmdb_context else 'without'} HMDB context, {prior_strength} strength)..."
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

    # Generate LLM priors
    differential_priors = get_llm_differential_priors(
        priors=priors_data,
        condition=condition,
        use_hmdb_context=use_hmdb_context,
        prior_strength=prior_strength,
    )

    # Save to cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(differential_priors, f)
        print(f"✅ Saved LLM priors to cache: {cache_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save to cache: {e}")

    return differential_priors


def fit_bayesian_model_subsample(
    abundance_subsample, group_labels_subsample, differential_priors
):
    """Fit Bayesian model on subsample with LLM priors (same as qmd)."""

    # Filter abundance_subsample to only include metabolites we have priors for
    available_metabolites = list(differential_priors.keys())

    # DEBUG: Check what columns are available vs what we need
    print(f"\nDEBUG FILTERING:")
    print(
        f"abundance_subsample.columns[:10]: {abundance_subsample.columns[:10].tolist()}"
    )
    print(f"available_metabolites[:10]: {available_metabolites[:10]}")
    print(f"Columns in abundance_subsample: {len(abundance_subsample.columns)}")
    print(f"Available metabolites: {len(available_metabolites)}")

    # Check intersection
    intersection = set(abundance_subsample.columns) & set(available_metabolites)
    print(f"Intersection: {len(intersection)} metabolites")
    print(f"First 10 intersecting: {list(intersection)[:10]}")

    abundance_subsample_filtered = abundance_subsample[list(intersection)]
    available_metabolites = list(
        intersection
    )  # Update to only use actually available ones

    # Extract LLM prior parameters (only for available metabolites)
    llm_priors_mean = np.array(
        [differential_priors[m]["expected_log2fc"] for m in available_metabolites]
    )
    llm_priors_sd = np.array(
        [differential_priors[m]["prior_sd"] for m in available_metabolites]
    )

    # DEBUG: Print shapes before PyMC model
    print(f"\n=== PYMC DEBUG ===")
    print(f"abundance_subsample_filtered.shape: {abundance_subsample_filtered.shape}")
    print(f"group_labels_subsample.shape: {group_labels_subsample.shape}")
    print(f"available_metabolites length: {len(available_metabolites)}")
    print(f"llm_priors_mean.shape: {llm_priors_mean.shape}")
    print(f"llm_priors_sd.shape: {llm_priors_sd.shape}")
    print(f"=================")

    with pm.Model() as model:
        # Use LLM priors for beta (same structure as qmd)
        beta = pm.Normal(
            "beta",
            mu=llm_priors_mean,
            sigma=llm_priors_sd,
            shape=len(available_metabolites),
        )

        # Priors for metabolite-specific intercepts and standard deviations
        alpha = pm.Normal(
            "alpha",
            mu=abundance_subsample_filtered.mean().values,
            sigma=2.5,
            shape=len(available_metabolites),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=1, shape=len(available_metabolites)
        )

        # Expected value of the data
        print(
            f"DEBUG: alpha.shape should be ({len(available_metabolites)},), beta.shape should be ({len(available_metabolites)},)"
        )
        print(
            f"DEBUG: group_labels_subsample[:, None].shape should be ({len(group_labels_subsample)}, 1)"
        )
        print(
            "DEBUG: About to compute mu = alpha + beta * group_labels_subsample[:, None]"
        )
        mu = alpha + beta * group_labels_subsample[:, None]

        # Likelihood
        y_obs = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_subsample_filtered.values,
        )

        # Sample from posterior (reduced for speed)
        idata = pm.sample(
            1000, tune=1000, target_accept=0.9, cores=4, return_inferencedata=True
        )

    # Get posterior means for beta
    beta_posterior_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values

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


def fit_bayesian_hierarchical_model(
    abundance_subsample, group_labels_subsample, differential_priors
):
    """Fit a hierarchical Bayesian model on subsample with LLM group priors."""

    # --- Data Preparation ---
    available_metabolites = list(differential_priors.keys())
    intersection = list(set(abundance_subsample.columns) & set(available_metabolites))
    abundance_subsample_filtered = abundance_subsample[intersection]
    metabolites = abundance_subsample_filtered.columns.tolist()

    # Create group assignments for each metabolite
    groups = [differential_priors[m]["group"] for m in metabolites]
    unique_groups = sorted(list(set(groups)))
    group_idx = [unique_groups.index(g) for g in groups]

    # --- Model Definition ---
    with pm.Model() as model:
        # Hyperpriors for group-level means and standard deviations
        group_mu = pm.Normal("group_mu", mu=0, sigma=1.5, shape=len(unique_groups))
        group_sigma = pm.HalfNormal("group_sigma", sigma=1, shape=len(unique_groups))

        # Priors for metabolite-specific intercepts and standard deviations
        alpha = pm.Normal(
            "alpha",
            mu=abundance_subsample_filtered.mean().values,
            sigma=2.5,
            shape=len(metabolites),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=1, shape=len(metabolites)
        )

        # Hierarchical prior for beta (effect sizes)
        beta = pm.Normal(
            "beta",
            mu=group_mu[group_idx],
            sigma=group_sigma[group_idx],
            shape=len(metabolites),
        )

        # Expected value and Likelihood
        mu = alpha + beta * group_labels_subsample[:, None]
        y_obs = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=metabolite_sigmas,
            observed=abundance_subsample_filtered.values,
        )

        # Sample from posterior
        idata = pm.sample(1000, tune=1000, target_accept=0.9, cores=2, return_inferencedata=True)

    # Get posterior means for beta
    beta_posterior_means = idata.posterior["beta"].mean(dim=["chain", "draw"]).values

    return beta_posterior_means, metabolites


def run_benchmark(
    abundance_data,
    group_labels,
    metabolite_names,
    sample_data,
    ground_truth_log2fc,
    sample_sizes=[5, 10, 15, 20],
    n_replicates=10,
):
    """Run the main benchmark experiment comparing multiple methods."""

    # Load or generate all LLM priors once
    print("--- Loading/generating all LLM priors ---")
    priors = {
        "conservative_with_context": load_or_generate_llm_priors(
            metabolite_names, sample_data, use_hmdb_context=True, prior_strength="conservative"
        ),
        "hierarchical_with_context": load_or_generate_llm_priors(
            metabolite_names, sample_data, use_hmdb_context=True, prior_strength="hierarchical"
        ),
    }
    print("--- All priors loaded ---")

    common_metabolites = list(set.intersection(*[set(p.keys()) for p in priors.values()]))
    ground_truth_filtered = ground_truth_log2fc[common_metabolites]
    print(f"Using {len(common_metabolites)} metabolites common to all prior methods")

    results = []

    for n_per_group in sample_sizes:
        print(f"\n=== Testing sample size: {n_per_group} per group ===")

        for replicate in range(n_replicates):
            print(f"Replicate {replicate + 1}/{n_replicates}")

            abundance_subsample, group_labels_subsample = subsample_balanced(
                abundance_data, group_labels, n_per_group, random_state=replicate
            )

            try:
                # Empirical baseline
                empirical_log2fc_subset = calculate_ground_truth_log2fc(
                    abundance_subsample, group_labels_subsample
                )[common_metabolites]
                corr_empirical, _ = pearsonr(ground_truth_filtered, empirical_log2fc_subset)
                rmse_empirical = np.sqrt(mean_squared_error(ground_truth_filtered, empirical_log2fc_subset))
                results.append({
                    "sample_size": n_per_group, "replicate": replicate, "method": "empirical_subset",
                    "correlation": corr_empirical, "rmse": rmse_empirical
                })
                print(f"  Empirical subset: r={corr_empirical:.3f}, RMSE={rmse_empirical:.3f}")

                # Conservative LLM method
                prior_set_cons = priors["conservative_with_context"]
                beta_cons, mets_cons = fit_bayesian_model_subsample(
                    abundance_subsample, group_labels_subsample, prior_set_cons
                )
                df_cons = pd.DataFrame({'metabolite': mets_cons, 'beta': beta_cons}).set_index('metabolite').reindex(common_metabolites).dropna()
                corr_cons, _ = pearsonr(ground_truth_filtered.reindex(df_cons.index), df_cons['beta'])
                rmse_cons = np.sqrt(mean_squared_error(ground_truth_filtered.reindex(df_cons.index), df_cons['beta']))
                results.append({
                    "sample_size": n_per_group, "replicate": replicate, "method": "llm_conservative_with_context",
                    "correlation": corr_cons, "rmse": rmse_cons
                })
                print(f"  LLM Conservative: r={corr_cons:.3f}, RMSE={rmse_cons:.3f}")

                # Hierarchical LLM method
                prior_set_hier = priors["hierarchical_with_context"]
                beta_hier, mets_hier = fit_bayesian_hierarchical_model(
                    abundance_subsample, group_labels_subsample, prior_set_hier
                )
                df_hier = pd.DataFrame({'metabolite': mets_hier, 'beta': beta_hier}).set_index('metabolite').reindex(common_metabolites).dropna()
                corr_hier, _ = pearsonr(ground_truth_filtered.reindex(df_hier.index), df_hier['beta'])
                rmse_hier = np.sqrt(mean_squared_error(ground_truth_filtered.reindex(df_hier.index), df_hier['beta']))
                results.append({
                    "sample_size": n_per_group, "replicate": replicate, "method": "llm_hierarchical_with_context",
                    "correlation": corr_hier, "rmse": rmse_hier
                })
                print(f"  LLM Hierarchical: r={corr_hier:.3f}, RMSE={rmse_hier:.3f}")

            except Exception as e:
                print(f"  FATAL ERROR in replicate {replicate}: {type(e).__name__}: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                raise e

    return pd.DataFrame(results)


def analyze_benchmark_results(results_df, ground_truth_log2fc):
    """Analyze and visualize benchmark results."""

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    summary = results_df.groupby(["sample_size", "method"]).agg({"correlation": ["mean", "std"], "rmse": ["mean", "std"]}).round(3)
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
        groups = {
            "empirical": subset[subset["method"] == "empirical_subset"],
            "conservative": subset[subset["method"] == "llm_conservative_with_context"],
            "hierarchical": subset[subset["method"] == "llm_hierarchical_with_context"],
        }

        def print_ttest(group1, group2, name1, name2):
            if len(group1) > 0 and len(group2) > 0:
                corr_stat, corr_pval = ttest_ind(group1["correlation"], group2["correlation"])
                rmse_stat, rmse_pval = ttest_ind(group2["rmse"], group1["rmse"]) # Lower is better
                print(f"  {name1} vs {name2}:")
                print(f"    Correlation improvement: t={corr_stat:.3f}, p={corr_pval:.4f}")
                print(f"    RMSE improvement: t={rmse_stat:.3f}, p={rmse_pval:.4f}")

        print_ttest(groups["conservative"], groups["empirical"], "LLM Conservative", "Empirical")
        print_ttest(groups["hierarchical"], groups["empirical"], "LLM Hierarchical", "Empirical")
        print_ttest(groups["hierarchical"], groups["conservative"], "LLM Hierarchical", "LLM Conservative")

    return summary


def create_benchmark_visualizations(results_df, ground_truth_log2fc):
    """Create visualizations for the benchmark results."""

    plt.style.use("default")
    sns.set_palette("colorblind")
    results_df["Method"] = results_df["method"].replace({
        "empirical_subset": "Empirical",
        "llm_conservative_with_context": "LLM (Conservative)",
        "llm_hierarchical_with_context": "LLM (Hierarchical)",
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    sns.boxplot(data=results_df, x="sample_size", y="correlation", hue="Method", ax=axes[0])
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
    plt.savefig(f"{output_dir}/benchmark_prior_recovery.png", dpi=300, bbox_inches="tight")
    plt.show()



def main():
    """Main benchmark execution."""

    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set! This benchmark requires LLM access.")
        return

    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()

    print(f"Using {len(metabolite_names)} metabolites for benchmark")

    print("Calculating ground truth log2FC from full dataset...")
    ground_truth_log2fc = calculate_ground_truth_log2fc(abundance_data, group_labels)

    print("Running benchmark experiment...")
    results_df = run_benchmark(
        abundance_data,
        group_labels,
        metabolite_names,
        sample_data,
        ground_truth_log2fc,
        sample_sizes=[5, 10, 15, 20],
        n_replicates=8,
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
    summary = analyze_benchmark_results(results_df, ground_truth_log2fc)

    print("Creating visualizations...")
    create_benchmark_visualizations(results_df, ground_truth_log2fc)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(f"{output_dir}/benchmark_prior_recovery_results.csv", index=False)
    summary.to_csv(f"{output_dir}/benchmark_prior_recovery_summary.csv")

    print(f"\nResults saved to {output_dir}/")
    print("Benchmark complete!")

    return results_df, summary


if __name__ == "__main__":
    result = main()
    if result is not None:
        results_df, summary = result
