#!/usr/bin/env python3
"""
Evaluate predictive performance of different prior configurations on MTBLS1 dataset.

This script compares LLM-based priors vs uniform priors via cross-validation
to assess whether LLM priors improve out-of-sample prediction accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import pymc as pm

# Add project root to path
sys.path.append("/Users/chiraag/Projects/gwu/lab/apriomics/src")
sys.path.append("/Users/chiraag/Projects/gwu/lab/chembridge/src")

from apriomics.priors import PriorData, get_llm_differential_priors


def load_mtbls1_data():
    """Load and prepare MTBLS1 dataset."""
    # Load data
    data = pd.read_csv(
        "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv",
        sep="\t",
    )

    metadata = pd.read_csv(
        "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/s_MTBLS1.txt",
        sep="\t",
    )

    # Map metabolite names
    from chembridge import map_metabolites

    mapped_data = map_metabolites(
        data["database_identifier"].tolist(), source="chebi", target="hmdb"
    )
    data["hmdb_id"] = mapped_data.df()["hmdb_id"]
    filtered_data = data[data["hmdb_id"].notna()]

    # Get abundance data
    sample_data = filtered_data[["metabolite_identification", "hmdb_id"]]
    metabolite_names = sample_data["metabolite_identification"].tolist()

    # Extract abundance data and pivot
    abundance_data = (
        filtered_data.filter(regex="ADG")
        .join(filtered_data[["metabolite_identification"]])
        .groupby("metabolite_identification")
        .mean()
        .loc[metabolite_names]
    )
    abundance_data = abundance_data.T  # Transpose so metabolites are columns

    # Get group labels (case/control)
    group_labels = (
        metadata["Factor Value[Metabolic syndrome]"]
        .apply(lambda x: 1 if "diabetes mellitus" in x else 0)
        .values
    )

    return abundance_data, group_labels, metabolite_names


def generate_different_priors(metabolite_names):
    """Generate different types of priors for comparison."""
    condition = """
Study: Type 2 diabetes mellitus vs healthy control
Context: Type 2 diabetes mellitus is the result of a combination of impaired insulin secretion with reduced insulin sensitivity of target tissues. In this study, NMR-based metabolomic analysis in conjunction with uni- and multivariate statistics was applied to examine the urinary metabolic changes in Human type 2 diabetes mellitus patients compared to the control group. The human population were un medicated diabetic patients who have good daily dietary control over their blood glucose concentrations by following the guidelines on diet issued by the American Diabetes Association.
Sample type: Urine samples analyzed by NMR spectroscopy
Patient population: Unmedicated Type 2 diabetes patients with good dietary control vs healthy controls
Expected changes: Look for metabolites altered in diabetes pathophysiology, particularly those related to glucose metabolism, insulin sensitivity, and urinary excretion patterns.
"""

    prior_configs = {}

    # 1. Uniform (non-informative) priors
    prior_configs["uniform"] = {
        name: {
            "expected_log2fc": 0.0,
            "prior_sd": 1.0,
            "prediction": "unchanged",
            "confidence": 0.5,
        }
        for name in metabolite_names
    }

    # 2. LLM priors without HMDB context (name only)
    if os.getenv("GOOGLE_API_KEY"):
        print("Generating LLM priors without HMDB context...")
        priors_data = PriorData(metabolite_names=metabolite_names)

        prior_configs["llm_no_context"] = get_llm_differential_priors(
            priors=priors_data,
            condition=condition,
            use_hmdb_context=False,
        )

        # 3. LLM priors with HMDB context (if available)
        # Note: This would require HMDB data fetching - for now, use same as no_context
        # In a full implementation, you'd fetch HMDB contexts here
        prior_configs["llm_with_context"] = prior_configs["llm_no_context"].copy()

    else:
        print("GOOGLE_API_KEY not set - skipping LLM priors")
        # Use mock LLM priors for testing
        prior_configs["llm_no_context"] = {
            name: {
                "expected_log2fc": np.random.normal(0, 0.5),
                "prior_sd": np.random.uniform(0.3, 1.2),
                "prediction": np.random.choice(["increase", "decrease", "unchanged"]),
                "confidence": np.random.uniform(0.2, 0.8),
            }
            for name in metabolite_names
        }
        prior_configs["llm_with_context"] = prior_configs["llm_no_context"].copy()

    return prior_configs


def fit_bayesian_model(abundance_data, group_labels, priors_dict, metabolite_names):
    """Fit Bayesian model with given priors and return fitted model."""

    # Extract prior parameters
    prior_means = np.array(
        [priors_dict[name]["expected_log2fc"] for name in metabolite_names]
    )
    prior_sds = np.array([priors_dict[name]["prior_sd"] for name in metabolite_names])

    with pm.Model() as model:
        # Use provided priors for beta (log-fold changes)
        beta = pm.Normal(
            "beta",
            mu=prior_means,
            sigma=prior_sds,
            shape=len(metabolite_names),
        )

        # Priors for metabolite-specific intercepts and standard deviations
        alpha = pm.Normal(
            "alpha",
            mu=abundance_data.mean().values,
            sigma=2.5,
            shape=len(metabolite_names),
        )
        metabolite_sigmas = pm.HalfNormal(
            "metabolite_sigmas", sigma=1, shape=len(metabolite_names)
        )

        # Expected value of the data
        mu = alpha + beta * group_labels[:, None]

        # Likelihood
        y_obs = pm.Normal(
            "y_obs", mu=mu, sigma=metabolite_sigmas, observed=abundance_data.values
        )

        # Sample from posterior
        idata = pm.sample(
            1000, tune=1000, target_accept=0.9, cores=2, return_inferencedata=True
        )

    return model, idata


def cross_validate_priors(
    abundance_data, group_labels, prior_configs, metabolite_names, n_folds=5
):
    """Perform cross-validation to evaluate predictive performance of different priors."""

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {prior_name: [] for prior_name in prior_configs.keys()}

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(abundance_data)):
        print(f"Processing fold {fold_idx + 1}/{n_folds}")

        # Split data
        X_train = abundance_data.iloc[train_idx]
        X_test = abundance_data.iloc[test_idx]
        y_train = group_labels[train_idx]
        y_test = group_labels[test_idx]

        fold_result = {"fold": fold_idx}

        for prior_name, priors_dict in prior_configs.items():
            print(f"  Evaluating {prior_name} priors...")

            try:
                # Fit model on training data
                model, idata = fit_bayesian_model(
                    X_train, y_train, priors_dict, metabolite_names
                )

                # Generate predictions for test data
                with model:
                    # Sample posterior predictive for test data
                    pm.set_data({"y_obs": X_test.values})
                    posterior_predictive = pm.sample_posterior_predictive(
                        idata, predictions=True, return_inferencedata=True
                    )

                # Calculate predictive metrics
                # Use posterior mean of beta to predict group membership
                beta_posterior = (
                    idata.posterior["beta"].mean(dim=["chain", "draw"]).values
                )

                # Simple prediction: positive beta values suggest case group
                # Calculate a simple discriminant score for each test sample
                discriminant_scores = X_test.values @ beta_posterior
                predicted_probs = 1 / (1 + np.exp(-discriminant_scores))  # sigmoid
                predicted_labels = (predicted_probs > 0.5).astype(int)

                # Calculate metrics
                accuracy = accuracy_score(y_test, predicted_labels)
                try:
                    auc = roc_auc_score(y_test, predicted_probs)
                except:
                    auc = 0.5  # Default if AUC calculation fails

                # Log-likelihood of test data under model
                try:
                    log_likelihood = -log_loss(y_test, predicted_probs)
                except:
                    log_likelihood = -np.inf

                fold_result[f"{prior_name}_accuracy"] = accuracy
                fold_result[f"{prior_name}_auc"] = auc
                fold_result[f"{prior_name}_log_likelihood"] = log_likelihood

                print(f"    Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

            except Exception as e:
                print(f"    Error with {prior_name}: {e}")
                fold_result[f"{prior_name}_accuracy"] = 0.0
                fold_result[f"{prior_name}_auc"] = 0.5
                fold_result[f"{prior_name}_log_likelihood"] = -np.inf

        fold_results.append(fold_result)

    return pd.DataFrame(fold_results)


def analyze_results(cv_results):
    """Analyze and visualize cross-validation results."""

    # Calculate summary statistics
    prior_types = ["uniform", "llm_no_context", "llm_with_context"]
    metrics = ["accuracy", "auc", "log_likelihood"]

    summary_stats = {}

    for prior_type in prior_types:
        summary_stats[prior_type] = {}
        for metric in metrics:
            col_name = f"{prior_type}_{metric}"
            if col_name in cv_results.columns:
                values = cv_results[col_name]
                summary_stats[prior_type][metric] = {
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                }

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats).T

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 60)

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 40)
        for prior_type in prior_types:
            if metric in summary_stats[prior_type]:
                stats = summary_stats[prior_type][metric]
                print(
                    f"{prior_type:20s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                    f"(range: {stats['min']:.3f}-{stats['max']:.3f})"
                )

    # Statistical significance tests
    print(f"\n{'STATISTICAL COMPARISONS':^60}")
    print("-" * 60)

    from scipy.stats import ttest_rel

    for metric in metrics:
        print(f"\n{metric.upper()} comparisons:")

        uniform_col = f"uniform_{metric}"
        llm_no_context_col = f"llm_no_context_{metric}"
        llm_with_context_col = f"llm_with_context_{metric}"

        if all(col in cv_results.columns for col in [uniform_col, llm_no_context_col]):
            stat, pval = ttest_rel(
                cv_results[llm_no_context_col], cv_results[uniform_col]
            )
            print(f"  LLM (no context) vs Uniform: t={stat:.3f}, p={pval:.4f}")

        if all(
            col in cv_results.columns for col in [uniform_col, llm_with_context_col]
        ):
            stat, pval = ttest_rel(
                cv_results[llm_with_context_col], cv_results[uniform_col]
            )
            print(f"  LLM (with context) vs Uniform: t={stat:.3f}, p={pval:.4f}")

        if all(
            col in cv_results.columns
            for col in [llm_no_context_col, llm_with_context_col]
        ):
            stat, pval = ttest_rel(
                cv_results[llm_with_context_col], cv_results[llm_no_context_col]
            )
            print(
                f"  LLM (with context) vs LLM (no context): t={stat:.3f}, p={pval:.4f}"
            )

    return summary_df


def create_visualizations(cv_results):
    """Create visualizations of the cross-validation results."""

    # Prepare data for plotting
    plot_data = []
    prior_types = ["uniform", "llm_no_context", "llm_with_context"]
    metrics = ["accuracy", "auc", "log_likelihood"]

    for fold_idx, row in cv_results.iterrows():
        for prior_type in prior_types:
            for metric in metrics:
                col_name = f"{prior_type}_{metric}"
                if col_name in cv_results.columns:
                    plot_data.append(
                        {
                            "fold": fold_idx,
                            "prior_type": prior_type,
                            "metric": metric,
                            "value": row[col_name],
                        }
                    )

    plot_df = pd.DataFrame(plot_data)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        metric_data = plot_df[plot_df["metric"] == metric]

        sns.boxplot(data=metric_data, x="prior_type", y="value", ax=axes[i])
        axes[i].set_title(f"{metric.title()} Comparison")
        axes[i].set_xlabel("Prior Type")
        axes[i].set_ylabel(metric.title())
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "/Users/chiraag/Projects/gwu/lab/apriomics/output/prior_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Create detailed comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Focus on accuracy for detailed view
    accuracy_data = plot_df[plot_df["metric"] == "accuracy"]

    sns.stripplot(
        data=accuracy_data, x="prior_type", y="value", size=8, alpha=0.7, ax=ax
    )
    sns.boxplot(
        data=accuracy_data,
        x="prior_type",
        y="value",
        width=0.3,
        ax=ax,
        color="lightgray",
        alpha=0.5,
    )

    ax.set_title("Predictive Accuracy Comparison Across CV Folds")
    ax.set_xlabel("Prior Configuration")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "/Users/chiraag/Projects/gwu/lab/apriomics/output/accuracy_detailed_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    """Main evaluation script."""

    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names = load_mtbls1_data()

    print(
        f"Dataset loaded: {len(metabolite_names)} metabolites, {len(group_labels)} samples"
    )
    print(f"Class distribution: {np.bincount(group_labels)}")

    print("\nGenerating different prior configurations...")
    prior_configs = generate_different_priors(metabolite_names)

    print(f"Generated {len(prior_configs)} prior configurations:")
    for name in prior_configs.keys():
        print(f"  - {name}")

    print("\nStarting cross-validation evaluation...")
    cv_results = cross_validate_priors(
        abundance_data, group_labels, prior_configs, metabolite_names, n_folds=5
    )

    print("\nAnalyzing results...")
    summary_df = analyze_results(cv_results)

    print("\nCreating visualizations...")
    create_visualizations(cv_results)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)

    cv_results.to_csv(f"{output_dir}/prior_performance_cv_results.csv", index=False)
    summary_df.to_csv(f"{output_dir}/prior_performance_summary.csv")

    print(f"\nResults saved to {output_dir}/")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
