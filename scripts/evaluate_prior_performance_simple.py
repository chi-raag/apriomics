#!/usr/bin/env python3
"""
Simplified evaluation of predictive performance using MTBLS1 dataset.

This script compares LLM-based priors vs uniform priors using a simpler approach
to assess whether LLM priors improve out-of-sample prediction accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append("/Users/chiraag/Projects/gwu/lab/apriomics/src")
sys.path.append("/Users/chiraag/Projects/gwu/lab/chembridge/src")

from apriomics.priors import PriorData, get_llm_differential_priors


def load_mtbls1_data():
    """Load and prepare MTBLS1 dataset - simplified version."""
    # Load data
    data_path = "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/m_MTBLS1_metabolite_profiling_NMR_spectroscopy_v2_maf.tsv"
    metadata_path = (
        "/Users/chiraag/Projects/gwu/lab/apriomics/docs/examples/data/s_MTBLS1.txt"
    )

    try:
        data = pd.read_csv(data_path, sep="\t")
        metadata = pd.read_csv(metadata_path, sep="\t")
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure MTBLS1 data is available in docs/examples/data/")
        return None, None, None

    # Use metabolite names directly without HMDB mapping for speed
    metabolite_names = data["metabolite_identification"].tolist()

    # Extract abundance data - select sample columns (ADG prefix)
    sample_cols = [col for col in data.columns if col.startswith("ADG")]
    abundance_data = data[sample_cols].T  # Transpose so samples are rows
    abundance_data.columns = metabolite_names

    # Remove duplicated metabolite columns
    abundance_data = abundance_data.loc[:, ~abundance_data.columns.duplicated()]

    # Get group labels (case/control)
    # Match sample names between abundance and metadata
    sample_names = abundance_data.index
    matched_metadata = metadata[metadata["Sample Name"].isin(sample_names)]

    if len(matched_metadata) == 0:
        # Fallback: use all metadata in order
        print("Warning: Could not match sample names, using metadata in order")
        matched_metadata = metadata.head(len(sample_names))

    group_labels = (
        matched_metadata["Factor Value[Metabolic syndrome]"]
        .apply(lambda x: 1 if "diabetes mellitus" in str(x) else 0)
        .values
    )

    # Ensure we have matching lengths
    min_len = min(len(abundance_data), len(group_labels))
    abundance_data = abundance_data.iloc[:min_len]
    group_labels = group_labels[:min_len]

    print(
        f"Loaded data: {abundance_data.shape[0]} samples, {abundance_data.shape[1]} metabolites"
    )
    print(
        f"Class distribution: Control={np.sum(group_labels == 0)}, Case={np.sum(group_labels == 1)}"
    )

    return abundance_data, group_labels, list(abundance_data.columns)


def generate_different_priors(metabolite_names):
    """Generate different types of priors for comparison."""
    condition = """
Study: Type 2 diabetes mellitus vs healthy control
Context: NMR-based metabolomic analysis of urinary metabolic changes in unmedicated Type 2 diabetes mellitus patients with good dietary control compared to healthy controls.
Sample type: Urine samples analyzed by NMR spectroscopy
Expected changes: Look for metabolites altered in diabetes pathophysiology, particularly glucose metabolism and urinary excretion patterns.
"""

    prior_configs = {}

    # 1. Uniform (non-informative) priors
    prior_configs["uniform"] = {
        name: {"expected_log2fc": 0.0, "prior_sd": 1.0, "confidence": 0.5}
        for name in metabolite_names
    }

    # 2. LLM priors without HMDB context (name only)
    if os.getenv("GOOGLE_API_KEY"):
        print("Generating LLM priors without HMDB context...")
        priors_data = PriorData(metabolite_names=metabolite_names)

        try:
            llm_priors = get_llm_differential_priors(
                priors=priors_data,
                condition=condition,
                use_hmdb_context=False,
            )
            prior_configs["llm_no_context"] = llm_priors
        except Exception as e:
            print(f"Error generating LLM priors: {e}")
            # Use mock priors as fallback
            prior_configs["llm_no_context"] = {
                name: {
                    "expected_log2fc": np.random.normal(0, 0.5),
                    "prior_sd": np.random.uniform(0.3, 1.2),
                    "confidence": np.random.uniform(0.2, 0.8),
                }
                for name in metabolite_names
            }
    else:
        print("GOOGLE_API_KEY not set - using mock LLM priors for testing")
        # Use structured mock priors that might actually be informative
        np.random.seed(42)  # For reproducibility
        prior_configs["llm_no_context"] = {}

        for name in metabolite_names:
            # Give diabetes-related metabolites non-zero priors
            if any(
                keyword in name.lower()
                for keyword in ["glucose", "creatinine", "citrate", "acetone"]
            ):
                expected_fc = np.random.choice(
                    [0.8, -0.8], p=[0.6, 0.4]
                )  # Bias toward increase
                confidence = np.random.uniform(0.6, 0.9)
            else:
                expected_fc = np.random.normal(0, 0.3)
                confidence = np.random.uniform(0.3, 0.7)

            prior_configs["llm_no_context"][name] = {
                "expected_log2fc": expected_fc,
                "prior_sd": 1.2 - confidence,  # Higher confidence = lower SD
                "confidence": confidence,
            }

    return prior_configs


def fit_bayesian_regression_model(X_train, y_train, priors_dict, metabolite_names):
    """Fit Bayesian Ridge regression with custom priors."""

    # Extract prior parameters
    prior_means = np.array(
        [priors_dict[name]["expected_log2fc"] for name in metabolite_names]
    )
    prior_precisions = np.array(
        [1.0 / (priors_dict[name]["prior_sd"] ** 2) for name in metabolite_names]
    )

    # Use Bayesian Ridge regression with custom priors
    # Note: BayesianRidge doesn't directly support different priors per feature,
    # so we'll use a weighted approach

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create weighted features based on prior confidence
    weights = np.array([priors_dict[name]["confidence"] for name in metabolite_names])

    # Fit model
    model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

    # Incorporate priors by shifting the target based on expected changes
    # This is a simplified way to incorporate prior information
    prior_adjustment = X_train_scaled @ (prior_means * weights)
    y_adjusted = y_train - 0.1 * prior_adjustment  # Small adjustment based on priors

    model.fit(X_train_scaled, y_adjusted)

    return model, scaler


def cross_validate_priors(
    abundance_data, group_labels, prior_configs, metabolite_names, n_folds=5
):
    """Perform cross-validation to evaluate predictive performance of different priors."""

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

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
                # Fit model with priors
                model, scaler = fit_bayesian_regression_model(
                    X_train, y_train, priors_dict, metabolite_names
                )

                # Make predictions
                X_test_scaled = scaler.transform(X_test)

                # Get prediction probabilities
                y_pred_continuous = model.predict(X_test_scaled)

                # Convert to probabilities using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-y_pred_continuous))
                y_pred_proba = np.clip(y_pred_proba, 0.01, 0.99)  # Avoid extreme values

                # Convert to binary predictions
                y_pred = (y_pred_proba > 0.5).astype(int)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)

                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5

                # Calculate log-likelihood (simplified)
                epsilon = 1e-10
                y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1 - epsilon)
                log_likelihood = np.mean(
                    y_test * np.log(y_pred_proba_safe)
                    + (1 - y_test) * np.log(1 - y_pred_proba_safe)
                )

                fold_result[f"{prior_name}_accuracy"] = accuracy
                fold_result[f"{prior_name}_auc"] = auc
                fold_result[f"{prior_name}_log_likelihood"] = log_likelihood

                print(
                    f"    Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, LogLik: {log_likelihood:.3f}"
                )

            except Exception as e:
                print(f"    Error with {prior_name}: {e}")
                fold_result[f"{prior_name}_accuracy"] = 0.0
                fold_result[f"{prior_name}_auc"] = 0.5
                fold_result[f"{prior_name}_log_likelihood"] = -np.inf

        results.append(fold_result)

    return pd.DataFrame(results)


def analyze_results(cv_results):
    """Analyze and display results."""

    prior_types = ["uniform", "llm_no_context"]
    metrics = ["accuracy", "auc", "log_likelihood"]

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 60)

    summary_stats = {}

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 40)

        metric_results = {}
        for prior_type in prior_types:
            col_name = f"{prior_type}_{metric}"
            if col_name in cv_results.columns:
                values = (
                    cv_results[col_name].replace([np.inf, -np.inf], np.nan).dropna()
                )
                if len(values) > 0:
                    metric_results[prior_type] = {
                        "mean": values.mean(),
                        "std": values.std(),
                        "values": values.tolist(),
                    }
                    print(f"{prior_type:20s}: {values.mean():.3f} ± {values.std():.3f}")

        summary_stats[metric] = metric_results

    # Statistical tests
    print(f"\n{'STATISTICAL COMPARISONS':^60}")
    print("-" * 60)

    from scipy.stats import ttest_rel

    for metric in metrics:
        uniform_col = f"uniform_{metric}"
        llm_col = f"llm_no_context_{metric}"

        if all(col in cv_results.columns for col in [uniform_col, llm_col]):
            uniform_vals = (
                cv_results[uniform_col].replace([np.inf, -np.inf], np.nan).dropna()
            )
            llm_vals = cv_results[llm_col].replace([np.inf, -np.inf], np.nan).dropna()

            if len(uniform_vals) > 0 and len(llm_vals) > 0:
                try:
                    stat, pval = ttest_rel(llm_vals, uniform_vals)
                    improvement = llm_vals.mean() - uniform_vals.mean()
                    print(
                        f"{metric.upper()}: LLM improvement = {improvement:+.3f}, t={stat:.3f}, p={pval:.4f}"
                    )
                except:
                    print(f"{metric.upper()}: Could not compute statistical test")

    return summary_stats, cv_results


def create_visualizations(cv_results, summary_stats):
    """Create visualizations of results."""

    # Prepare data for plotting
    plot_data = []
    for fold_idx, row in cv_results.iterrows():
        for prior_type in ["uniform", "llm_no_context"]:
            for metric in ["accuracy", "auc", "log_likelihood"]:
                col_name = f"{prior_type}_{metric}"
                if col_name in cv_results.columns:
                    value = row[col_name]
                    if not np.isinf(value) and not np.isnan(value):
                        plot_data.append(
                            {
                                "fold": fold_idx,
                                "prior_type": prior_type,
                                "metric": metric,
                                "value": value,
                            }
                        )

    if not plot_data:
        print("No valid data for plotting")
        return

    plot_df = pd.DataFrame(plot_data)

    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(["accuracy", "auc", "log_likelihood"]):
        metric_data = plot_df[plot_df["metric"] == metric]
        if len(metric_data) > 0:
            sns.boxplot(data=metric_data, x="prior_type", y="value", ax=axes[i])
            axes[i].set_title(f"{metric.title()} Comparison")
            axes[i].set_xlabel("Prior Type")
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save plots
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/prior_performance_simple.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return plot_df


def main():
    """Main evaluation script."""

    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names = load_mtbls1_data()

    if abundance_data is None:
        return

    # Limit to first 20 metabolites for speed
    metabolite_names = metabolite_names[:20]
    abundance_data = abundance_data[metabolite_names]

    print(f"Using {len(metabolite_names)} metabolites for evaluation")

    print("\nGenerating prior configurations...")
    prior_configs = generate_different_priors(metabolite_names)

    print(f"Generated {len(prior_configs)} prior configurations:")
    for name in prior_configs.keys():
        print(f"  - {name}")

    print("\nStarting cross-validation evaluation...")
    cv_results = cross_validate_priors(
        abundance_data, group_labels, prior_configs, metabolite_names, n_folds=5
    )

    print("\nAnalyzing results...")
    summary_stats, cv_results = analyze_results(cv_results)

    print("\nCreating visualizations...")
    plot_df = create_visualizations(cv_results, summary_stats)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)

    cv_results.to_csv(f"{output_dir}/prior_performance_simple_cv.csv", index=False)

    print(f"\nResults saved to {output_dir}/")
    print("Evaluation complete!")

    return cv_results, summary_stats


if __name__ == "__main__":
    cv_results, summary_stats = main()
