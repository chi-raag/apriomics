"""
Test and Visualize the LLM-Informed Hierarchical Model.

This script provides a focused test of the LLM-Informed Hierarchical model.
It compares its performance against the noisy empirical log fold changes
calculated from a small data subsample.

Methodology:
1.  Take a small, representative subsample of the MTBLS1 dataset (n=10 per group).
2.  Calculate the "empirical lnFC" directly from this small subsample. These estimates
    are expected to be noisy and have high variance.
3.  Fit the sophisticated LLM-Informed Hierarchical model on the same subsample.
    This model uses LLM predictions to group metabolites and applies partial pooling
    within those groups.
4.  Calculate the "ground truth" lnFC from the full dataset for a fair comparison.
5.  Generate a scatter plot comparing:
    - The noisy empirical estimates vs. the ground truth.
    - The shrunken hierarchical estimates vs. the ground truth.
6.  This visualization will demonstrate how the hierarchical model's informed
    shrinkage pulls the noisy estimates closer to the true values.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import necessary functions from the main benchmark script
from benchmark_prior_recovery import (
    load_mtbls1_data,
    calculate_ground_truth_lnfc,
    subsample_balanced,
    fit_llm_informed_hierarchical_model,
    load_or_generate_qualitative_predictions,
    apply_prior_strength_mapping,
)

warnings.filterwarnings("ignore")


def calculate_empirical_lnfc(abundance_subsample, group_labels_subsample):
    """Calculates the simple, noisy empirical lnFC from a data subsample."""
    control_indices = group_labels_subsample == 0
    case_indices = group_labels_subsample == 1

    control_means = abundance_subsample[control_indices].mean()
    case_means = abundance_subsample[case_indices].mean()

    min_val = abundance_subsample.values[abundance_subsample.values > 0].min() * 0.01

    lnfc = np.log((case_means + min_val) / (control_means + min_val))
    return lnfc.rename("empirical_lnfc")


def run_focused_test(
    abundance_data,
    group_labels,
    metabolite_names,
    sample_data,
    ground_truth_lnfc,
    n_per_group=10,
    random_state=42,
):
    """Runs the focused test and generates the comparison plot."""

    print(f"Creating a balanced subsample with n={n_per_group} per group...")
    abundance_subsample, group_labels_subsample = subsample_balanced(
        abundance_data, group_labels, n_per_group, random_state=random_state
    )

    # --- 1. Calculate Noisy Empirical lnFC from Subsample ---
    print("Calculating empirical lnFC from the small subsample...")
    empirical_lnfc = calculate_empirical_lnfc(
        abundance_subsample, group_labels_subsample
    )

    # --- 2. Get LLM Priors ---
    print("Loading LLM priors to inform the hierarchical model...")
    qualitative_preds = load_or_generate_qualitative_predictions(
        metabolite_names,
        sample_data,
        use_hmdb_context=False,
        model_name="gemini-2.0-flash",
        temperature=0.0,
    )
    llm_priors = apply_prior_strength_mapping(qualitative_preds, "moderate")

    # --- 3. Fit LLM-Informed Hierarchical Model ---
    print("Fitting the LLM-Informed Hierarchical Model...")
    hierarchical_betas, hierarchical_mets = fit_llm_informed_hierarchical_model(
        abundance_subsample, group_labels_subsample, llm_priors
    )
    df_hierarchical = pd.DataFrame(
        {"metabolite": hierarchical_mets, "hierarchical_lnfc": hierarchical_betas}
    ).set_index("metabolite")

    # --- 4. Combine for Plotting ---
    df_compare = pd.DataFrame(empirical_lnfc).join(df_hierarchical, how="inner")
    df_compare = df_compare.join(
        ground_truth_lnfc.rename("ground_truth_lnfc"), how="inner"
    )

    # --- 5. Visualize ---
    print("Generating visualization...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Noisy Empirical vs. Ground Truth
    ax.scatter(
        df_compare["ground_truth_lnfc"],
        df_compare["empirical_lnfc"],
        alpha=0.6,
        label=f"Empirical lnFC (from n={n_per_group * 2} subsample)",
        s=50,
        facecolors="none",
        edgecolors="red",
    )

    # Plot Shrunken Hierarchical vs. Ground Truth
    ax.scatter(
        df_compare["ground_truth_lnfc"],
        df_compare["hierarchical_lnfc"],
        alpha=0.8,
        label="LLM-Informed Hierarchical lnFC",
        s=50,
        color="blue",
    )

    # Add y=x line for reference
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.7, zorder=0, label="Perfect Correlation (y=x)")

    ax.set_xlabel("Ground Truth lnFC (from full dataset)", fontsize=12)
    ax.set_ylabel("Estimated lnFC (from subsample)", fontsize=12)
    ax.set_title(
        "LLM-Informed Hierarchical Model Shrinks Noisy Estimates Toward Truth",
        fontsize=14,
    )
    ax.legend()

    plt.tight_layout()
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hierarchical_shrinkage_test.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()


def main():
    """Main execution function."""
    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()

    print("Calculating ground truth lnFC from full dataset...")
    ground_truth_lnfc = calculate_ground_truth_lnfc(abundance_data, group_labels)

    run_focused_test(
        abundance_data, group_labels, metabolite_names, sample_data, ground_truth_lnfc
    )


if __name__ == "__main__":
    main()
