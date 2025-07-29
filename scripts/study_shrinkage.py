import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings

# Import the necessary functions from the benchmark script
from benchmark_prior_recovery import (
    load_mtbls1_data,
    calculate_ground_truth_lnfc,
    subsample_balanced,
    fit_uninformative_bayesian_baseline,
    fit_hierarchical_empirical_bayes,
)


def study_shrinkage(
    abundance_data,
    group_labels,
    ground_truth_lnfc,
    n_per_group=10,
    random_state=42,
):
    """Runs the shrinkage analysis and generates a comparison plot."""
    print(f"Creating a balanced subsample with n={n_per_group} per group...")
    abundance_subsample, group_labels_subsample = subsample_balanced(
        abundance_data, group_labels, n_per_group, random_state=random_state
    )

    # --- 1. Get Unshrunken Estimates ---
    print("Fitting uninformative Bayesian model (for unshrunken estimates)...")
    unshrunken_betas, unshrunken_mets = fit_uninformative_bayesian_baseline(
        abundance_subsample, group_labels_subsample
    )
    df_unshrunken = pd.DataFrame(
        {"metabolite": unshrunken_mets, "unshrunken_lnfc": unshrunken_betas}
    ).set_index("metabolite")

    # --- 2. Get Shrunken Estimates ---
    print("Fitting hierarchical empirical Bayes model (for shrunken estimates)...")
    shrunken_betas, shrunken_mets, shrinkage_info = fit_hierarchical_empirical_bayes(
        abundance_subsample, group_labels_subsample
    )
    df_shrunken = pd.DataFrame(
        {"metabolite": shrunken_mets, "shrunken_lnfc": shrunken_betas}
    ).set_index("metabolite")

    # --- 3. Combine and Analyze ---
    print("Combining results and analyzing shrinkage...")
    df_compare = df_unshrunken.join(df_shrunken, how="inner")
    df_compare = df_compare.join(
        ground_truth_lnfc.rename("ground_truth_lnfc"), how="inner"
    )

    # Extract shrinkage parameters
    global_mean = shrinkage_info["global_mean"]
    global_tau = shrinkage_info["global_tau"]

    print(f"\nHierarchical Model Learned Parameters:")
    print(f"  - Global Mean (Center of Shrinkage): {global_mean:.4f}")
    print(f"  - Global Tau (Between-Metabolite SD): {global_tau:.4f}")

    # --- 4. Quantify Performance ---
    mse_unshrunken = mean_squared_error(
        df_compare["ground_truth_lnfc"], df_compare["unshrunken_lnfc"]
    )
    mse_shrunken = mean_squared_error(
        df_compare["ground_truth_lnfc"], df_compare["shrunken_lnfc"]
    )

    print(f"\nPerformance vs. Ground Truth:")
    print(f"  - MSE (Unshrunken): {mse_unshrunken:.4f}")
    print(f"  - MSE (Shrunken):   {mse_shrunken:.4f}")

    if mse_shrunken < mse_unshrunken:
        print(
            "✅ CONCLUSION: Shrinkage is beneficial, as it reduces the overall error."
        )
    else:
        print(
            "⚠️  CONCLUSION: Shrinkage is NOT beneficial; it increases overall error (potential overshrinking)."
        )

    # --- 5. Visualize ---
    print("\nGenerating visualization...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df_compare["unshrunken_lnfc"],
        df_compare["shrunken_lnfc"],
        c=abs(df_compare["ground_truth_lnfc"]),
        cmap="viridis",
        alpha=0.7,
        s=50,
    )

    # Add y=x line for reference (no shrinkage)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.7, zorder=0, label="No Shrinkage (y=x)")

    # Add horizontal line at the learned global mean
    ax.axhline(
        global_mean,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Global Mean = {global_mean:.2f}",
    )

    ax.set_xlabel("Unshrunken lnFC (from Uninformative Model)", fontsize=12)
    ax.set_ylabel("Shrunken lnFC (from Hierarchical Model)", fontsize=12)
    ax.set_title(
        f"Effect of Hierarchical Shrinkage (n={n_per_group} per group)", fontsize=14
    )
    ax.legend()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute Ground Truth lnFC", fontsize=12)

    plt.tight_layout()
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shrinkage_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()

    return df_compare


def main():
    """Main execution function."""
    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()

    print("Calculating ground truth lnFC from full dataset...")
    ground_truth_lnfc = calculate_ground_truth_lnfc(abundance_data, group_labels)

    study_shrinkage(abundance_data, group_labels, ground_truth_lnfc)


if __name__ == "__main__":
    main()
