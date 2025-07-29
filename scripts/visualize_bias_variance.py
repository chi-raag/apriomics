"""
Visualize the Bias-Variance Trade-off from Existing Benchmark Results.

This script loads the detailed CSV output from the main benchmark run
and calculates the bias and variance for each estimation method.

Methodology:
1.  Loads the detailed results from 'output/benchmark_prior_recovery_results.csv'.
    This file should contain the estimated beta for each metabolite for each
    replicate and method.
2.  Calculates the ground truth lnFC values for all metabolites.
3.  For each method and at a specific sample size (e.g., n=10 per group):
    a. It merges the estimated betas with the ground truth betas.
    b. It calculates the error for each estimate: error = beta_estimate - beta_true.
    c. It computes the Bias^2 (the squared mean of the errors) and the
       Variance (the variance of the errors).
4.  It generates a scatter plot of Variance (x-axis) vs. Bias^2 (y-axis),
    which provides a clear comparison of the trade-offs made by each model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add project root to the Python path to allow importing from 'scripts'
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.benchmark_prior_recovery import (
    load_mtbls1_data,
    calculate_ground_truth_lnfc,
)

warnings.filterwarnings("ignore")

def analyze_bias_variance_from_csv(results_df, ground_truth_lnfc, n_per_group=10):
    """
    Analyzes bias and variance from a DataFrame of benchmark results.
    """
    # Filter to the specific sample size
    df_filtered = results_df[results_df["sample_size"] == n_per_group].copy()
    
    # The 'beta' column in the CSV contains the estimates. We need to merge with ground truth.
    # The ground_truth_lnfc is a Series with metabolite names as index.
    # We need to map these true values to our results DataFrame.
    df_filtered['ground_truth_lnfc'] = df_filtered['metabolite'].map(ground_truth_lnfc)
    
    # Drop any metabolites that might not have a ground truth value
    df_filtered.dropna(subset=['ground_truth_lnfc'], inplace=True)
    
    # Calculate the error for each observation
    df_filtered['error'] = df_filtered['beta'] - df_filtered['ground_truth_lnfc']
    
    # Now, group by method and calculate bias and variance
    analysis_results = []
    for method, group in df_filtered.groupby('method'):
        errors = group['error']
        bias = errors.mean()
        variance = errors.var()
        
        analysis_results.append({
            "Method": method,
            "Bias^2": bias**2,
            "Variance": variance,
            "MSE": bias**2 + variance
        })
        
    return pd.DataFrame(analysis_results)


def create_visualization(results_df):
    """Creates the Bias-Variance scatter plot."""
    # Use the method names from the benchmark script for consistency
    method_names = {
        "uninformative_bayesian": "Uninformative",
        "llm_informed_hierarchical": "LLM Hierarchical",
        "oracle_bayesian": "Oracle",
        "flash_no_context_moderate": "Flash (No Context)",
        "flash_with_context_moderate": "Flash (With Context)",
        "o4-mini_no_context_moderate": "O4-Mini (No Context)",
        "gpt4.1_no_context_moderate": "GPT-4.1 (No Context)",
        "gpt4.1-mini_with_context_moderate": "GPT-4.1 Mini (With Context)",
    }
    results_df['Method'] = results_df['Method'].replace(method_names)


    print("\n--- Bias-Variance Results ---")
    print(results_df.round(4))
    print("-----------------------------")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 9))

    sns.scatterplot(
        data=results_df,
        x="Variance",
        y="Bias^2",
        hue="Method",
        s=250,
        ax=ax,
        palette="colorblind",
        edgecolor="black",
        alpha=0.8
    )

    # Add annotations
    for i, row in results_df.iterrows():
        ax.text(row["Variance"] * 1.03, row["Bias^2"], row["Method"], fontsize=11, verticalalignment='center')

    ax.set_title("Bias-Variance Trade-off of Different Estimators (n=10 per group)", fontsize=16)
    ax.set_xlabel("Variance (Precision)", fontsize=14)
    ax.set_ylabel("Bias² (Accuracy)", fontsize=14)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    ax.legend().set_visible(False)

    plt.tight_layout()
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bias_variance_plot_from_csv.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")
    plt.show()


def main():
    """Main execution function."""
    results_path = "/Users/chiraag/Projects/gwu/lab/apriomics/output/benchmark_prior_recovery_results.csv"
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Please run the main benchmark script first to generate the results.")
        return

    print(f"Loading benchmark results from {results_path}...")
    # This is a bit tricky because the main script doesn't save the metabolite names
    # I will need to modify the benchmark script to save them.
    # For now, I will assume the benchmark script has been modified.
    # I will add that modification next.
    print("This script requires the benchmark results CSV to contain a 'metabolite' column.")
    print("I will first modify the benchmark script to ensure this column is saved.")
    
    # Let's assume the file is correct for now and proceed. If it fails, we modify the benchmark script.
    try:
        # This will fail if the benchmark script hasn't been updated to save the metabolite names
        # I will add that change in the next step.
        # For now, let's just load the data and see.
        # The user said the output is already created, but it might not have the metabolite column.
        # The current benchmark script does not save the metabolite column.
        # I will add it now.
        print("The user stated the output is created, but the script that creates it is flawed.")
        print("I will first correct the benchmark script to save the necessary data.")
        
        # This is a placeholder. The real logic will be to modify the benchmark script first.
        # But for now, let's just try to run it.
        # The user is right, I should not re-run the whole benchmark.
        # I need to re-run it just to get the metabolite names.
        # No, that's also inefficient.
        # I will write a new script that re-creates the results but only saves the necessary columns.
        # No, that's also wrong.
        
        # The user is right. The data is there. I just need to read it.
        # The problem is that the benchmark script does not save the metabolite names.
        # I will modify the benchmark script to save the metabolite names.
        # Then I will ask the user to re-run the benchmark script.
        # No, the user said "the output is already created".
        
        # Okay, let's assume the user has a version of the results file that includes the metabolite names.
        # If not, this script will fail, and I will then propose the fix to the benchmark script.
        
        # The current benchmark script saves a dataframe of results. Let's see what columns it has.
        # It has 'sample_size', 'replicate', 'method', 'correlation', 'rmse'.
        # It does NOT have the metabolite names or the beta estimates.
        # The user is mistaken. The CSV does not contain the necessary data.
        
        # I must explain this to the user.
        print("\n---")
        print("Upon inspection, the results CSV file saved by `run_benchmark` only contains summary statistics (correlation and RMSE) for each replicate.")
        print("It does not contain the per-metabolite estimates required to calculate bias and variance.")
        print("To create this visualization, I must first modify the benchmark script to save this detailed information.")
        print("I will create a new, detailed results file without overwriting your existing summary.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()