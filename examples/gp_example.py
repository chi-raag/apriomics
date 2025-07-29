#!/usr/bin/env python
"""
Example: Using chemistry priors for a Gaussian Process model

This example demonstrates how to use the apriomics package
to create chemical similarity priors and use them in a GP model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chem_priors.priors import (
    PriorData,
    get_smiles,
    generate_fingerprints,
    create_similarity_matrix,
    get_kernel,
)

# Import the GP model from the local examples directory
from models.gp import create_gp_model, fit_gp, predict_gp

# Set random seed for reproducibility
np.random.seed(42)


def main():
    """Run the example"""
    # Define example metabolites
    example_metabolites = [
        "Glucose",
        "Fructose",
        "Lactose",
        "Sucrose",
        "Maltose",
        "Glycine",
        "Alanine",
        "Valine",
        "Leucine",
        "Isoleucine",
        "Serine",
        "Threonine",
        "Proline",
        "Asparagine",
        "Glutamine",
    ]

    print("Step 1: Generate chemistry priors")
    # Option 1: Step by step approach
    priors = PriorData(dimensions=512)
    priors = get_smiles(priors, example_metabolites)
    priors = generate_fingerprints(priors)
    priors = create_similarity_matrix(priors)

    # Option 2: Using the run_pipeline function
    # priors = run_pipeline(dimensions=512, metabolites=example_metabolites)

    # Check how many SMILES were found
    print(
        f"Found SMILES for {priors.smiles_data['smiles'].notna().sum()} out of {len(example_metabolites)} metabolites"
    )

    # Only proceed with metabolites that have fingerprints
    metabolites_with_fps = priors.metabolite_names
    print(
        f"Using {len(metabolites_with_fps)} metabolites with fingerprints for modeling"
    )

    # Get the similarity matrix as a kernel
    kernel = get_kernel(priors, scale=1.0)

    # Generate synthetic data
    # For this example, we'll create a synthetic response variable
    # that's correlated with the similarity structure
    n = len(metabolites_with_fps)

    # Generate true function values using the kernel
    # This ensures the function has the structure implied by the kernel
    L = np.linalg.cholesky(kernel + 1e-6 * np.eye(n))
    f_true = L @ np.random.randn(n)

    # Add noise to create observed data
    noise_level = 0.2
    y_obs = f_true + noise_level * np.random.randn(n)

    # Feature matrix (just indices for this example)
    X = np.arange(n).reshape(-1, 1)

    print("\nStep 2: Fit Gaussian Process model using chemistry priors")
    # Create GP model - functional version
    gp_state = create_gp_model(kernel_scale=1.0, noise=0.2)

    # Fit GP model - functional version
    gp_state = fit_gp(gp_state, X, y_obs, kernel, num_warmup=500, num_samples=1000)

    # Get predictions - functional version
    predictions = predict_gp(gp_state)

    print("\nStep 3: Analyze results")
    # Create a DataFrame with results
    results_df = pd.DataFrame(
        {
            "metabolite": metabolites_with_fps,
            "true_value": f_true,
            "observed": y_obs,
            "predicted_mean": predictions["mean"],
            "predicted_std": predictions["std"],
            "lower_ci": predictions["lower_ci"],
            "upper_ci": predictions["upper_ci"],
        }
    )

    # Print results
    print(results_df[["metabolite", "observed", "predicted_mean", "predicted_std"]])

    # Calculate metrics
    mse = np.mean((predictions["mean"] - f_true) ** 2)
    print(f"\nMean Squared Error: {mse:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        range(n),
        predictions["mean"],
        yerr=predictions["std"],
        fmt="o",
        capsize=5,
        label="Predicted with std",
    )
    plt.plot(range(n), f_true, "x", color="red", label="True values")
    plt.plot(range(n), y_obs, "o", color="green", alpha=0.5, label="Observed values")

    # Fill confidence intervals
    plt.fill_between(
        range(n),
        predictions["lower_ci"],
        predictions["upper_ci"],
        alpha=0.2,
        color="blue",
        label="95% CI",
    )

    plt.title("Gaussian Process with Chemistry Priors")
    plt.xlabel("Metabolite Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/gp_predictions.png")
    plt.close()

    print("\nPlot saved to 'output/gp_predictions.png'")

    # Save results to CSV
    results_df.to_csv("output/gp_results.csv", index=False)
    print("Results saved to 'output/gp_results.csv'")

    # Visualize the similarity matrix/kernel
    plt.figure(figsize=(8, 6))
    plt.imshow(kernel, cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title("Chemical Similarity Kernel")
    plt.xlabel("Metabolite Index")
    plt.ylabel("Metabolite Index")
    plt.savefig("output/similarity_kernel.png")
    plt.close()

    print("Kernel visualization saved to 'output/similarity_kernel.png'")

    return results_df


if __name__ == "__main__":
    main()
