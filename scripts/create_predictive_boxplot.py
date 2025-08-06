"""
Create predictive performance benchmark boxplot with LLM-Adaptive-LASSO.

This script runs cross-validation to generate multiple performance measurements
for creating meaningful boxplot comparisons.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, ".")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from comprehensive_regularization_test import (
    load_and_prepare_data,
    fit_llm_adaptive_lasso,
    fit_llm_informed_bayesian,
    fit_standard_adaptive_lasso,
    fit_elastic_net,
    fit_standard_lasso,
    fit_ridge,
    fit_uninformative_bayesian,
)
from apriomics.priors import PriorData, get_llm_qualitative_predictions
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def run_cv_predictive_benchmark(n_folds=5):
    """Run cross-validation benchmark for predictive performance."""

    print("Loading data...")
    abundance_data, group_labels, metabolite_names, sample_data = (
        load_and_prepare_data()
    )

    X = abundance_data.values
    y = group_labels

    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Class balance: {np.bincount(y)}")

    # Get LLM qualitative predictions (cached if available)
    condition = """
Study: Type 2 diabetes mellitus vs healthy control
Context: Type 2 diabetes mellitus is the result of a combination of impaired insulin secretion with reduced insulin sensitivity of target tissues. In this study, NMR-based metabolomic analysis in conjunction with uni- and multivariate statistics was applied to examine the urinary metabolic changes in Human type 2 diabetes mellitus patients compared to the control group.
Sample type: Urine samples analyzed by NMR spectroscopy  
Patient population: Unmedicated Type 2 diabetes patients with good dietary control vs healthy controls
Expected changes: Look for metabolites altered in diabetes pathophysiology, particularly those related to glucose metabolism, insulin sensitivity, and urinary excretion patterns.
"""

    priors_data = PriorData(metabolite_names=metabolite_names)

    try:
        # Try to load cached predictions
        cache_path = "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gemini_2_0_flash_*metabolites.pkl"
        import glob
        import pickle

        cached_files = glob.glob(cache_path)

        if cached_files:
            with open(cached_files[0], "rb") as f:
                qualitative_predictions = pickle.load(f)
            print("✅ Using cached LLM predictions")
        else:
            print("🔄 Generating fresh LLM predictions...")
            qualitative_predictions = get_llm_qualitative_predictions(
                priors=priors_data,
                condition=condition,
                use_hmdb_context=False,
                model_name="gemini-2.0-flash",
            )
    except Exception as e:
        print(f"⚠️ Error with LLM predictions: {e}")
        # Create dummy predictions for testing
        qualitative_predictions = {
            name: {
                "prediction": np.random.choice(["increase", "decrease", "unchanged"]),
                "magnitude": "moderate",
                "confidence": np.random.uniform(0.3, 0.9),
                "reasoning": "Test",
            }
            for name in metabolite_names
        }

    # Define methods to test
    methods = [
        (
            "LLM-Adaptive-LASSO",
            lambda X_tr, y_tr, X_te: fit_llm_adaptive_lasso(
                X_tr, y_tr, X_te, metabolite_names, qualitative_predictions
            ),
        ),
        (
            "LLM-Informed-Bayesian",
            lambda X_tr, y_tr, X_te: fit_llm_informed_bayesian(
                X_tr, y_tr, X_te, metabolite_names, qualitative_predictions
            ),
        ),
        ("Standard-Adaptive-LASSO", fit_standard_adaptive_lasso),
        ("Elastic-Net", fit_elastic_net),
        ("Standard-LASSO", fit_standard_lasso),
        ("Ridge", fit_ridge),
        ("Uninformative-Bayesian", fit_uninformative_bayesian),
    ]

    # Stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    pbar = tqdm(total=n_folds * len(methods), desc="CV Benchmark")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for method_name, fit_func in methods:
            pbar.set_description(f"Fold {fold + 1}/{n_folds}: {method_name}")

            try:
                y_pred, y_prob, _ = fit_func(X_train_scaled, y_train, X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)

                results.append(
                    {"fold": fold, "method": method_name, "accuracy": acc, "auc": auc}
                )

            except Exception as e:
                pbar.write(f"❌ Error with {method_name} in fold {fold}: {e}")
                # Add NaN result to maintain structure
                results.append(
                    {
                        "fold": fold,
                        "method": method_name,
                        "accuracy": np.nan,
                        "auc": np.nan,
                    }
                )

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def create_predictive_boxplot(results_df):
    """Create boxplot visualization for predictive performance."""

    # Filter out any NaN results
    results_df = results_df.dropna()

    if len(results_df) == 0:
        print("❌ No valid results to plot")
        return

    # Create figure with subplots
    plt.style.use("default")
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy boxplot
    sns.boxplot(data=results_df, x="method", y="accuracy", ax=axes[0])
    axes[0].set_title(
        "Classification Accuracy by Method", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Method", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45, labelsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)  # Set reasonable y-axis limits

    # AUC boxplot
    sns.boxplot(data=results_df, x="method", y="auc", ax=axes[1])
    axes[1].set_title(
        "Area Under ROC Curve (AUC) by Method", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Method", fontsize=12)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45, labelsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)  # Set reasonable y-axis limits

    plt.tight_layout()

    # Save the plot
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        f"{output_dir}/predictive_performance_boxplot_with_llm_adaptive_lasso.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"📊 Boxplot saved to {output_dir}/predictive_performance_boxplot_with_llm_adaptive_lasso.png"
    )

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("PREDICTIVE PERFORMANCE SUMMARY")
    print("=" * 70)

    summary = (
        results_df.groupby("method")
        .agg(
            {
                "accuracy": ["mean", "std", "min", "max"],
                "auc": ["mean", "std", "min", "max"],
            }
        )
        .round(3)
    )

    print("\nAccuracy Statistics:")
    acc_summary = summary["accuracy"]
    for method in acc_summary.index:
        mean_acc = acc_summary.loc[method, "mean"]
        std_acc = acc_summary.loc[method, "std"]
        min_acc = acc_summary.loc[method, "min"]
        max_acc = acc_summary.loc[method, "max"]
        print(
            f"  {method:25s}: {mean_acc:.3f} ± {std_acc:.3f} (range: {min_acc:.3f}-{max_acc:.3f})"
        )

    print("\nAUC Statistics:")
    auc_summary = summary["auc"]
    for method in auc_summary.index:
        mean_auc = auc_summary.loc[method, "mean"]
        std_auc = auc_summary.loc[method, "std"]
        min_auc = auc_summary.loc[method, "min"]
        max_auc = auc_summary.loc[method, "max"]
        print(
            f"  {method:25s}: {mean_auc:.3f} ± {std_auc:.3f} (range: {min_auc:.3f}-{max_auc:.3f})"
        )

    # Highlight best performers
    best_acc_method = results_df.groupby("method")["accuracy"].mean().idxmax()
    best_auc_method = results_df.groupby("method")["auc"].mean().idxmax()
    best_acc_score = results_df.groupby("method")["accuracy"].mean().max()
    best_auc_score = results_df.groupby("method")["auc"].mean().max()

    print(f"\n🏆 Best Mean Accuracy: {best_acc_method} ({best_acc_score:.3f})")
    print(f"🏆 Best Mean AUC: {best_auc_method} ({best_auc_score:.3f})")

    return summary


def main():
    """Main execution."""

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set! Will use dummy predictions.")

    print("Running cross-validation predictive performance benchmark...")
    results_df = run_cv_predictive_benchmark(n_folds=5)

    if len(results_df) == 0:
        print("❌ No results generated!")
        return

    print("Creating boxplot visualization...")
    summary = create_predictive_boxplot(results_df)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    results_df.to_csv(
        f"{output_dir}/predictive_performance_cv_results.csv", index=False
    )
    summary.to_csv(f"{output_dir}/predictive_performance_cv_summary.csv")

    print(f"\n📁 Results saved to {output_dir}/")
    print("Predictive performance benchmark complete!")

    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()
