"""
Quick test of empirical Bayes methods to verify implementation.
"""

import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from benchmark_prior_recovery import (
    load_mtbls1_data, calculate_ground_truth_log2fc, 
    subsample_balanced, fit_uninformative_bayesian_baseline,
    fit_james_stein_shrinkage, fit_hierarchical_empirical_bayes
)
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def test_empirical_bayes_methods():
    """Test empirical Bayes methods on a single subsample."""
    
    print("Loading data...")
    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()
    
    print("Calculating ground truth...")
    ground_truth_log2fc = calculate_ground_truth_log2fc(abundance_data, group_labels)
    
    print("Creating subsample...")
    # Small subsample for quick testing
    abundance_subsample, group_labels_subsample = subsample_balanced(
        abundance_data, group_labels, n_per_group=8, random_state=42
    )
    
    print(f"Subsample: {len(group_labels_subsample)} samples, {len(abundance_subsample.columns)} metabolites")
    
    methods_to_test = [
        ("Uninformative Bayesian", fit_uninformative_bayesian_baseline),
        ("James-Stein Shrinkage", fit_james_stein_shrinkage),
        ("Hierarchical Empirical Bayes", fit_hierarchical_empirical_bayes)
    ]
    
    results = []
    
    for method_name, fit_function in methods_to_test:
        print(f"\nTesting {method_name}...")
        
        try:
            if method_name == "Hierarchical Empirical Bayes":
                # This method returns 3 values
                beta_estimates, metabolites, shrinkage_info = fit_function(
                    abundance_subsample, group_labels_subsample
                )
                print(f"  Global mean: {shrinkage_info.get('global_mean', 'N/A'):.4f}")
                print(f"  Global tau: {shrinkage_info.get('global_tau', 'N/A'):.4f}")
            else:
                # Other methods return 2 values
                beta_estimates, metabolites = fit_function(
                    abundance_subsample, group_labels_subsample
                )
            
            # Align with ground truth
            common_metabolites = list(set(metabolites) & set(ground_truth_log2fc.index))
            
            if len(common_metabolites) == 0:
                print(f"  ❌ No common metabolites with ground truth")
                continue
                
            method_estimates = pd.Series(beta_estimates, index=metabolites)[common_metabolites]
            ground_truth_filtered = ground_truth_log2fc[common_metabolites]
            
            # Calculate metrics
            correlation, _ = pearsonr(ground_truth_filtered, method_estimates)
            rmse = np.sqrt(mean_squared_error(ground_truth_filtered, method_estimates))
            
            print(f"  Correlation with ground truth: {correlation:.4f}")
            print(f"  RMSE from ground truth: {rmse:.6f}")
            print(f"  Metabolites used: {len(common_metabolites)}")
            
            # Check some statistics
            print(f"  Mean estimate: {np.mean(method_estimates):.4f}")
            print(f"  Std estimate: {np.std(method_estimates):.4f}")
            print(f"  Mean ground truth: {np.mean(ground_truth_filtered):.4f}")
            print(f"  Std ground truth: {np.std(ground_truth_filtered):.4f}")
            
            results.append({
                "method": method_name,
                "correlation": correlation,
                "rmse": rmse,
                "n_metabolites": len(common_metabolites),
                "mean_estimate": np.mean(method_estimates),
                "std_estimate": np.std(method_estimates)
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
    
    print("\n" + "="*60)
    print("EMPIRICAL BAYES METHODS COMPARISON")
    print("="*60)
    
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Compare to uninformative baseline
        if len(results) > 1:
            baseline = results_df[results_df["method"] == "Uninformative Bayesian"]
            if len(baseline) > 0:
                baseline_corr = baseline["correlation"].iloc[0]
                baseline_rmse = baseline["rmse"].iloc[0]
                
                print(f"\nComparisons to Uninformative Bayesian:")
                for _, row in results_df.iterrows():
                    if row["method"] != "Uninformative Bayesian":
                        corr_diff = row["correlation"] - baseline_corr
                        rmse_diff = row["rmse"] - baseline_rmse
                        print(f"  {row['method']}:")
                        print(f"    Correlation improvement: {corr_diff:+.4f}")
                        print(f"    RMSE change: {rmse_diff:+.6f} ({'better' if rmse_diff < 0 else 'worse'})")
    else:
        print("No successful results to compare")
    
    return results

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set, but not needed for this test")
    
    results = test_empirical_bayes_methods()