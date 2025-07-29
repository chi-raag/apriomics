"""
Create classification accuracy plot with individual points instead of boxplot.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '.')

def create_accuracy_points_plot():
    """Create accuracy plot with individual data points."""
    
    # Load the CV results from the previous run
    results_path = "/Users/chiraag/Projects/gwu/lab/apriomics/output/predictive_performance_cv_results.csv"
    
    if not os.path.exists(results_path):
        print("❌ Results file not found. Please run the CV benchmark first.")
        return
    
    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} results from {results_path}")
    
    # Filter out any NaN results
    results_df = results_df.dropna()
    
    # Create figure
    plt.style.use("default")
    sns.set_palette("colorblind")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create strip plot with individual points
    sns.stripplot(data=results_df, x="method", y="accuracy", 
                  size=8, alpha=0.8, jitter=True, ax=ax)
    
    # Add mean lines
    method_means = results_df.groupby("method")["accuracy"].mean()
    for i, (method, mean_acc) in enumerate(method_means.items()):
        ax.hlines(mean_acc, i-0.3, i+0.3, colors='red', linestyles='solid', 
                 linewidth=2, alpha=0.8)
        # Add mean value text
        ax.text(i, mean_acc + 0.01, f'{mean_acc:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Customize plot
    ax.set_title("Classification Accuracy by Method\n(Individual Cross-Validation Results)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Method", fontsize=14, fontweight='bold')
    ax.set_ylabel("Classification Accuracy", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)  # Focus on the relevant range
    
    # Add horizontal line for random baseline
    dummy_baseline = 0.636  # From validation
    ax.axhline(y=dummy_baseline, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(0.02, dummy_baseline + 0.005, f'Dummy baseline: {dummy_baseline:.3f}', 
           transform=ax.get_yaxis_transform(), fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    plt.savefig(f"{output_dir}/classification_accuracy_points.png", 
                dpi=300, bbox_inches="tight")
    print(f"📊 Plot saved to {output_dir}/classification_accuracy_points.png")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("CLASSIFICATION ACCURACY SUMMARY")
    print("="*70)
    
    summary_stats = results_df.groupby("method")["accuracy"].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    print("\nMethod Statistics:")
    print(f"{'Method':<25} {'Count':<6} {'Mean':<6} {'Std':<6} {'Min':<6} {'Max':<6}")
    print("-" * 70)
    
    for method in summary_stats.index:
        count = int(summary_stats.loc[method, 'count'])
        mean_acc = summary_stats.loc[method, 'mean']
        std_acc = summary_stats.loc[method, 'std']
        min_acc = summary_stats.loc[method, 'min']
        max_acc = summary_stats.loc[method, 'max']
        
        print(f"{method:<25} {count:<6} {mean_acc:<6.3f} {std_acc:<6.3f} {min_acc:<6.3f} {max_acc:<6.3f}")
    
    # Rank methods by mean accuracy
    ranked_methods = summary_stats.sort_values('mean', ascending=False)
    
    print(f"\n🏆 Ranking by Mean Accuracy:")
    for rank, (method, stats) in enumerate(ranked_methods.iterrows(), 1):
        mean_acc = stats['mean']
        std_acc = stats['std']
        print(f"  {rank}. {method}: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Compare LLM methods
    llm_methods = [m for m in summary_stats.index if 'LLM' in m]
    if llm_methods:
        print(f"\n🤖 LLM Methods Comparison:")
        for method in llm_methods:
            mean_acc = summary_stats.loc[method, 'mean']
            std_acc = summary_stats.loc[method, 'std']
            print(f"  {method}: {mean_acc:.3f} ± {std_acc:.3f}")
        
        best_llm = max(llm_methods, key=lambda m: summary_stats.loc[m, 'mean'])
        best_llm_acc = summary_stats.loc[best_llm, 'mean']
        print(f"  Best LLM method: {best_llm} ({best_llm_acc:.3f})")
    
    return results_df, summary_stats

if __name__ == "__main__":
    results_df, summary_stats = create_accuracy_points_plot()