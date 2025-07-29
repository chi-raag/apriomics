"""
Quick test to see if the mapping is causing LLM underperformance.
Compare current mapping vs oracle-like mapping using same LLM predictions.
"""

import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
sys.path.insert(0, '.')

from apriomics.priors.base import map_qualitative_to_numerical_priors

def load_ground_truth():
    """Load ground truth log2FC from previous benchmark run."""
    # Use empirical estimates as ground truth proxy
    data_path = "/Users/chiraag/Projects/gwu/lab/apriomics/output/mtbls1_hmdb_mapped_data.pkl"
    
    with open(data_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    filtered_data = cached_data["filtered_data"]
    
    # Get control vs case groups (simplified)
    control_samples = filtered_data.iloc[:84]  # First 84 are controls
    case_samples = filtered_data.iloc[84:]     # Rest are cases
    
    # Calculate empirical log2FC
    control_means = control_samples.mean()
    case_means = case_samples.mean()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    control_means = control_means + epsilon
    case_means = case_means + epsilon
    
    empirical_log2fc = np.log2(case_means / control_means)
    
    return empirical_log2fc

def create_oracle_mapping(llm_predictions, ground_truth):
    """Create oracle-like mapping that uses ground truth to set perfect priors."""
    oracle_priors = {}
    
    for metabolite, pred in llm_predictions.items():
        if metabolite in ground_truth.index:
            true_logfc = ground_truth[metabolite]
            
            # Use LLM direction but ground truth magnitude
            direction = pred.get('prediction', 'unchanged')
            confidence = pred.get('confidence', 0.5)
            
            if direction == 'increase' and true_logfc > 0:
                # LLM got direction right - use true magnitude with small noise
                oracle_mean = true_logfc * 0.8  # 80% of true effect
                oracle_sd = 0.2  # Small uncertainty
            elif direction == 'decrease' and true_logfc < 0:
                # LLM got direction right
                oracle_mean = true_logfc * 0.8
                oracle_sd = 0.2
            elif direction == 'unchanged' and abs(true_logfc) < 0.1:
                # LLM correctly identified unchanged
                oracle_mean = 0.0
                oracle_sd = 0.2
            else:
                # LLM got direction wrong - use weaker prior
                oracle_mean = true_logfc * 0.3  # Weak pull toward truth
                oracle_sd = 0.8  # High uncertainty
            
            oracle_priors[metabolite] = {
                'expected_log2fc': oracle_mean,
                'prior_sd': oracle_sd
            }
    
    return oracle_priors

def test_mapping_impact():
    """Test if mapping is the bottleneck in LLM performance."""
    
    print("🧪 TESTING MAPPING IMPACT ON LLM PERFORMANCE")
    print("="*60)
    
    # Load latest LLM predictions (use any available)
    pred_files = [
        "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gpt_4o_mini_2024_07_18_temp01_53metabolites.pkl",
        "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gemini_2_0_flash_temp01_53metabolites.pkl",
        "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gpt_4_1_2025_04_14_temp01_53metabolites.pkl"
    ]
    
    llm_predictions = None
    used_file = None
    
    for pred_file in pred_files:
        try:
            with open(pred_file, 'rb') as f:
                llm_predictions = pickle.load(f)
            used_file = pred_file
            break
        except FileNotFoundError:
            continue
    
    if llm_predictions is None:
        print("❌ No LLM prediction files found!")
        return
    
    print(f"📁 Using predictions from: {used_file.split('/')[-1]}")
    print(f"🔢 Number of predictions: {len(llm_predictions)}")
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Get common metabolites
    common_metabolites = set(llm_predictions.keys()) & set(ground_truth.index)
    print(f"🎯 Common metabolites: {len(common_metabolites)}")
    
    if len(common_metabolites) < 10:
        print("❌ Too few common metabolites for meaningful test!")
        return
    
    # Test different mappings
    mappings_to_test = {
        "current_conservative": map_qualitative_to_numerical_priors(llm_predictions, "conservative"),
        "current_moderate": map_qualitative_to_numerical_priors(llm_predictions, "moderate"),
        "oracle_mapping": create_oracle_mapping(llm_predictions, ground_truth)
    }
    
    print(f"\n📊 MAPPING COMPARISON RESULTS:")
    print("="*60)
    
    results = []
    
    for mapping_name, priors in mappings_to_test.items():
        # Get predictions for common metabolites
        prior_means = []
        true_values = []
        
        for metabolite in common_metabolites:
            if metabolite in priors:
                prior_means.append(priors[metabolite]['expected_log2fc'])
                true_values.append(ground_truth[metabolite])
        
        if len(prior_means) > 0:
            correlation, p_value = pearsonr(prior_means, true_values)
            rmse = np.sqrt(np.mean((np.array(prior_means) - np.array(true_values))**2))
            
            results.append({
                'mapping': mapping_name,
                'correlation': correlation,
                'p_value': p_value,
                'rmse': rmse,
                'n_metabolites': len(prior_means)
            })
            
            print(f"{mapping_name:20s}: r={correlation:.3f} (p={p_value:.3f}), RMSE={rmse:.3f}, n={len(prior_means)}")
    
    # Analysis
    print(f"\n🔍 INTERPRETATION:")
    print("="*60)
    
    if len(results) >= 2:
        current_best = max([r for r in results if 'current' in r['mapping']], key=lambda x: x['correlation'])
        oracle_result = next((r for r in results if 'oracle' in r['mapping']), None)
        
        if oracle_result:
            improvement = oracle_result['correlation'] - current_best['correlation']
            print(f"📈 Oracle mapping improvement: +{improvement:.3f} correlation")
            print(f"📈 Oracle RMSE improvement: {current_best['rmse'] - oracle_result['rmse']:.3f}")
            
            if improvement > 0.1:
                print("✅ MAPPING IS LIKELY THE ISSUE - Oracle mapping shows substantial improvement!")
                print("   Recommendation: Redesign the magnitude/confidence → prior mapping")
            elif improvement > 0.05:
                print("⚠️  MAPPING CONTRIBUTES TO ISSUE - Moderate improvement possible")
                print("   Recommendation: Refine mapping, but also check LLM prediction quality")
            else:
                print("❌ MAPPING NOT THE MAIN ISSUE - Oracle mapping doesn't help much")
                print("   Recommendation: Focus on LLM prediction quality, not mapping")
        
        # Show example mappings
        print(f"\n📋 EXAMPLE MAPPINGS (first 5 metabolites):")
        print("="*60)
        
        example_metabolites = list(common_metabolites)[:5]
        for metabolite in example_metabolites:
            pred = llm_predictions[metabolite]
            true_val = ground_truth[metabolite]
            
            print(f"\n{metabolite}:")
            print(f"  LLM: {pred.get('prediction', 'N/A')} / {pred.get('magnitude', 'N/A')} / conf={pred.get('confidence', 'N/A')}")
            print(f"  True log2FC: {true_val:.3f}")
            
            for mapping_name, priors in mappings_to_test.items():
                if metabolite in priors:
                    prior = priors[metabolite]
                    print(f"  {mapping_name}: μ={prior['expected_log2fc']:.3f}, σ={prior['prior_sd']:.3f}")

if __name__ == "__main__":
    test_mapping_impact()