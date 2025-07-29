"""
Check and analyze the o4-deep-research predictions.
"""

import pickle
import pandas as pd
import numpy as np

def check_deep_research_predictions():
    """Load and analyze the deep research predictions."""
    
    pkl_file = "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_o4_mini_deep_research_2025_06_26_temp01_53metabolites.pkl"
    
    print("🔍 ANALYZING O4-DEEP-RESEARCH PREDICTIONS")
    print("="*60)
    
    try:
        with open(pkl_file, 'rb') as f:
            predictions = pickle.load(f)
        
        print(f"Total metabolites: {len(predictions)}")
        
        # Analyze prediction patterns
        directions = []
        magnitudes = []
        confidences = []
        reasoning_lengths = []
        
        for metabolite, pred in predictions.items():
            directions.append(pred.get('prediction', 'unknown'))
            magnitudes.append(pred.get('magnitude', 'unknown'))
            confidences.append(pred.get('confidence', 0.0))
            reasoning_lengths.append(len(pred.get('reasoning', '')))
        
        # Direction distribution
        direction_counts = pd.Series(directions).value_counts()
        print(f"\n📊 Direction distribution:")
        for direction, count in direction_counts.items():
            pct = count / len(directions) * 100
            print(f"   {direction}: {count} ({pct:.1f}%)")
        
        # Magnitude distribution  
        magnitude_counts = pd.Series(magnitudes).value_counts()
        print(f"\n📏 Magnitude distribution:")
        for magnitude, count in magnitude_counts.items():
            pct = count / len(magnitudes) * 100
            print(f"   {magnitude}: {count} ({pct:.1f}%)")
        
        # Confidence statistics
        conf_array = np.array(confidences)
        print(f"\n🎯 Confidence statistics:")
        print(f"   Mean: {np.mean(conf_array):.3f}")
        print(f"   Std: {np.std(conf_array):.3f}")
        print(f"   Min: {np.min(conf_array):.3f}")
        print(f"   Max: {np.max(conf_array):.3f}")
        print(f"   Median: {np.median(conf_array):.3f}")
        
        # Reasoning quality (length as proxy)
        reasoning_array = np.array(reasoning_lengths)
        print(f"\n📝 Reasoning quality (character count):")
        print(f"   Mean: {np.mean(reasoning_array):.0f} chars")
        print(f"   Min: {np.min(reasoning_array)} chars")
        print(f"   Max: {np.max(reasoning_array)} chars")
        
        # Show some detailed examples
        print(f"\n🔬 DETAILED EXAMPLES:")
        print("="*60)
        
        # Show a few diverse examples
        example_metabolites = list(predictions.keys())[:5]
        
        for i, metabolite in enumerate(example_metabolites, 1):
            pred = predictions[metabolite]
            print(f"\n{i}. {metabolite}:")
            print(f"   Direction: {pred.get('prediction', 'N/A')}")
            print(f"   Magnitude: {pred.get('magnitude', 'N/A')}")
            print(f"   Confidence: {pred.get('confidence', 'N/A')}")
            reasoning = pred.get('reasoning', 'N/A')
            # Truncate long reasoning for display
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            print(f"   Reasoning: {reasoning}")
        
        # Quality assessment
        print(f"\n📋 QUALITY ASSESSMENT:")
        print("="*60)
        
        # Check for diversity
        direction_diversity = len(direction_counts) / 3 * 100  # 3 possible directions
        magnitude_diversity = len(magnitude_counts) / 3 * 100  # 3 possible magnitudes
        conf_range = np.max(conf_array) - np.min(conf_array)
        
        print(f"Direction diversity: {direction_diversity:.1f}% (66.7% = good)")
        print(f"Magnitude diversity: {magnitude_diversity:.1f}% (100% = excellent)")
        print(f"Confidence range: {conf_range:.3f} (>0.5 = good)")
        print(f"Average reasoning length: {np.mean(reasoning_array):.0f} chars (>100 = detailed)")
        
        # Overall assessment
        issues = []
        strengths = []
        
        if direction_counts.get('increase', 0) / len(directions) > 0.8:
            issues.append("Heavy directional bias toward 'increase'")
        else:
            strengths.append("Good directional balance")
            
        if magnitude_counts.get('large', 0) == 0:
            issues.append("No 'large' magnitude predictions")
        else:
            strengths.append("Uses 'large' magnitude predictions")
            
        if conf_range < 0.3:
            issues.append("Narrow confidence range")
        else:
            strengths.append("Good confidence range")
            
        if np.mean(reasoning_array) < 50:
            issues.append("Very short reasoning explanations")
        else:
            strengths.append("Detailed reasoning provided")
        
        print(f"\n✅ STRENGTHS:")
        for strength in strengths:
            print(f"   • {strength}")
            
        if issues:
            print(f"\n⚠️ POTENTIAL ISSUES:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n🎉 No major issues identified!")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return None

if __name__ == "__main__":
    predictions = check_deep_research_predictions()