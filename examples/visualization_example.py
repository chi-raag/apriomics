#!/usr/bin/env python3
"""
Example: Visualizing Markov Field and LLM Priors
===============================================

This example demonstrates how to use the apriomics visualization suite to 
analyze and visualize metabolomics priors from:
1. Markov field priors derived from metabolic reaction networks
2. LLM-based priors from HMDB XML data

Usage:
    uv run python examples/visualization_example.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apriomics.build_signed_edges import process_reactions
from apriomics.visualization import MarkovFieldVisualizer, LLMPriorVisualizer, analyze_priors
from dataclasses import dataclass
from typing import List

@dataclass
class LLMMetaboliteScore:
    """Simple structure for LLM metabolite scoring (for demonstration)."""
    metabolite: str
    score: float
    direction: str
    rationale: str
    expected_log2fc: float = 0.0
    prior_sd: float = 0.5
    magnitude: str = "moderate"
    confidence: str = "moderate"


def create_example_signed_edges():
    """Create example signed edges from KEGG reactions."""
    print("Creating example signed metabolic network...")
    
    # Example KEGG reaction IDs - these are real reactions
    example_reactions = [
        "R00200",  # Glycolysis: glucose-6-phosphate <=> fructose-6-phosphate
        "R00658",  # Citrate cycle: citrate <=> isocitrate
        "R01518",  # Amino acid metabolism: alanine <=> pyruvate
        "R00214",  # Glycolysis: fructose-1,6-bisphosphate <=> DHAP + G3P
        "R00756",  # Citrate cycle: isocitrate <=> 2-oxoglutarate
    ]
    
    try:
        # Process reactions to get signed edges
        signed_edges = process_reactions(example_reactions)
        print(f"Generated {len(signed_edges)} signed edges from {len(example_reactions)} reactions")
        return signed_edges
    except Exception as e:
        print(f"Error processing KEGG reactions: {e}")
        print("Using mock signed edges instead...")
        
        # Create mock signed edges for demonstration
        metabolites = [
            "C00031",  # D-Glucose
            "C00092",  # D-Glucose 6-phosphate
            "C00085",  # D-Fructose 6-phosphate
            "C00354",  # D-Fructose 1,6-bisphosphate
            "C00111",  # Glycerone phosphate
            "C00118",  # D-Glyceraldehyde 3-phosphate
            "C00022",  # Pyruvate
            "C00036",  # Oxaloacetate
            "C00158",  # Citrate
            "C00311",  # Isocitrate
        ]
        
        signed_edges = {}
        
        # Add some substrate-product relationships (-1)
        substrate_product_pairs = [
            ("C00031", "C00092"),  # Glucose -> G6P
            ("C00092", "C00085"),  # G6P -> F6P
            ("C00085", "C00354"),  # F6P -> F1,6BP
            ("C00354", "C00111"),  # F1,6BP -> DHAP
            ("C00354", "C00118"),  # F1,6BP -> G3P
            ("C00118", "C00022"),  # G3P -> Pyruvate
            ("C00158", "C00311"),  # Citrate -> Isocitrate
        ]
        
        for sub, prod in substrate_product_pairs:
            signed_edges[(sub, prod)] = -1
            signed_edges[(prod, sub)] = -1  # Reversible
        
        # Add some same-side relationships (+1)
        same_side_pairs = [
            ("C00111", "C00118"),  # DHAP and G3P (both products of F1,6BP)
            ("C00092", "C00031"),  # G6P and Glucose (substrate-product but also cofactor)
        ]
        
        for met1, met2 in same_side_pairs:
            signed_edges[(met1, met2)] = 1
            signed_edges[(met2, met1)] = 1
        
        return signed_edges


def create_example_bayesian_scores():
    """Create example Bayesian metabolite prior scores."""
    print("Creating example Bayesian metabolite prior scores...")
    
    # Example metabolites with realistic prior scoring
    example_metabolites = [
        ("Glucose", "increase", "large", "high", "Primary substrate elevated in diabetes"),
        ("Lactate", "increase", "moderate", "high", "Elevated due to anaerobic metabolism"),
        ("Citrate", "decrease", "small", "moderate", "Reduced TCA cycle activity"),
        ("Alanine", "increase", "small", "moderate", "Increased protein catabolism"),
        ("Glutamine", "decrease", "moderate", "high", "Consumed for energy production"),
        ("Pyruvate", "unclear", "moderate", "low", "Variable depending on metabolic state"),
        ("Acetate", "minimal", "minimal", "high", "Generally stable across conditions"),
        ("Creatine", "increase", "small", "moderate", "Muscle-related changes"),
        ("Succinate", "increase", "moderate", "moderate", "Mitochondrial dysfunction marker"),
        ("Fumarate", "increase", "small", "low", "Associated with metabolic stress"),
    ]
    
    bayesian_scores = []
    
    for metabolite, direction, magnitude, confidence, rationale in example_metabolites:
        # Calculate expected log2fc and prior_sd based on direction and magnitude
        direction_mapping = {
            'increase': 1,
            'decrease': -1,
            'minimal': 0,
            'unclear': 0
        }
        
        magnitude_mapping = {
            'minimal': 0.3,
            'small': 0.6,
            'moderate': 1.0,
            'large': 1.6
        }
        
        confidence_mapping = {
            'high': 0.8,
            'moderate': 1.0,
            'low': 1.4
        }
        
        expected_log2fc = direction_mapping[direction] * magnitude_mapping[magnitude]
        prior_sd = 0.3 * magnitude_mapping[magnitude] * confidence_mapping[confidence]
        
        # Calculate importance score (0-1) based on abs(log2fc)
        importance_score = min(abs(expected_log2fc) / 2.0, 1.0)
        
        score = LLMMetaboliteScore(
            metabolite=metabolite,
            score=importance_score,
            direction=direction,
            rationale=rationale,
            expected_log2fc=expected_log2fc,
            prior_sd=prior_sd,
            magnitude=magnitude,
            confidence=confidence
        )
        
        bayesian_scores.append(score)
    
    return bayesian_scores


def main():
    """Main example function."""
    print("=" * 60)
    print("APRIOMICS VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Create example data
    signed_edges = create_example_signed_edges()
    bayesian_scores = create_example_bayesian_scores()
    
    print(f"\nData summary:")
    print(f"- Signed edges: {len(signed_edges)}")
    print(f"- Bayesian scores: {len(bayesian_scores)}")
    
    # Create visualizers
    print("\nCreating visualizers...")
    markov_viz = MarkovFieldVisualizer(signed_edges)
    bayesian_viz = LLMPriorVisualizer(bayesian_scores)
    
    # Display network statistics
    print("\nNetwork Statistics:")
    network_stats = markov_viz.get_network_statistics()
    for key, value in network_stats.items():
        print(f"  {key}: {value}")
    
    # Display Bayesian statistics
    print("\nBayesian Prior Statistics:")
    bayesian_stats = bayesian_viz.get_summary_statistics()
    for key, value in bayesian_stats.items():
        print(f"  {key}: {value}")
    
    # Generate individual plots
    print("\nGenerating individual visualizations...")
    
    # Markov field visualizations
    print("1. Plotting signed metabolic network...")
    try:
        fig1 = markov_viz.plot_network(figsize=(12, 8))
        print("   ✓ Network plot generated")
    except Exception as e:
        print(f"   ✗ Network plot failed: {e}")
    
    print("2. Plotting Laplacian heatmap...")
    try:
        fig2 = markov_viz.plot_laplacian_heatmap(figsize=(10, 8))
        print("   ✓ Laplacian heatmap generated")
    except Exception as e:
        print(f"   ✗ Laplacian heatmap failed: {e}")
    
    print("3. Plotting degree distribution...")
    try:
        fig3 = markov_viz.plot_degree_distribution(figsize=(12, 6))
        print("   ✓ Degree distribution generated")
    except Exception as e:
        print(f"   ✗ Degree distribution failed: {e}")
    
    # Bayesian prior visualizations
    print("4. Plotting Bayesian importance scores...")
    try:
        fig4 = bayesian_viz.plot_importance_scores(top_n=10, figsize=(12, 8))
        print("   ✓ Importance scores generated")
    except Exception as e:
        print(f"   ✗ Importance scores failed: {e}")
    
    print("5. Plotting prior distributions...")
    try:
        fig5 = bayesian_viz.plot_prior_distributions(figsize=(12, 8))
        print("   ✓ Prior distributions generated")
    except Exception as e:
        print(f"   ✗ Prior distributions failed: {e}")
    
    print("6. Plotting confidence analysis...")
    try:
        fig6 = bayesian_viz.plot_confidence_analysis(figsize=(12, 6))
        print("   ✓ Confidence analysis generated")
    except Exception as e:
        print(f"   ✗ Confidence analysis failed: {e}")
    
    print("7. Plotting direction analysis...")
    try:
        fig7 = bayesian_viz.plot_direction_analysis(figsize=(12, 8))
        print("   ✓ Direction analysis generated")
    except Exception as e:
        print(f"   ✗ Direction analysis failed: {e}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    try:
        analyze_priors(signed_edges, bayesian_scores, output_dir="output/visualization_report")
        print("   ✓ Comprehensive report generated in output/visualization_report/")
    except Exception as e:
        print(f"   ✗ Comprehensive report failed: {e}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the generated plots in your environment")
    print("2. Explore the comprehensive report in output/visualization_report/")
    print("3. Try the interactive explorer in a Jupyter notebook")
    print("4. Adapt the code for your own metabolomics data")


if __name__ == "__main__":
    main()