"""
Simple test to see if mapping is causing LLM underperformance.
Compare current mapping vs stronger magnitude mapping using same LLM predictions.
"""

import pickle
import numpy as np
import sys

sys.path.insert(0, ".")

from apriomics.priors.base import map_qualitative_to_numerical_priors


def create_stronger_mapping(llm_predictions):
    """Create mapping with stronger magnitude effects."""
    stronger_priors = {}

    for metabolite, pred in llm_predictions.items():
        prediction = pred.get("prediction", "unchanged").lower()
        magnitude = pred.get("magnitude", "small").lower()
        confidence = pred.get("confidence", 0.5)

        # STRONGER magnitude effects (2x current values)
        magnitude_effects = {
            "small": 0.16,  # was 0.08
            "moderate": 0.30,  # was 0.15
            "large": 0.50,  # was 0.25
        }

        base_effect = magnitude_effects.get(magnitude, 0.20)

        if prediction == "increase":
            prior_mean = base_effect
        elif prediction == "decrease":
            prior_mean = -base_effect
        else:
            prior_mean = 0.0

        # TIGHTER confidence mapping
        if confidence > 0.8:
            prior_sd = 0.3  # was 0.5
        elif confidence > 0.6:
            prior_sd = 0.5  # was 0.7
        else:
            prior_sd = 0.7  # was 0.9

        stronger_priors[metabolite] = {
            "expected_log2fc": prior_mean,
            "prior_sd": prior_sd,
        }

    return stronger_priors


def test_mapping_variations():
    """Test different mapping variations to see impact."""

    print("🧪 TESTING MAPPING VARIATIONS")
    print("=" * 50)

    # Load LLM predictions
    pred_file = "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gpt_4o_mini_2024_07_18_temp01_53metabolites.pkl"

    try:
        with open(pred_file, "rb") as f:
            llm_predictions = pickle.load(f)
    except FileNotFoundError:
        print("❌ Prediction file not found!")
        return

    print(f"📁 Using: {pred_file.split('/')[-1]}")
    print(f"🔢 Predictions: {len(llm_predictions)}")

    # Create different mappings
    mappings = {
        "current_conservative": map_qualitative_to_numerical_priors(
            llm_predictions, "conservative"
        ),
        "current_moderate": map_qualitative_to_numerical_priors(
            llm_predictions, "moderate"
        ),
        "stronger_magnitudes": create_stronger_mapping(llm_predictions),
    }

    print("\n📊 MAPPING ANALYSIS:")
    print("=" * 50)

    # Analyze each mapping
    for mapping_name, priors in mappings.items():
        means = [p["expected_log2fc"] for p in priors.values()]
        sds = [p["prior_sd"] for p in priors.values()]

        print(f"\n{mapping_name}:")
        print(
            f"  Mean effects: {np.mean(np.abs(means)):.3f} ± {np.std(np.abs(means)):.3f}"
        )
        print(f"  Mean SD: {np.mean(sds):.3f} ± {np.std(sds):.3f}")
        print(f"  Effect range: [{np.min(means):.3f}, {np.max(means):.3f}]")

        # Show direction distribution
        increases = sum(1 for m in means if m > 0.01)
        decreases = sum(1 for m in means if m < -0.01)
        unchanged = sum(1 for m in means if abs(m) <= 0.01)

        print(f"  Directions: {increases} inc, {decreases} dec, {unchanged} unch")

    # Show example differences
    print("\n📋 EXAMPLE DIFFERENCES (first 5 metabolites):")
    print("=" * 50)

    example_metabolites = list(llm_predictions.keys())[:5]
    for metabolite in example_metabolites:
        pred = llm_predictions[metabolite]
        print(f"\n{metabolite}:")
        print(
            f"  LLM: {pred.get('prediction', 'N/A')} / {pred.get('magnitude', 'N/A')} / conf={pred.get('confidence', 'N/A'):.2f}"
        )

        for mapping_name, priors in mappings.items():
            if metabolite in priors:
                prior = priors[metabolite]
                print(
                    f"  {mapping_name:20s}: μ={prior['expected_log2fc']:+.3f}, σ={prior['prior_sd']:.2f}"
                )

    # Key insight
    current_effects = [
        abs(p["expected_log2fc"]) for p in mappings["current_moderate"].values()
    ]
    stronger_effects = [
        abs(p["expected_log2fc"]) for p in mappings["stronger_magnitudes"].values()
    ]

    print("\n🔍 KEY INSIGHT:")
    print("=" * 50)
    print(f"Current avg magnitude: {np.mean(current_effects):.3f}")
    print(f"Stronger avg magnitude: {np.mean(stronger_effects):.3f}")
    print(
        f"Magnitude increase: {np.mean(stronger_effects) / np.mean(current_effects):.1f}x"
    )

    if np.mean(stronger_effects) / np.mean(current_effects) > 1.5:
        print("💡 HYPOTHESIS: Current magnitudes may be too weak!")
        print("   Try stronger magnitude mapping in benchmark.")
    else:
        print("💭 Current vs stronger magnitudes similar.")
        print("   Issue may be elsewhere (LLM quality, distribution choice, etc.)")


if __name__ == "__main__":
    test_mapping_variations()
