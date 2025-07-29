"""
Inspect LLM-Generated Metabolite Groupings.

This script loads the cached qualitative predictions from an LLM and
counts the number of metabolites assigned to each directional group
("increase", "decrease", "unchanged").

This is used to diagnose potential overshrinking in the hierarchical model,
which can be caused by highly imbalanced group sizes.
"""

import os
import pickle
import pandas as pd
import sys


# Use the same function from the benchmark script to ensure consistency
from benchmark_prior_recovery import (
    load_mtbls1_data,
    load_or_generate_qualitative_predictions,
)


def inspect_groups(qualitative_predictions: dict):
    """Counts and prints the number of metabolites in each group."""

    # --- Group metabolites by LLM prediction type ---
    llm_groups = {"increase": [], "decrease": [], "unchanged": []}

    for metabolite, pred_data in qualitative_predictions.items():
        direction = pred_data.get(
            "prediction", "unchanged"
        )  # Note: key is 'prediction' not 'direction'
        if "increase" in direction.lower():
            llm_groups["increase"].append(metabolite)
        elif "decrease" in direction.lower():
            llm_groups["decrease"].append(metabolite)
        else:
            llm_groups["unchanged"].append(metabolite)

    print("\n" + "=" * 40)
    print("LLM-Generated Group Sizes")
    print("=" * 40)

    total_metabolites = len(qualitative_predictions)

    for group_name, members in llm_groups.items():
        count = len(members)
        percentage = (count / total_metabolites) * 100 if total_metabolites > 0 else 0
        print(f"- {group_name.title():<10}: {count:>3} metabolites ({percentage:.1f}%)")

    print("=" * 40)

    if any(len(m) < 5 for m in llm_groups.values()) and total_metabolites > 0:
        print("\n⚠️  WARNING: One or more groups have very few members (< 5).")
        print("This can lead to unstable estimates and potential overshrinking")
        print("in the hierarchical model for those groups.")

    return llm_groups


def main():
    """Main execution function."""
    print("Loading MTBLS1 dataset to get metabolite list...")
    # We need the full list of metabolites to load the correct cache file
    _, _, metabolite_names, sample_data = load_mtbls1_data()

    print("\nLoading cached LLM qualitative predictions...")
    # Use the same parameters as the test script to load the correct cache
    qualitative_preds = load_or_generate_qualitative_predictions(
        metabolite_names,
        sample_data,
        use_hmdb_context=False,
        model_name="gemini-2.0-flash",
        temperature=0.0,
    )

    inspect_groups(qualitative_preds)


if __name__ == "__main__":
    main()
