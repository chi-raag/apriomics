"""
Interactive test script for HMDB utility functions.

This script allows you to manually test the functions in apriomics.hmdb_utils.
You can modify the example calls or add new ones to test different scenarios.
"""

from apriomics.hmdb_utils import (
    get_hmdb_metabolite_data,
    get_metabolite_context,
    batch_get_metabolite_contexts,
    EXAMPLE_METABOLITE_MAPPINGS,
    HMDBMetabolite  # Import for type hinting if needed, or direct use
)

def main():
    """Main function to run interactive tests."""
    print("Starting interactive tests for HMDB utilities...")

    # --- Test 1: Get data for a single metabolite by HMDB ID ---
    print("\n--- Test 1: get_hmdb_metabolite_data ---")
    glucose_id = "HMDB0000122"
    print(f"Fetching data for Glucose (ID: {glucose_id})...")
    glucose_data = get_hmdb_metabolite_data(glucose_id)
    if glucose_data:
        print(f"Name: {glucose_data.name}")
        print(f"Formula: {glucose_data.chemical_formula}")
        print(f"Description (first 100 chars): {glucose_data.description[:100] if glucose_data.description else 'N/A'}...")
        print(f"Pathways: {glucose_data.pathways[:3] if glucose_data.pathways else 'N/A'}") # Show first 3
        print(f"SMILES: {glucose_data.smiles}")
    else:
        print(f"Could not retrieve data for {glucose_id}")

    # --- Test 2: Get data for a metabolite that might not exist or cause an error ---
    print("\n--- Test 2: get_hmdb_metabolite_data (error case) ---")
    invalid_id = "HMDB0000000" # Likely non-existent or problematic
    print(f"Fetching data for a potentially invalid ID: {invalid_id}...")
    invalid_data = get_hmdb_metabolite_data(invalid_id)
    if invalid_data:
        print(f"Name: {invalid_data.name}")
    else:
        print(f"Could not retrieve data for {invalid_id} (as expected for a test).")

    # --- Test 2.1: see diseases output ---
    print("\n--- Test 2.1: get_hmdb_metabolite_data (diseases output) ---")
    # L-Lactic acid is known to be associated with several diseases
    lactic_acid_id = "HMDB0000190" 
    print(f"Fetching data for L-Lactic acid (ID: {lactic_acid_id}) to check diseases...")
    lactic_acid_data = get_hmdb_metabolite_data(lactic_acid_id)
    if lactic_acid_data:
        print(f"Name: {lactic_acid_data.name}")
        print(f"Associated Diseases: {lactic_acid_data.diseases if lactic_acid_data.diseases else 'N/A'}")
        if lactic_acid_data.diseases:
            print(f"Number of diseases found: {len(lactic_acid_data.diseases)}")
            print(f"First few diseases: {lactic_acid_data.diseases[:5]}") # Show up to 5
        else:
            print("No diseases listed for this metabolite in HMDB.")
    else:
        print(f"Could not retrieve data for {lactic_acid_id}")

    # --- Test 3: Get formatted context for a single metabolite ---
    print("\n--- Test 3: get_metabolite_context ---")
    pyruvate_name = "pyruvate"
    pyruvate_id = EXAMPLE_METABOLITE_MAPPINGS.get(pyruvate_name)
    print(f"Fetching context for {pyruvate_name} (ID: {pyruvate_id})...")
    pyruvate_context = get_metabolite_context(pyruvate_name, hmdb_id=pyruvate_id)
    print(f"Context for {pyruvate_name}:")
    print(pyruvate_context)

    print(f"\nFetching context for Alanine (ID: {EXAMPLE_METABOLITE_MAPPINGS.get('alanine')})...")
    alanine_context = get_metabolite_context("alanine", hmdb_id=EXAMPLE_METABOLITE_MAPPINGS.get('alanine'))
    print(f"Context for Alanine:")
    print(alanine_context)
    
    print(f"\nFetching context for a metabolite not in EXAMPLE_METABOLITE_MAPPINGS (will attempt search if enabled):")
    # Note: search_hmdb_metabolite currently returns None, so this will show "data not available"
    unknown_metabolite_context = get_metabolite_context("MetaboliteX")
    print(f"Context for MetaboliteX:")
    print(unknown_metabolite_context)


    # --- Test 4: Batch get contexts for multiple metabolites ---
    print("\n--- Test 4: batch_get_metabolite_contexts ---")
    metabolite_names_to_test = ["glucose", "lactate", "serine", "NonExistentMetabolite"]
    # Using a subset of EXAMPLE_METABOLITE_MAPPINGS for this batch test
    test_mappings = {
        "glucose": EXAMPLE_METABOLITE_MAPPINGS["glucose"],
        "lactate": EXAMPLE_METABOLITE_MAPPINGS["lactate"],
        "serine": EXAMPLE_METABOLITE_MAPPINGS["serine"],
        # "NonExistentMetabolite" will rely on search_hmdb_metabolite
    }
    print(f"Fetching batch contexts for: {', '.join(metabolite_names_to_test)}")
    batch_contexts = batch_get_metabolite_contexts(metabolite_names_to_test, hmdb_mapping=test_mappings)
    
    for name, context in batch_contexts.items():
        print(f"\nContext for {name}:")
        print(context)
        
    # --- Test 5: Using a metabolite name not in the provided mapping for batch ---
    print("\n--- Test 5: batch_get_metabolite_contexts (name not in mapping) ---")
    # 'ATP' is in EXAMPLE_METABOLITE_MAPPINGS but we won't provide it in the call's mapping
    # to test if get_metabolite_context -> search_hmdb_metabolite path is triggered (it will be, and search will return None).
    atp_context_batch = batch_get_metabolite_contexts(["ATP"]) 
    print(f"Context for ATP (via batch, no explicit mapping):")
    print(atp_context_batch.get("ATP"))


    print("\n--- Interactive tests completed ---")
    print("You can add more test cases below or modify existing ones.")
    
    # Example of how to test with a new metabolite not in defaults
    # print("\n--- Custom Test ---")
    # custom_id = "HMDB0000056" # Example: Citric acid
    # custom_data = get_hmdb_metabolite_data(custom_id)
    # if custom_data:
    #     print(f"Custom metabolite ({custom_id}) Name: {custom_data.name}")
    #     print(get_metabolite_context(custom_data.name, custom_id))
    # else:
    #     print(f"Could not fetch data for custom ID {custom_id}")

if __name__ == "__main__":
    main() 