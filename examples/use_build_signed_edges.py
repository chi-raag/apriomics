"""
Example script demonstrating the usage of the build_signed_edges module.

This script takes a list of metabolite names and an output CSV file path
as command-line arguments, processes the reactions, and writes the resulting
signed edges to the specified file.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing the module
# Adjust this if your project structure is different
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from apriomics.build_signed_edges import process_reactions, write_signed_edges_csv
except ImportError as e:
    print(f"Error: Could not import the required module. Ensure the package is installed or the path is correct. {e}")
    sys.exit(1)

# Attempt to import necessary functions
try:
    from apriomics.kegg_utils import get_kegg_id_from_name, get_reactions_for_compound
except ImportError as e:
    # Provide a more helpful message if modules aren't found
    print(f"Error: Could not import required modules. {e}")
    print("Please ensure the 'apriomics' package is installed (e.g., 'pip install -e .')")
    print("or that the script is run from the project root directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Build signed edges for reactions involving specified metabolites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "metabolite_names",
        metavar="MetaboliteName",
        nargs='+',
        help="One or more metabolite names (e.g., 'Glucose' 'Pyruvate')."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output CSV file for signed edges."
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Require exact name matching when looking up metabolite KEGG IDs."
    )

    args = parser.parse_args()

    output_file = Path(args.output)
    if not output_file.parent.exists():
        print(f"Error: Output directory {output_file.parent} does not exist.")
        sys.exit(1)

    print(f"Finding KEGG Compound IDs for: {args.metabolite_names}")
    compound_ids = set()
    name_to_id_map = {}
    for name in args.metabolite_names:
        kegg_id = get_kegg_id_from_name(name, exact_match=args.exact_match)
        if kegg_id:
            print(f"  Found '{name}' -> {kegg_id}")
            compound_ids.add(kegg_id)
            name_to_id_map[name] = kegg_id
        else:
            print(f"  Warning: Could not find KEGG ID for '{name}'. Skipping.")

    if not compound_ids:
        print("Error: No KEGG Compound IDs found for the provided metabolite names.")
        sys.exit(1)

    print(f"\nFinding KEGG Reaction IDs linked to compounds: {list(compound_ids)}")
    all_reaction_ids = set()
    for c_id in compound_ids:
        reaction_ids = get_reactions_for_compound(c_id)
        if reaction_ids:
            print(f"  Found {len(reaction_ids)} reactions for {c_id}")
            all_reaction_ids.update(reaction_ids)
        # else: No need to print if none found, get_reactions_for_compound handles info messages

    if not all_reaction_ids:
        print("Error: No KEGG Reaction IDs found linked to the specified metabolites.")
        sys.exit(1)

    print(f"\nProcessing {len(all_reaction_ids)} unique reactions: {list(all_reaction_ids)}")
    edges = process_reactions(list(all_reaction_ids))

    if not edges:
        print("No edges were generated. This might be due to errors fetching/parsing reactions.")
        sys.exit(0) # Exit gracefully if no edges, maybe reactions were invalid

    print(f"\nWriting {len(edges)} edges to {output_file}...")
    write_signed_edges_csv(edges, output_file)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main() 