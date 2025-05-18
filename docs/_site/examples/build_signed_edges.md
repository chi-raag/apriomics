# Using `apriomics` to Build Signed Edges


## Overview

This document demonstrates how to use the functions within the
`apriomics` package to generate a signed edge graph based on KEGG
reactions involving specified metabolites. The goal is to obtain a data
structure representing relationships between metabolites within
biochemical reactions:

- `sign = -1`: For substrate → product pairs.
- `sign = +1`: For pairs on the same side of a reaction (co-substrates
  or co-products).

The core functions involved are:

1.  `apriomics.kegg_utils.get_kegg_id_from_name`: Finds KEGG Compound
    IDs from metabolite names.
2.  `apriomics.kegg_utils.get_reactions_for_compound`: Finds KEGG
    Reaction IDs associated with Compound IDs.
3.  `apriomics.build_signed_edges.process_reactions`: Fetches reaction
    details and builds the signed edge dictionary.
4.  `apriomics.build_signed_edges.write_signed_edges_csv`: Saves the
    graph to a CSV file.

## Step-by-Step Example

Let’s walk through the process using “Glucose” and “Pyruvate” as example
metabolites.

### 1. Imports

First, import the necessary functions:

``` python
import sys
from pathlib import Path

# If apriomics is installed (e.g., pip install -e .), these imports work directly.
# Otherwise, ensure the project root is in sys.path
# project_root = Path(".").resolve() # Assuming run from project root
# sys.path.insert(0, str(project_root))

try:
    from apriomics.build_signed_edges import process_reactions, write_signed_edges_csv
    from apriomics.kegg_utils import get_kegg_id_from_name, get_reactions_for_compound
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print("Ensure the 'apriomics' package is installed or the project root is in the Python path.")
    # Depending on the context, might raise error or exit
```

### 2. Define Input Metabolites

Specify the list of metabolite names you are interested in.

``` python
metabolite_names = ["Glucose", "Pyruvate"]
output_directory = Path("output") # Define where to save the CSV
output_filename = output_directory / "glucose_pyruvate_edges_example.csv"

# Ensure output directory exists
output_directory.mkdir(exist_ok=True)
```

### 3. Get KEGG Compound IDs

Use `get_kegg_id_from_name` to find the corresponding KEGG Compound IDs.
We’ll collect them in a set to handle potential duplicates if multiple
names map to the same ID.

``` python
print(f"Finding KEGG Compound IDs for: {metabolite_names}")
compound_ids = set()
name_to_id_map = {}
for name in metabolite_names:
    # Using exact_match=True for more specific results
    kegg_id = get_kegg_id_from_name(name, exact_match=True)
    if kegg_id:
        print(f"  Found '{name}' -> {kegg_id}")
        compound_ids.add(kegg_id)
        name_to_id_map[name] = kegg_id
    else:
        print(f"  Warning: Could not find KEGG ID for '{name}'.")

# Check if we found any IDs
if not compound_ids:
    print("Error: No KEGG Compound IDs found. Cannot proceed.")
    # Handle error appropriately
```

### 4. Get KEGG Reaction IDs

Now, use `get_reactions_for_compound` for each Compound ID to find all
associated KEGG Reaction IDs. We collect these in a set to get unique
reactions.

``` python
print(f"\nFinding KEGG Reaction IDs linked to compounds: {list(compound_ids)}")
all_reaction_ids = set()
if compound_ids:
    for c_id in compound_ids:
        reaction_ids = get_reactions_for_compound(c_id)
        if reaction_ids:
            print(f"  Found {len(reaction_ids)} reactions for {c_id}")
            all_reaction_ids.update(reaction_ids)

# Check if we found any reaction IDs
if not all_reaction_ids:
    print("Error: No KEGG Reaction IDs found for the specified compounds.")
    # Handle error appropriately
```

### 5. Process Reactions and Build Edges

Pass the list of unique Reaction IDs to `process_reactions`. This
function fetches the reaction details from KEGG and computes the signed
edges.

``` python
print(f"\nProcessing {len(all_reaction_ids)} unique reactions...")
edges_dict = {}
if all_reaction_ids:
    edges_dict = process_reactions(list(all_reaction_ids))

# The edges_dict looks like: {('C00031', 'C00022'): -1, ('C00022', 'C00031'): -1, ...}
# Keys are (metabolite_i_id, metabolite_j_id) tuples, values are sign (+1 or -1)

if not edges_dict:
    print("Warning: No edges were generated.")
else:
    print(f"Generated {len(edges_dict)} signed edges.")
    # Optionally, print a few example edges:
    # count = 0
    # for edge, sign in edges_dict.items():
    #     print(f"  Edge: {edge}, Sign: {sign}")
    #     count += 1
    #     if count >= 5:
    #         break
```

### 6. Write Edges to CSV

Finally, use `write_signed_edges_csv` to save the generated dictionary
to a CSV file.

``` python
if edges_dict:
    print(f"\nWriting edges to {output_filename}...")
    write_signed_edges_csv(edges_dict, output_filename)
    print("Processing complete.")
```

## Running the Original Script

While this document shows how to use the functions programmatically, the
original example script (`examples/use_build_signed_edges.py`) provides
a command-line interface for this workflow. You can run it from your
terminal like this:

``` bash
# Ensure the output directory exists
mkdir -p output

# Run the script
python examples/use_build_signed_edges.py Glucose Pyruvate -o output/glucose_pyruvate_edges.csv

# Example with L-Alanine and exact matching
python examples/use_build_signed_edges.py "L-Alanine" --exact-match -o output/alanine_edges.csv
```
