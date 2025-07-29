"""
Utilities for interacting with the KEGG database API.
"""

import requests
import sys
import typing

KEGG_FIND_URL = "https://rest.kegg.jp/find/compound/{}"
KEGG_LINK_REACTION_URL = "https://rest.kegg.jp/link/reaction/{}"


def get_kegg_id_from_name(
    metabolite_name: str, exact_match: bool = True
) -> typing.Union[str, None]:
    """
    Finds the KEGG Compound ID for a given metabolite name using the KEGG REST API.

    Args:
        metabolite_name: The common name of the metabolite (e.g., "Glucose").
        exact_match: If True, requires an exact match between the input name
                     and one of the names returned by KEGG (case-insensitive).
                     If False, returns the first ID found containing the name.

    Returns:
        The KEGG Compound ID (e.g., "C00022") or None if not found or an error occurs.
    """
    if not metabolite_name:
        print("Warning: Empty metabolite name provided.", file=sys.stderr)
        return None

    # Use lowercase for search and comparison, URL encode the original name
    search_term = metabolite_name.lower()
    url_encoded_name = requests.utils.quote(metabolite_name)
    url = KEGG_FIND_URL.format(url_encoded_name)

    try:
        response = requests.get(url, timeout=15)  # Increased timeout slightly
        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
        text = response.text.strip()

        if not text:
            # KEGG returns empty string for no matches
            # print(f"Info: No results found for '{metabolite_name}' on KEGG.", file=sys.stderr)
            return None

        lines = text.splitlines()
        possible_matches = []
        first_match_id = None

        for line in lines:
            if "\t" not in line:  # Ensure the line has the expected format
                continue
            kegg_id_full, names_str = line.split("\t", 1)

            # Extract Cxxxxx from cpd:Cxxxxx or similar prefixes
            if ":" in kegg_id_full:
                kegg_id = kegg_id_full.split(":")[1]
            else:
                continue  # Skip if ID format is unexpected

            if first_match_id is None:
                first_match_id = kegg_id  # Store the first ID found

            # Check for exact match (case-insensitive)
            names = [name.strip().lower() for name in names_str.split(";")]
            if search_term in names:
                if (
                    kegg_id not in possible_matches
                ):  # Avoid duplicate IDs if name appears multiple times
                    possible_matches.append(kegg_id)

        if exact_match:
            if len(possible_matches) == 1:
                return possible_matches[0]
            elif len(possible_matches) > 1:
                print(
                    f"Warning: Found multiple exact KEGG ID matches for '{metabolite_name}': {possible_matches}. Returning the first one found: {possible_matches[0]}.",
                    file=sys.stderr,
                )
                return possible_matches[0]  # Return first exact match found
            else:
                # print(f"Info: Found KEGG results for '{metabolite_name}' but no exact name match.", file=sys.stderr)
                return None  # No exact match found
        else:
            # Return the first ID found if exact_match is False and any results exist
            if first_match_id:
                return first_match_id
            else:
                # This case should ideally not be reached if text was not empty, but included for safety
                return None

    except requests.exceptions.Timeout:
        print(
            f"Error: Timeout occurred while fetching KEGG data for '{metabolite_name}'.",
            file=sys.stderr,
        )
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KEGG data for '{metabolite_name}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Catch potential errors during parsing etc.
        print(
            f"An unexpected error occurred while processing '{metabolite_name}': {e}",
            file=sys.stderr,
        )
        return None


def get_reactions_for_compound(compound_id: str) -> list[str]:
    """
    Finds KEGG Reaction IDs associated with a given KEGG Compound ID using the KEGG REST API.

    Args:
        compound_id: The KEGG Compound ID (e.g., "C00022"). It should be the ID only,
                     without any prefix like "cpd:".

    Returns:
        A list of associated KEGG Reaction IDs (e.g., ["R00123", "R00456"]),
        or an empty list if none are found or an error occurs.
    """
    if not compound_id or not compound_id.startswith("C"):
        print(
            f"Warning: Invalid or empty compound ID provided: '{compound_id}'. Expected format like 'Cxxxxx'.",
            file=sys.stderr,
        )
        return []

    url = KEGG_LINK_REACTION_URL.format(compound_id)
    reaction_ids = []

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        text = response.text.strip()

        if not text:
            # print(f"Info: No reactions found linked to compound '{compound_id}'.", file=sys.stderr)
            return []

        lines = text.splitlines()
        for line in lines:
            if "\t" not in line:
                continue
            c_id_full, r_id_full = line.split("\t", 1)
            # Extract Rxxxxx from rn:Rxxxxx
            if ":" in r_id_full:
                r_id = r_id_full.split(":")[1]
                reaction_ids.append(r_id)
            else:
                print(
                    f"Warning: Unexpected reaction ID format found for compound '{compound_id}': '{r_id_full}'. Skipping.",
                    file=sys.stderr,
                )

    except requests.exceptions.Timeout:
        print(
            f"Error: Timeout occurred while fetching linked reactions for compound '{compound_id}'.",
            file=sys.stderr,
        )
        return []  # Return empty list on error
    except requests.exceptions.RequestException as e:
        print(
            f"Error fetching linked reactions for compound '{compound_id}': {e}",
            file=sys.stderr,
        )
        return []  # Return empty list on error
    except Exception as e:
        print(
            f"An unexpected error occurred while processing linked reactions for '{compound_id}': {e}",
            file=sys.stderr,
        )
        return []  # Return empty list on error

    return reaction_ids


# Example usage:
# if __name__ == "__main__":
#     names_to_test = ["Glucose", "Pyruvate", "ATP", "L-Alanine", "NonExistentMetabolite", "water"]
#     print("--- Testing Exact Matches ---")
#     for name in names_to_test:
#         kegg_id = get_kegg_id_from_name(name, exact_match=True)
#         print(f"Name: '{name}' -> KEGG ID: {kegg_id}")

#     print("\n--- Testing First Match (Non-Exact) ---")
#     for name in names_to_test:
#          kegg_id_non_exact = get_kegg_id_from_name(name, exact_match=False)
#          print(f"Name: '{name}' -> KEGG ID: {kegg_id_non_exact}")

#     # Example of ambiguity
#     print("\n--- Testing Ambiguous Name ---")
#     ambiguous_name = "Alanine"
#     kegg_id_ambiguous = get_kegg_id_from_name(ambiguous_name, exact_match=True)
#     print(f"Name: '{ambiguous_name}' -> KEGG ID: {kegg_id_ambiguous}")
#     kegg_id_ambiguous_first = get_kegg_id_from_name(ambiguous_name, exact_match=False)
#     print(f"Name: '{ambiguous_name}' (first match) -> KEGG ID: {kegg_id_ambiguous_first}")
