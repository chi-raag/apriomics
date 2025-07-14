import time
import requests
import json
import pandas as pd
import concurrent.futures
from tqdm.auto import tqdm

def get_smiles_from_names(names_vector, max_workers=4):
    """
    Retrieves Canonical SMILES strings for a list of metabolite names in parallel.

    This function queries the ChEBI database for each name. If a high-quality
    match is found, its SMILES string is used. Requests for different names are 
    performed concurrently using a thread pool.

    Args:
        names_vector (list): A list of metabolite names (strings) to query.
        max_workers (int, optional): The maximum number of concurrent worker
                                     threads to use for API requests.
                                     Defaults to 4. Adjust based on API limits
                                     and system resources.

    Returns:
        pandas.DataFrame: A DataFrame containing the query results, with columns:
            - metabolite: The original input metabolite name.
            - smiles: The retrieved Canonical SMILES string, or None if not found.
            - source: The database source ('ChEBI' or 'Not Found').
            - identifier: The ChEBI ID, or None.
            - match_type: The method used to find the match ('best_match' or 'not_found').
            - quality_stars: The ChEBI curation quality rating (stars), or None.
            - search_score: The ChEBI search relevance score, or None.

    Raises:
        ImportError: If the 'tqdm', 'requests', or 'pandas' libraries are not installed.
    """

    # --- Internal Helper Function: ChEBI Search ---
    def fetch_from_chebi(name):
        """
        Performs an advanced search on the ChEBI API for a given name.

        Sorts results by quality (stars) and relevance score, returning
        details of the top match if found.

        Args:
            name (str): The metabolite name to search for.

        Returns:
            dict or None: A dictionary containing SMILES and metadata if found,
                          otherwise None.
        """
        # ChEBI API endpoint for advanced search (POST request expected)
        chebi_search_url = "https://www.ebi.ac.uk/chebi/beta/api/public/advanced_search/"
        # Payload defines the search criteria (searching synonyms for the name)
        payload = {
            "text_search_specification": {
                "and_specification": [{
                    "categories": ["synonyms.name"],
                    "texts": [name]
                }]
            }
        }
        # Standard headers for JSON POST request
        headers = {'Content-Type': 'application/json', 'accept': '*/*'}
        # Query parameters for pagination and filtering (applied to the URL)
        search_params = {
            'three_star_only': 'false', # Include results with less than 3 stars
            'page': 1,
            'size': 15, # Retrieve enough results to find the best match
            'download': 'false'
        }
        # API request timeout in seconds
        api_timeout = 15

        try:
            # Execute POST request to ChEBI API
            search_response = requests.post(
                chebi_search_url, # URL includes trailing slash based on observed API behavior
                params=search_params,
                json=payload,
                headers=headers,
                timeout=api_timeout
            )
            # Raise an exception for HTTP error codes (4xx or 5xx)
            search_response.raise_for_status()
            # Parse the JSON response
            search_data = search_response.json()

            # Check if the response contains results
            if search_data.get('results') and len(search_data['results']) > 0:
                # Sort results: prioritize higher star ratings, then higher relevance scores
                results = sorted(
                    search_data['results'],
                    key=lambda x: (x['_source'].get('stars', 0), x['_score']),
                    reverse=True # Descending order (best first)
                )
                # Select the top result after sorting
                best_match = results[0]['_source']
                smiles = best_match.get('smiles')
                chebi_id = best_match.get('chebi_accession')

                # Ensure essential information (SMILES and ID) is present
                if smiles and chebi_id:
                    return {
                        "smiles": smiles,
                        "source": "ChEBI",
                        "identifier": chebi_id,
                        "match_type": "best_match", # Indicates match found via sorted results
                        "quality_stars": best_match.get('stars'),
                        "search_score": results[0]['_score']
                    }

        # Handle potential errors during the API request or response processing
        except requests.exceptions.RequestException as e:
            # Print warnings; tqdm handles progress bar display separately
            print(f"\nWarning: ChEBI API request error for '{name}': {str(e)}")
        except (KeyError, IndexError) as e:
            # Indicates unexpected structure in the API response JSON
            print(f"\nWarning: ChEBI response parsing error for '{name}': {str(e)}")
        except json.JSONDecodeError as e:
             # Indicates the response was not valid JSON
             print(f"\nWarning: ChEBI JSON decode error for '{name}': {str(e)}")

        # Return None if no suitable match was found or an error occurred
        return None

    # --- Worker Function for Parallel Execution ---
    def process_single_name(name):
        """
        Processes a single metabolite name to find its SMILES string.

        This function calls the ChEBI fetcher for one name 
        and returns a standardized result dictionary.

        Args:
            name (str): The metabolite name to process.

        Returns:
            dict: A dictionary containing the results for the processed name.
        """
        # Add a small delay before starting processing for this name
        # Helps to space out requests across workers for rate limiting.
        time.sleep(0.1) # Adjust this value as needed

        # --- Step 1: Attempt ChEBI Search ---
        chebi_result = fetch_from_chebi(name)

        # If ChEBI finds a valid SMILES string, return the formatted result
        if chebi_result and chebi_result.get("smiles"):
            return {
                "metabolite": name,
                "smiles": chebi_result["smiles"],
                "source": chebi_result["source"],
                "identifier": chebi_result["identifier"],
                "match_type": chebi_result["match_type"],
                "quality_stars": chebi_result.get("quality_stars"), # ChEBI specific
                "search_score": chebi_result.get("search_score")   # ChEBI specific
            }

        # --- Step 2: Record Not Found if ChEBI search failed ---
        return {
            "metabolite": name,
            "smiles": None,
            "source": "Not Found",
            "identifier": None,
            "match_type": "not_found",
            "quality_stars": None,
            "search_score": None
        }

    # --- Main Parallel Execution Logic ---
    results_list = []
    # Use ThreadPoolExecutor for I/O-bound tasks like network requests.
    # The 'with' statement ensures threads are cleaned up properly.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to apply the worker function to each name.
        # It returns an iterator that yields results in the order tasks were submitted.
        # This is suitable for collecting all results before DataFrame creation.
        results_iterator = executor.map(process_single_name, names_vector)

        # Wrap the results iterator with tqdm for a progress bar.
        # `total` is set to the number of input names.
        # `desc` provides a label, `unit` describes the items being processed.
        results_list = list(tqdm(results_iterator, total=len(names_vector), desc="Fetching SMILES", unit="name"))

    # Convert the list of result dictionaries (returned by workers) into a Pandas DataFrame
    return pd.DataFrame(results_list) 