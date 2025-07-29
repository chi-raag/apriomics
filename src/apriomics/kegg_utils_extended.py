import requests
from collections import defaultdict
import time
from typing import List, Dict, Set, Tuple

KEGG_GET_URL = "http://rest.kegg.jp/get/"


def get_reaction_participants(reaction_id: str) -> List[str]:
    """
    Fetches a KEGG reaction entry and parses it to find all participating compounds.

    Args:
        reaction_id: The KEGG Reaction ID (e.g., "R00123").

    Returns:
        A list of KEGG Compound IDs (e.g., ["C00022", "C00031"]) involved in the reaction.
        Returns an empty list if the reaction is not found or an error occurs.
    """
    url = KEGG_GET_URL + reaction_id
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        participants = []
        in_equation_section = False

        for line in response.text.splitlines():
            if line.startswith("EQUATION"):
                in_equation_section = True
                # Clean up the line
                line = line.replace("EQUATION", "").strip()

            if in_equation_section:
                parts = line.split()
                for part in parts:
                    if part.startswith("C") and part[1:].isdigit():
                        participants.append(part)
                # If the line does not start with a space, the equation section has ended
                if not line.startswith(" "):
                    in_equation_section = False

        # Return only unique participants
        return list(set(participants))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching reaction data for '{reaction_id}': {e}")
        return []


def get_metabolite_reaction_edges(
    metabolite_kegg_ids: List[str], kegg_utils_module, sleep_time: float = 0.1
) -> List[Tuple[str, str]]:
    """
    Builds a list of edges between metabolites that share a common reaction.

    Args:
        metabolite_kegg_ids: A list of KEGG compound IDs to build the network from.
        kegg_utils_module: The original kegg_utils module containing `get_reactions_for_compound`.
        sleep_time: Delay between API calls to be respectful to the KEGG API.

    Returns:
        A list of tuples, where each tuple is a pair of compound IDs representing an edge.
    """
    # Map from reaction ID to all metabolites in that reaction
    reaction_to_metabolites: Dict[str, Set[str]] = defaultdict(set)

    # Map from each metabolite to the set of reactions it's in
    metabolite_to_reactions: Dict[str, Set[str]] = defaultdict(set)

    print(f"Finding reactions for {len(metabolite_kegg_ids)} metabolites...")
    for kegg_id in metabolite_kegg_ids:
        reactions = kegg_utils_module.get_reactions_for_compound(kegg_id)
        metabolite_to_reactions[kegg_id].update(reactions)
        time.sleep(sleep_time)  # API rate limiting

    print("Fetching reaction participants...")
    all_reactions = set.union(*metabolite_to_reactions.values())
    for reaction_id in all_reactions:
        participants = get_reaction_participants(reaction_id)
        # We only care about participants that are in our original list
        valid_participants = [p for p in participants if p in metabolite_kegg_ids]
        if len(valid_participants) > 1:
            reaction_to_metabolites[reaction_id].update(valid_participants)
        time.sleep(sleep_time)  # API rate limiting

    edges: Set[Tuple[str, str]] = set()
    for reaction, metabolites in reaction_to_metabolites.items():
        met_list = sorted(list(metabolites))  # Sort to ensure consistent edge pairs
        for i in range(len(met_list)):
            for j in range(i + 1, len(met_list)):
                edges.add((met_list[i], met_list[j]))

    print(f"Found {len(edges)} unique metabolic edges.")
    return list(edges)


def filter_reactions_by_metabolite_overlap(
    reaction_ids: List[str],
    target_metabolites: List[str],
    min_overlap: int = 2,
    sleep_time: float = 0.1,
) -> List[str]:
    """
    Filter reactions to only those involving multiple target metabolites.

    This function reduces network noise by focusing on reactions that connect
    the metabolites of interest to each other, rather than including all
    reactions each metabolite participates in.

    Args:
        reaction_ids: List of KEGG reaction IDs to filter
        target_metabolites: List of KEGG compound IDs we're interested in
        min_overlap: Minimum number of target metabolites that must participate
                    in a reaction for it to be included (default: 2)
        sleep_time: Delay between API calls to be respectful to KEGG API

    Returns:
        Filtered list of reaction IDs that involve at least min_overlap
        target metabolites

    Example:
        >>> target_mets = ['C00002', 'C00008', 'C00020']  # ATP, ADP, AMP
        >>> all_reactions = ['R00086', 'R00127', ...]  # 500+ reactions
        >>> filtered = filter_reactions_by_metabolite_overlap(
        ...     all_reactions, target_mets, min_overlap=2
        ... )
        >>> len(filtered)  # Much smaller, more focused set
        25
    """
    if not reaction_ids or not target_metabolites:
        return []

    if min_overlap < 1:
        raise ValueError("min_overlap must be at least 1")

    if min_overlap > len(target_metabolites):
        print(
            f"Warning: min_overlap ({min_overlap}) is greater than number of target metabolites ({len(target_metabolites)})"
        )
        return []

    target_set = set(target_metabolites)
    filtered_reactions = []

    print(
        f"Filtering {len(reaction_ids)} reactions for {min_overlap}+ target metabolite overlap..."
    )

    for i, reaction_id in enumerate(reaction_ids):
        if i > 0 and i % 100 == 0:
            print(f"  Processed {i}/{len(reaction_ids)} reactions...")

        # Get all participants in this reaction
        participants = get_reaction_participants(reaction_id)

        # Count how many are in our target set
        target_participants = target_set.intersection(participants)

        if len(target_participants) >= min_overlap:
            filtered_reactions.append(reaction_id)

        # Rate limiting
        time.sleep(sleep_time)

    print(
        f"✅ Filtered to {len(filtered_reactions)} reactions with {min_overlap}+ target metabolites"
    )
    print(
        f"   Reduction: {len(reaction_ids)} → {len(filtered_reactions)} ({len(filtered_reactions) / len(reaction_ids) * 100:.1f}%)"
    )

    return filtered_reactions
