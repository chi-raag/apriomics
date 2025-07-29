"""
Test script to inspect the context retrieved from HMDB for a single metabolite.
"""

from chembridge.databases.hmdb import HMDBClient


def test_inspect_hmdb_context():
    """
    This test retrieves and prints the HMDB context for a single metabolite (glucose)
    using a known HMDB ID.
    """
    hmdb_id = "HMDB0000122"  # D-Glucose
    metabolite_name = "glucose"

    hmdb_client = HMDBClient()
    hmdb_contexts = {}

    try:
        metabolite_data = hmdb_client.get_metabolite_info(hmdb_id)
        if metabolite_data:
            context_parts = [f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id})"]
            if metabolite_data.get("description"):
                context_parts.append(f"Description: {metabolite_data['description']!r}")
            if metabolite_data.get("pathways"):
                pathways_str = ", ".join(metabolite_data["pathways"][:5])
                context_parts.append(f"Pathways: {pathways_str}")
            if metabolite_data.get("diseases"):
                diseases_str = ", ".join(metabolite_data["diseases"][:5])
                context_parts.append(f"Associated diseases: {diseases_str}")
            if metabolite_data.get("cellular_locations"):
                cellular_str = ", ".join(metabolite_data["cellular_locations"][:5])
                context_parts.append(f"Cellular locations: {cellular_str}")
            if metabolite_data.get("biofluid_locations"):
                biofluid_str = ", ".join(metabolite_data["biofluid_locations"][:5])
                context_parts.append(f"Biofluid locations: {biofluid_str}")
            if metabolite_data.get("tissue_locations"):
                tissue_str = ", ".join(metabolite_data["tissue_locations"][:5])
                context_parts.append(f"Tissue locations: {tissue_str}")
            hmdb_contexts[metabolite_name] = " | ".join(context_parts)
        else:
            hmdb_contexts[metabolite_name] = (
                f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id}, no data found)"
            )
    except Exception as e:
        print(f"Error fetching data for {hmdb_id}: {e}")
        hmdb_contexts[metabolite_name] = (
            f"Metabolite: {metabolite_name} (HMDB ID: {hmdb_id}, error fetching)"
        )

    # Print the context for inspection
    print(f"--- Context for {metabolite_name} ---")
    print(hmdb_contexts.get(metabolite_name, "Context not found."))
    print("\n")

    assert "glucose" in hmdb_contexts
