"""
Package data directory for pre-built indices and configurations.

Contains bundled HMDB vector indices and helper functions for accessing them.
"""

from pathlib import Path
import sys

# Get package data directory
DATA_DIR = Path(__file__).parent
HMDB_INDEX_DIR = DATA_DIR / "hmdb_index"


def get_default_hmdb_index() -> Path:
    """
    Get path to bundled HMDB index.
    
    Returns:
        Path to the default HMDB vector index directory
        
    Raises:
        FileNotFoundError: If the index is not found
    """
    if HMDB_INDEX_DIR.exists():
        return HMDB_INDEX_DIR
    else:
        raise FileNotFoundError(
            f"HMDB index not found at {HMDB_INDEX_DIR}. "
            "Build with: python -m apriomics.cli build-index <xml_path> <output_dir>"
        )


def get_default_retriever():
    """
    Get HMDB retriever with bundled index.
    
    Returns:
        HMDBRetriever instance using the default bundled index
    """
    try:
        from ..rag import HMDBRetriever
    except ImportError as e:
        print(f"RAG components not available: {e}", file=sys.stderr)
        print("Install with: uv add sentence-transformers faiss-cpu", file=sys.stderr)
        raise
    
    index_path = get_default_hmdb_index()
    return HMDBRetriever(index_path)


def is_index_available() -> bool:
    """Check if the default HMDB index is available."""
    return HMDB_INDEX_DIR.exists() and (HMDB_INDEX_DIR / "hmdb_index.faiss").exists()


def get_index_info() -> dict:
    """
    Get information about the bundled index.
    
    Returns:
        Dictionary with index statistics and metadata
    """
    if not is_index_available():
        return {"available": False, "error": "Index not found"}
    
    try:
        retriever = get_default_retriever()
        stats = retriever.get_stats()
        stats["available"] = True
        stats["location"] = str(HMDB_INDEX_DIR)
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}


__all__ = [
    'DATA_DIR',
    'HMDB_INDEX_DIR', 
    'get_default_hmdb_index',
    'get_default_retriever',
    'is_index_available',
    'get_index_info'
]