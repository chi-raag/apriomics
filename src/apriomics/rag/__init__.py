"""
RAG (Retrieval-Augmented Generation) module for HMDB metabolite data.

This module provides functionality to:
1. Parse large HMDB XML dumps efficiently
2. Create vector embeddings of metabolite information
3. Build and query FAISS indices for fast retrieval
4. Integrate with DSPy pipeline for enhanced LLM context
"""

from .hmdb_parser import HMDBParser, MetaboliteChunk
from .vector_builder import HMDBVectorBuilder
from .retriever import HMDBRetriever

__all__ = ["HMDBParser", "MetaboliteChunk", "HMDBVectorBuilder", "HMDBRetriever"]
