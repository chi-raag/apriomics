"""
HMDB retriever for RAG-based metabolite information retrieval.

This module provides fast semantic search over HMDB metabolite data
using pre-built FAISS indices.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys

# Import macOS fixes first to suppress warnings
from ..utils.macos_fixes import suppress_stderr, capture_warnings

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependencies for retrieval: {e}")
    print("Install with: uv add sentence-transformers faiss-cpu")
    sys.exit(1)


@dataclass
class RetrievalResult:
    """Result from HMDB retrieval."""
    hmdb_id: str
    chunk_type: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"RetrievalResult(hmdb_id='{self.hmdb_id}', chunk_type='{self.chunk_type}', score={self.score:.3f})"


class HMDBRetriever:
    """
    Fast semantic retrieval over HMDB metabolite database.
    
    Provides similarity search and context generation for metabolites
    using pre-built FAISS indices and BioBERT embeddings.
    """
    
    def __init__(self, index_dir: Path):
        """
        Initialize retriever with pre-built index.
        
        Args:
            index_dir: Directory containing FAISS index and metadata
        """
        # Fix for macOS multiprocessing issues during inference
        import platform
        if platform.system() == "Darwin":  # macOS
            import multiprocessing
            import os
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.index_dir = Path(index_dir)
        
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load configuration
        config_path = self.index_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load embedding model with warning suppression
        print(f"Loading embedding model: {self.config['embedding_model']}", file=sys.stderr)
        with capture_warnings():
            self.embedding_model = SentenceTransformer(self.config['embedding_model'], device="cpu")
        
        # Force single-threaded inference on macOS
        import platform
        if platform.system() == "Darwin":
            self.embedding_model.encode = self._safe_encode_wrapper(self.embedding_model.encode)
        
        # Load FAISS index
        index_path = self.index_dir / "hmdb_index.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load PCA matrix if exists
        pca_path = self.index_dir / "pca_matrix.npy"
        self.pca_matrix = None
        if pca_path.exists():
            self.pca_matrix = np.load(pca_path)
        
        print(f"HMDB retriever loaded: {len(self.metadata)} chunks", file=sys.stderr)
    
    def _safe_encode_wrapper(self, original_encode):
        """Wrapper for encode method to handle macOS multiprocessing issues."""
        def safe_encode(*args, **kwargs):
            # Force specific settings for macOS stability
            kwargs['show_progress_bar'] = False
            kwargs['convert_to_numpy'] = True
            # Remove any multiprocessing parameters
            kwargs.pop('num_workers', None)
            return original_encode(*args, **kwargs)
        return safe_encode
    
    def search(self, 
               query: str, 
               k: int = 10, 
               chunk_types: Optional[List[str]] = None,
               min_score: float = 0.0) -> List[RetrievalResult]:
        """
        Search for relevant metabolite information.
        
        Args:
            query: Search query (metabolite name, condition, etc.)
            k: Number of results to return
            chunk_types: Optional filter by chunk types ('basic', 'pathways', etc.)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Ensure float32 for FAISS compatibility
        query_embedding = query_embedding.astype(np.float32)
        
        # Apply PCA if used during index building
        if self.pca_matrix is not None:
            query_embedding = query_embedding @ self.pca_matrix.T.astype(np.float32)
        
        # Ensure contiguous array for FAISS
        query_embedding = np.ascontiguousarray(query_embedding)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k * 3)  # Get extra results for filtering
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            chunk_meta = self.metadata[idx]
            
            # Apply chunk type filter
            if chunk_types and chunk_meta['chunk_type'] not in chunk_types:
                continue
            
            # Apply score threshold
            if score < min_score:
                continue
            
            result = RetrievalResult(
                hmdb_id=chunk_meta['hmdb_id'],
                chunk_type=chunk_meta['chunk_type'],
                content=chunk_meta['content'],
                score=float(score),
                metadata=chunk_meta['metadata']
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_metabolite_context(self, 
                             metabolite_name: str, 
                             condition: Optional[str] = None,
                             max_chunks: int = 5) -> str:
        """
        Get comprehensive context for a specific metabolite.
        
        Args:
            metabolite_name: Name of the metabolite
            condition: Optional condition/disease context
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string
        """
        # Build search query
        query = metabolite_name
        if condition:
            query = f"{metabolite_name} {condition}"
        
        # Search for relevant chunks
        results = self.search(query, k=max_chunks * 2)  # Get extra for diversity
        
        # Group by chunk type for balanced representation
        chunks_by_type = {}
        for result in results:
            chunk_type = result.chunk_type
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(result)
        
        # Select diverse chunks
        selected_chunks = []
        chunk_types_order = ['basic', 'pathways', 'diseases', 'tissues', 'concentrations']
        
        for chunk_type in chunk_types_order:
            if chunk_type in chunks_by_type and len(selected_chunks) < max_chunks:
                selected_chunks.append(chunks_by_type[chunk_type][0])  # Take best from each type
        
        # Fill remaining slots with best results
        for result in results:
            if result not in selected_chunks and len(selected_chunks) < max_chunks:
                selected_chunks.append(result)
        
        # Format context
        if not selected_chunks:
            return f"No detailed information available for {metabolite_name}"
        
        context_parts = [f"Information about {metabolite_name}:"]
        
        for i, chunk in enumerate(selected_chunks, 1):
            context_parts.append(f"{i}. {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def get_metabolite_contexts_batch(self, 
                                    metabolite_names: List[str],
                                    condition: Optional[str] = None,
                                    max_chunks_per_metabolite: int = 3) -> Dict[str, str]:
        """
        Get contexts for multiple metabolites efficiently.
        
        Args:
            metabolite_names: List of metabolite names
            condition: Optional condition context
            max_chunks_per_metabolite: Max chunks per metabolite
            
        Returns:
            Dictionary mapping metabolite names to contexts
        """
        contexts = {}
        
        for metabolite in metabolite_names:
            try:
                context = self.get_metabolite_context(
                    metabolite, 
                    condition=condition,
                    max_chunks=max_chunks_per_metabolite
                )
                contexts[metabolite] = context
            except Exception as e:
                print(f"Error getting context for {metabolite}: {e}", file=sys.stderr)
                contexts[metabolite] = f"Error retrieving information for {metabolite}"
        
        return contexts
    
    def search_by_condition(self, 
                          condition: str, 
                          k: int = 20,
                          chunk_types: Optional[List[str]] = None) -> List[RetrievalResult]:
        """
        Search for metabolites relevant to a specific condition.
        
        Args:
            condition: Disease or experimental condition
            k: Number of results
            chunk_types: Optional chunk type filter
            
        Returns:
            List of relevant metabolite chunks
        """
        return self.search(
            query=condition,
            k=k,
            chunk_types=chunk_types or ['diseases', 'pathways']
        )
    
    def get_similar_metabolites(self, 
                              metabolite_name: str, 
                              k: int = 10) -> List[Tuple[str, float]]:
        """
        Find metabolites similar to the given one.
        
        Args:
            metabolite_name: Reference metabolite name
            k: Number of similar metabolites to return
            
        Returns:
            List of (metabolite_name, similarity_score) tuples
        """
        # Search for basic information about similar metabolites
        results = self.search(
            query=metabolite_name,
            k=k * 2,
            chunk_types=['basic']
        )
        
        # Extract unique metabolites with their best scores
        metabolite_scores = {}
        for result in results:
            name = result.metadata.get('name', result.hmdb_id)
            if name.lower() != metabolite_name.lower():  # Exclude self
                if name not in metabolite_scores or result.score > metabolite_scores[name]:
                    metabolite_scores[name] = result.score
        
        # Sort by score and return top k
        similar_metabolites = sorted(
            metabolite_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return similar_metabolites
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded index."""
        chunk_type_counts = {}
        metabolite_ids = set()
        
        for chunk_meta in self.metadata:
            chunk_type = chunk_meta['chunk_type']
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            metabolite_ids.add(chunk_meta['hmdb_id'])
        
        return {
            'total_chunks': len(self.metadata),
            'unique_metabolites': len(metabolite_ids),
            'chunk_types': chunk_type_counts,
            'embedding_model': self.config['embedding_model'],
            'embedding_dimension': self.config['final_dim'],
            'index_type': self.config['index_type']
        }


def search_cli():
    """Command-line interface for HMDB search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search HMDB metabolite database")
    parser.add_argument("index_dir", help="Path to HMDB index directory")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--chunk-types", nargs="+", help="Filter by chunk types")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    retriever = HMDBRetriever(Path(args.index_dir))
    
    if args.stats:
        stats = retriever.get_stats()
        print("Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
    
    results = retriever.search(
        query=args.query,
        k=args.k,
        chunk_types=args.chunk_types,
        min_score=args.min_score
    )
    
    print(f"Search results for '{args.query}':")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.hmdb_id} ({result.chunk_type}) - Score: {result.score:.3f}")
        print(f"   {result.content[:200]}...")
        print()


if __name__ == "__main__":
    search_cli()