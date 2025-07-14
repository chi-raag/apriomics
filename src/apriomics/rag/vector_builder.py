"""
Vector builder for creating FAISS indices from HMDB metabolite chunks.

This module handles:
1. Embedding generation using BioBERT/SciiBERT
2. FAISS index creation and optimization  
3. Metadata storage for chunk retrieval
4. Index persistence and loading
"""

import json
import pickle
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
from tqdm.auto import tqdm
import sys

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependencies for vector building: {e}")
    print("Install with: uv add sentence-transformers faiss-cpu")
    sys.exit(1)

from .hmdb_parser import MetaboliteChunk, HMDBParser
from .hmdb_parser_focused import FocusedHMDBParser


class HMDBVectorBuilder:
    """
    Builds vector indices from HMDB metabolite chunks.
    
    Creates optimized FAISS indices with metadata storage for
    fast similarity search and retrieval.
    """
    
    def __init__(self, 
                 embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli",
                 device: str = "cpu",
                 embedding_dim: Optional[int] = None):
        """
        Initialize vector builder.
        
        Args:
            embedding_model: HuggingFace model for embeddings
            device: Device for embedding computation ("cpu" or "cuda")
            embedding_dim: Optional dimension for PCA reduction
        """
        # Fix for macOS multiprocessing issues and enable optimizations
        import platform
        if platform.system() == "Darwin":  # macOS
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Try to use MPS (Apple Silicon GPU) if available and device is cpu
            if device == "cpu":
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        device = "mps"
                        print("✓ Using Apple Silicon GPU (MPS) for acceleration", file=sys.stderr)
                    else:
                        # Use more CPU threads for faster processing
                        os.environ["OMP_NUM_THREADS"] = "4"
                        os.environ["MKL_NUM_THREADS"] = "4"
                        print("✓ Using multi-threaded CPU processing", file=sys.stderr)
                except ImportError:
                    os.environ["OMP_NUM_THREADS"] = "4"
                    os.environ["MKL_NUM_THREADS"] = "4"
            else:
                # For explicit cpu/cuda requests, respect the choice
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
        
        self.embedding_model_name = embedding_model
        self.device = device
        self.embedding_dim = embedding_dim
        
        print(f"Loading embedding model: {embedding_model}", file=sys.stderr)
        
        # Initialize with macOS-friendly settings
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Disable multiprocessing for sentence-transformers on macOS
        if platform.system() == "Darwin":
            # Force sequential processing
            self.embedding_model.encode = self._safe_encode_wrapper(self.embedding_model.encode)
        
        # Get actual embedding dimension
        self.native_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.final_dim = embedding_dim if embedding_dim else self.native_dim
        
        print(f"Embedding dimension: {self.native_dim} -> {self.final_dim}", file=sys.stderr)
        
        # Will be set during index building
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.pca_matrix: Optional[np.ndarray] = None
    
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
    
    def build_index_from_xml(self, 
                           xml_path: Path, 
                           output_dir: Path,
                           max_metabolites: Optional[int] = None,
                           batch_size: int = 1000,
                           use_focused_parser: bool = False) -> Path:
        """
        Build complete FAISS index from HMDB XML dump.
        
        Args:
            xml_path: Path to HMDB XML file
            output_dir: Directory to save index and metadata
            max_metabolites: Optional limit for testing
            batch_size: Batch size for embedding generation
            
        Returns:
            Path to saved index directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Building HMDB vector index...", file=sys.stderr)
        print(f"Input: {xml_path}", file=sys.stderr)
        print(f"Output: {output_dir}", file=sys.stderr)
        
        # Parse chunks from XML
        if use_focused_parser:
            parser = FocusedHMDBParser(xml_path)
            print(f"Using focused parser (biological relevance only)", file=sys.stderr)
        else:
            parser = HMDBParser(xml_path)
            print(f"Using standard parser (all HMDB data)", file=sys.stderr)
        
        chunks = list(parser.parse_metabolites(max_metabolites=max_metabolites))
        
        print(f"Parsed {len(chunks)} chunks", file=sys.stderr)
        
        if not chunks:
            raise ValueError("No chunks parsed from XML file")
        # Build index from chunks
        self.build_index_from_chunks(chunks, batch_size=batch_size)
        
        # Save index and metadata
        index_path = self.save_index(output_dir)
        
        print(f"Index saved to: {index_path}", file=sys.stderr)
        return index_path
    
    def build_index_from_chunks(self, 
                              chunks: List[MetaboliteChunk],
                              batch_size: int = 1000):
        """
        Build FAISS index from metabolite chunks using streaming approach.
        
        Args:
            chunks: List of metabolite chunks to index
            batch_size: Batch size for embedding generation
        """
        print(f"Building streaming index for {len(chunks)} chunks...", file=sys.stderr)
        print(f"Using batch size: {batch_size}", file=sys.stderr)
        
        # Create temporary directory for streaming embeddings
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"Using temporary directory: {temp_path}", file=sys.stderr)
            
            # Stream embeddings to disk and collect metadata
            self.metadata = []
            embedding_files = []
            total_vectors = 0
            
            # Phase 1: Generate embeddings and stream to disk
            print("Phase 1: Generating and streaming embeddings...", file=sys.stderr)
            
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings for this batch with optimized settings
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(64, len(batch_texts)),  # Larger internal batches
                    normalize_embeddings=True  # Do normalization during encoding
                )
                
                # Convert to float32 to save space
                batch_embeddings = batch_embeddings.astype(np.float32)
                
                # Save embeddings to temporary file
                embedding_file = temp_path / f"embeddings_batch_{i:06d}.npy"
                np.save(embedding_file, batch_embeddings)
                embedding_files.append(embedding_file)
                total_vectors += len(batch_embeddings)
                
                # Store metadata
                for chunk in batch_chunks:
                    chunk_metadata = {
                        'hmdb_id': chunk.hmdb_id,
                        'chunk_type': chunk.chunk_type,
                        'content': chunk.content,
                        'metadata': chunk.metadata
                    }
                    self.metadata.append(chunk_metadata)
                
                # Clear batch from memory immediately
                del batch_embeddings
            
            print(f"Generated {total_vectors} embeddings in {len(embedding_files)} files", file=sys.stderr)
            
            # Phase 2: Apply PCA if needed (streaming approach)
            embedding_dim = self.native_dim
            if self.embedding_dim and self.embedding_dim < self.native_dim:
                print("Phase 2: Computing PCA transformation...", file=sys.stderr)
                embedding_dim = self._compute_pca_streaming(embedding_files, total_vectors)
            
            # Phase 3: Build FAISS index incrementally
            print("Phase 3: Building FAISS index...", file=sys.stderr)
            self._build_faiss_index_streaming(embedding_files, total_vectors, embedding_dim)
            
        print(f"Streaming index built with {self.index.ntotal} vectors", file=sys.stderr)
    
    def _compute_pca_streaming(self, embedding_files: List[Path], total_vectors: int) -> int:
        """Compute PCA transformation using streaming approach for memory efficiency."""
        from sklearn.decomposition import IncrementalPCA
        
        # Use incremental PCA for large datasets
        ipca = IncrementalPCA(n_components=self.embedding_dim, batch_size=1000)
        
        print(f"Computing PCA: {self.native_dim} -> {self.embedding_dim}", file=sys.stderr)
        
        # Fit PCA incrementally
        for embedding_file in tqdm(embedding_files, desc="PCA fitting"):
            embeddings = np.load(embedding_file)
            ipca.partial_fit(embeddings)
            del embeddings
        
        # Transform and save embeddings back to disk
        for embedding_file in tqdm(embedding_files, desc="PCA transform"):
            embeddings = np.load(embedding_file)
            transformed = ipca.transform(embeddings).astype(np.float32)
            np.save(embedding_file, transformed)
            del embeddings, transformed
        
        # Store PCA components for query-time transformation
        self.pca_matrix = ipca.components_
        
        print(f"PCA explained variance ratio: {ipca.explained_variance_ratio_.sum():.3f}", file=sys.stderr)
        return self.embedding_dim
    
    def _build_faiss_index_streaming(self, embedding_files: List[Path], total_vectors: int, dim: int):
        """Build FAISS index incrementally from streaming embeddings."""
        
        # Create appropriate index based on size
        if total_vectors > 100000:
            # Use IVF for large datasets
            nlist = min(4096, total_vectors // 39)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            print(f"Training IVF index with nlist={nlist}...", file=sys.stderr)
            
            # Train index with sample of data
            training_samples = min(50000, total_vectors)
            training_data = []
            samples_collected = 0
            
            for embedding_file in embedding_files:
                if samples_collected >= training_samples:
                    break
                embeddings = np.load(embedding_file)
                needed = min(len(embeddings), training_samples - samples_collected)
                training_data.append(embeddings[:needed])
                samples_collected += needed
                del embeddings
            
            if training_data:
                training_matrix = np.vstack(training_data)
                faiss.normalize_L2(training_matrix)
                self.index.train(training_matrix)
                del training_data, training_matrix
                import gc
                gc.collect()
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatIP(dim)
        
        # Add all embeddings to index
        print("Adding embeddings to FAISS index...", file=sys.stderr)
        for embedding_file in tqdm(embedding_files, desc="Adding to index"):
            embeddings = np.load(embedding_file)
            # Skip normalization if already done during encoding
            if not hasattr(self, '_normalized_during_encoding'):
                faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            del embeddings
        
        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(32, self.index.nlist)
    
    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        from sklearn.decomposition import PCA
        
        print(f"Applying PCA: {embeddings.shape[1]} -> {self.embedding_dim}", file=sys.stderr)
        
        pca = PCA(n_components=self.embedding_dim)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Store PCA matrix for query-time transformation
        self.pca_matrix = pca.components_
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}", file=sys.stderr)
        
        return reduced_embeddings.astype(np.float32)
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create and populate FAISS index."""
        dim = embeddings.shape[1]
        
        # Create index based on size and requirements
        if embeddings.shape[0] > 100000:
            # Use IVF for large datasets
            nlist = min(4096, embeddings.shape[0] // 39)  # Rule of thumb
            quantizer = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            print(f"Training IVF index with nlist={nlist}...", file=sys.stderr)
            self.index.train(embeddings)
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatIP(dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(32, self.index.nlist)  # Search parameters
    
    def save_index(self, output_dir: Path) -> Path:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            output_dir: Directory to save index files
            
        Returns:
            Path to index directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / "hmdb_index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save configuration
        config = {
            'embedding_model': self.embedding_model_name,
            'native_dim': self.native_dim,
            'final_dim': self.final_dim,
            'num_chunks': len(self.metadata),
            'index_type': type(self.index).__name__
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save PCA matrix if used
        if self.pca_matrix is not None:
            pca_path = output_dir / "pca_matrix.npy"
            np.save(pca_path, self.pca_matrix)
        
        print(f"Index saved to {output_dir}", file=sys.stderr)
        print(f"  - FAISS index: {index_path}", file=sys.stderr)
        print(f"  - Metadata: {metadata_path}", file=sys.stderr)
        print(f"  - Config: {config_path}", file=sys.stderr)
        if self.pca_matrix is not None:
            print(f"  - PCA matrix: {pca_path}", file=sys.stderr)
        
        return output_dir
    
    @classmethod
    def load_index(cls, index_dir: Path) -> 'HMDBVectorBuilder':
        """
        Load a previously built index.
        
        Args:
            index_dir: Directory containing saved index
            
        Returns:
            Loaded HMDBVectorBuilder instance
        """
        index_dir = Path(index_dir)
        
        # Load configuration
        config_path = index_dir / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            embedding_model=config['embedding_model'],
            embedding_dim=config['final_dim'] if config['final_dim'] != config['native_dim'] else None
        )
        
        # Load FAISS index
        index_path = index_dir / "hmdb_index.faiss"
        instance.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            instance.metadata = json.load(f)
        
        # Load PCA matrix if exists
        pca_path = index_dir / "pca_matrix.npy"
        if pca_path.exists():
            instance.pca_matrix = np.load(pca_path)
        
        print(f"Loaded index from {index_dir}", file=sys.stderr)
        print(f"  - {len(instance.metadata)} chunks", file=sys.stderr)
        print(f"  - Embedding model: {config['embedding_model']}", file=sys.stderr)
        print(f"  - Dimensions: {config['native_dim']} -> {config['final_dim']}", file=sys.stderr)
        
        return instance


def build_hmdb_index_cli():
    """Command-line interface for building HMDB index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build HMDB vector index from XML dump")
    parser.add_argument("xml_path", help="Path to HMDB XML file")
    parser.add_argument("output_dir", help="Output directory for index")
    parser.add_argument("--max-metabolites", type=int, help="Limit number of metabolites (for testing)")
    parser.add_argument("--embedding-model", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli",
                      help="HuggingFace embedding model")
    parser.add_argument("--embedding-dim", type=int, help="PCA dimension reduction")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for embedding")
    parser.add_argument("--device", default="cpu", help="Device for embeddings (cpu/cuda)")
    parser.add_argument("--focused", action="store_true", help="Use focused parser (biological info only)")
    
    args = parser.parse_args()
    
    builder = HMDBVectorBuilder(
        embedding_model=args.embedding_model,
        device=args.device,
        embedding_dim=args.embedding_dim
    )
    
    index_path = builder.build_index_from_xml(
        xml_path=Path(args.xml_path),
        output_dir=Path(args.output_dir),
        max_metabolites=args.max_metabolites,
        batch_size=args.batch_size,
        use_focused_parser=args.focused
    )
    
    print(f"Index built successfully: {index_path}")


if __name__ == "__main__":
    build_hmdb_index_cli()