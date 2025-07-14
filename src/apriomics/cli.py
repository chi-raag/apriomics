"""
Command-line interface for apriomics RAG system.

Provides commands for building and managing HMDB vector indices.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

def build_hmdb_index():
    """Build HMDB vector index from XML dump."""
    parser = argparse.ArgumentParser(
        description="Build HMDB vector index for RAG-based retrieval",
        prog="apriomics build-index"
    )
    
    parser.add_argument(
        "xml_path", 
        help="Path to HMDB XML dump file (hmdb_metabolites.xml)"
    )
    
    parser.add_argument(
        "output_dir", 
        help="Output directory for vector index"
    )
    
    parser.add_argument(
        "--max-metabolites", 
        type=int, 
        help="Limit number of metabolites (for testing)"
    )
    
    parser.add_argument(
        "--embedding-model", 
        default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli",
        help="HuggingFace embedding model (default: BioBERT)"
    )
    
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        help="PCA dimension reduction (e.g., 384)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1000, 
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--device", 
        default="cpu", 
        choices=["cpu", "cuda"],
        help="Device for embedding computation"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid loading dependencies unless needed
    try:
        from .rag.vector_builder import HMDBVectorBuilder
    except ImportError as e:
        print(f"Error: Missing dependencies for index building: {e}")
        print("Install with: uv add sentence-transformers faiss-cpu")
        sys.exit(1)
    
    print("Building HMDB vector index...")
    print(f"Input XML: {args.xml_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding model: {args.embedding_model}")
    
    if args.max_metabolites:
        print(f"Limiting to {args.max_metabolites} metabolites (testing mode)")
    
    builder = HMDBVectorBuilder(
        embedding_model=args.embedding_model,
        device=args.device,
        embedding_dim=args.embedding_dim
    )
    
    try:
        index_path = builder.build_index_from_xml(
            xml_path=Path(args.xml_path),
            output_dir=Path(args.output_dir),
            max_metabolites=args.max_metabolites,
            batch_size=args.batch_size
        )
        
        print(f"✓ Index built successfully: {index_path}")
        
        # Show some stats
        from .rag.retriever import HMDBRetriever
        retriever = HMDBRetriever(index_path)
        stats = retriever.get_stats()
        
        print("\nIndex Statistics:")
        for key, value in stats.items():
            if key == "chunk_types":
                print(f"  {key}:")
                for chunk_type, count in value.items():
                    print(f"    {chunk_type}: {count}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)


def search_hmdb():
    """Search HMDB index."""
    parser = argparse.ArgumentParser(
        description="Search HMDB metabolite database",
        prog="apriomics search"
    )
    
    parser.add_argument(
        "index_dir", 
        help="Path to HMDB index directory"
    )
    
    parser.add_argument(
        "query", 
        help="Search query (metabolite name, condition, etc.)"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=10, 
        help="Number of results to return"
    )
    
    parser.add_argument(
        "--chunk-types", 
        nargs="+", 
        choices=["basic", "pathways", "diseases", "tissues", "concentrations"],
        help="Filter by chunk types"
    )
    
    parser.add_argument(
        "--min-score", 
        type=float, 
        default=0.0, 
        help="Minimum similarity score threshold"
    )
    
    parser.add_argument(
        "--metabolite-context", 
        action="store_true",
        help="Get comprehensive context for a specific metabolite"
    )
    
    args = parser.parse_args()
    
    try:
        from .rag.retriever import HMDBRetriever
    except ImportError as e:
        print(f"Error: Missing dependencies for search: {e}")
        print("Install with: uv add sentence-transformers faiss-cpu")
        sys.exit(1)
    
    try:
        retriever = HMDBRetriever(Path(args.index_dir))
        
        if args.metabolite_context:
            # Get comprehensive metabolite context
            context = retriever.get_metabolite_context(args.query)
            print(f"Context for '{args.query}':")
            print("=" * 50)
            print(context)
        else:
            # Regular search
            results = retriever.search(
                query=args.query,
                k=args.k,
                chunk_types=args.chunk_types,
                min_score=args.min_score
            )
            
            print(f"Search results for '{args.query}' ({len(results)} results):")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.hmdb_id} ({result.chunk_type}) - Score: {result.score:.3f}")
                print(f"   {result.content[:150]}...")
                if len(result.content) > 150:
                    print("   [truncated]")
                print()
                
    except Exception as e:
        print(f"Error during search: {e}")
        sys.exit(1)


def show_stats():
    """Show HMDB index statistics."""
    parser = argparse.ArgumentParser(
        description="Show HMDB index statistics",
        prog="apriomics stats"
    )
    
    parser.add_argument(
        "index_dir", 
        help="Path to HMDB index directory"
    )
    
    args = parser.parse_args()
    
    try:
        from .rag.retriever import HMDBRetriever
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        sys.exit(1)
    
    try:
        retriever = HMDBRetriever(Path(args.index_dir))
        stats = retriever.get_stats()
        
        print("HMDB Index Statistics")
        print("=" * 30)
        
        for key, value in stats.items():
            if key == "chunk_types":
                print(f"{key}:")
                for chunk_type, count in value.items():
                    print(f"  {chunk_type}: {count:,}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="apriomics: Chemical similarity priors with HMDB RAG",
        prog="apriomics"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build index command
    build_parser = subparsers.add_parser(
        "build-index", 
        help="Build HMDB vector index from XML dump"
    )
    
    # Search command  
    search_parser = subparsers.add_parser(
        "search", 
        help="Search HMDB metabolite database"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats", 
        help="Show index statistics"
    )
    
    args = parser.parse_args()
    
    if args.command == "build-index":
        build_hmdb_index()
    elif args.command == "search":
        search_hmdb()
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()