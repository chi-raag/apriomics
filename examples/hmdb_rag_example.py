#!/usr/bin/env python3
"""
Example: Using HMDB RAG system for enhanced metabolite prior generation.

This example demonstrates:
1. Building HMDB vector index from XML dump (one-time setup)
2. Using RAG retriever for enhanced metabolite contexts
3. Integration with DSPy pipeline for LLM-based priors
4. Comparison with traditional API-based approach
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import apriomics
from apriomics import PriorData


def build_hmdb_index_example():
    """Example: Building HMDB vector index from XML dump."""

    print("=== Building HMDB Vector Index ===\n")

    # Check if RAG components are available
    if not apriomics.RAG_AVAILABLE:
        print("RAG components not available. Install dependencies:")
        print("uv add sentence-transformers faiss-cpu")
        return None

    from apriomics.rag import HMDBVectorBuilder

    # Path to HMDB XML dump (adjust as needed)
    xml_path = Path.home() / "Downloads" / "hmdb_metabolites.xml"

    if not xml_path.exists():
        print(f"HMDB XML file not found: {xml_path}")
        print("Please download hmdb_metabolites.xml and place it in ~/Downloads/")
        print("Or adjust the path in this example.")
        return None

    # Output directory for index
    output_dir = Path("hmdb_index")

    print(f"Building index from: {xml_path}")
    print(f"Output directory: {output_dir}")
    print("This may take 30-60 minutes for the full HMDB...")

    # Create vector builder with optimized settings
    builder = HMDBVectorBuilder(
        embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli",
        device="cpu",  # Use "cuda" if you have GPU
        embedding_dim=384,  # Reduce from 768 for smaller index
    )

    try:
        # Build index (use max_metabolites=100 for quick testing)
        index_path = builder.build_index_from_xml(
            xml_path=xml_path,
            output_dir=output_dir,
            max_metabolites=100,  # Remove this line for full HMDB
            batch_size=500,
        )

        print(f"✓ Index built successfully: {index_path}")
        return index_path

    except Exception as e:
        print(f"Error building index: {e}")
        return None


def rag_retrieval_example(index_dir: Path):
    """Example: Using HMDB RAG retriever for metabolite information."""

    print("\n=== HMDB RAG Retrieval Example ===\n")

    from apriomics.rag import HMDBRetriever

    # Load the retriever
    retriever = HMDBRetriever(index_dir)

    # Show index statistics
    stats = retriever.get_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        if key == "chunk_types":
            print(f"  {key}:")
            for chunk_type, count in value.items():
                print(f"    {chunk_type}: {count}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)

    # Example 1: Search for specific metabolite
    print("\n1. Searching for 'glucose':")
    results = retriever.search("glucose", k=5)
    for i, result in enumerate(results, 1):
        print(
            f"   {i}. {result.hmdb_id} ({result.chunk_type}) - Score: {result.score:.3f}"
        )
        print(f"      {result.content[:100]}...")

    # Example 2: Get comprehensive metabolite context
    print("\n2. Getting comprehensive context for 'glucose':")
    context = retriever.get_metabolite_context("glucose", condition="diabetes")
    print(context[:500] + "..." if len(context) > 500 else context)

    # Example 3: Search by condition
    print("\n3. Searching for diabetes-related metabolites:")
    diabetes_results = retriever.search_by_condition("diabetes", k=5)
    for result in diabetes_results:
        metabolite_name = result.metadata.get("name", result.hmdb_id)
        print(f"   - {metabolite_name} (Score: {result.score:.3f})")

    # Example 4: Find similar metabolites
    print("\n4. Finding metabolites similar to 'glucose':")
    similar = retriever.get_similar_metabolites("glucose", k=5)
    for name, score in similar:
        print(f"   - {name} (Similarity: {score:.3f})")

    return retriever


def rag_dspy_integration_example(retriever):
    """Example: Integration with DSPy pipeline for enhanced priors."""

    print("\n=== RAG + DSPy Integration Example ===\n")

    # Create sample metabolite data
    metabolites = ["glucose", "insulin", "lactate", "pyruvate", "alanine"]

    # Create PriorData with metabolite names
    priors = PriorData()
    priors.metabolite_names = metabolites

    print(f"Generating LLM priors for metabolites: {metabolites}")
    print("Condition: diabetes vs control")

    # Check if we have API key for LLM
    if not os.getenv("GOOGLE_API_KEY"):
        print("Note: GOOGLE_API_KEY not set, will use uniform priors")

        # Use RAG retriever to get enhanced contexts (without LLM scoring)
        enhanced_contexts = retriever.get_metabolite_contexts_batch(
            metabolites, condition="diabetes"
        )

        print("\nEnhanced contexts from HMDB RAG:")
        for metabolite, context in enhanced_contexts.items():
            print(f"\n{metabolite}:")
            print(f"  {context[:200]}...")

        return

    try:
        # Use RAG-enhanced LLM priors
        llm_priors = apriomics.get_llm_differential_priors(
            priors=priors,
            condition="diabetes vs control",
            hmdb_retriever=retriever,  # Use RAG instead of traditional HMDB API
            use_dspy=True,
        )

        print("\nLLM-generated importance scores (with RAG enhancement):")
        for metabolite, score in sorted(
            llm_priors.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {metabolite}: {score:.3f}")

    except Exception as e:
        print(f"Error generating LLM priors: {e}")
        print("This might be due to DSPy configuration or API issues")


def comparison_example():
    """Example: Compare traditional vs RAG approach."""

    print("\n=== Traditional vs RAG Comparison ===\n")

    metabolites = ["glucose", "lactate"]

    print("1. Traditional HMDB API approach:")
    try:
        # Traditional approach using individual API calls
        traditional_contexts = apriomics.batch_get_metabolite_contexts(
            metabolites,
            hmdb_mapping={"glucose": "HMDB0000122", "lactate": "HMDB0000190"},
        )

        for metabolite, context in traditional_contexts.items():
            print(f"   {metabolite}: {context[:150]}...")

    except Exception as e:
        print(f"   Error with traditional approach: {e}")

    print("\n2. RAG approach (if index is available):")

    index_dir = Path("hmdb_index")
    if index_dir.exists():
        try:
            from apriomics.rag import HMDBRetriever

            retriever = HMDBRetriever(index_dir)

            rag_contexts = retriever.get_metabolite_contexts_batch(
                metabolites, condition="metabolic disorders"
            )

            for metabolite, context in rag_contexts.items():
                print(f"   {metabolite}: {context[:150]}...")

        except Exception as e:
            print(f"   Error with RAG approach: {e}")
    else:
        print("   RAG index not available (run build_hmdb_index_example first)")


def main():
    """Main example runner."""

    print("HMDB RAG System Examples")
    print("=" * 50)

    # Example 1: Build index (optional, can skip if already built)
    print("\nWould you like to build a new HMDB index? (y/n): ", end="")
    if input().lower().startswith("y"):
        index_path = build_hmdb_index_example()
        if index_path is None:
            print("Failed to build index, using existing if available")
            index_path = Path("hmdb_index")
    else:
        index_path = Path("hmdb_index")

    # Example 2: RAG retrieval
    if index_path.exists():
        retriever = rag_retrieval_example(index_path)

        # Example 3: DSPy integration
        rag_dspy_integration_example(retriever)
    else:
        print(f"Index not found at {index_path}")
        print("Run the build_hmdb_index_example first")

    # Example 4: Comparison
    comparison_example()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Build full HMDB index (remove max_metabolites limit)")
    print("2. Use in your own metabolomics pipelines")
    print("3. Experiment with different embedding models")
    print("4. Try different DSPy optimizers for prompt tuning")


if __name__ == "__main__":
    main()
