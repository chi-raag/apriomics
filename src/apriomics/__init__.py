"""
apriomics: A package for generating chemical similarity priors for Bayesian models

This package provides tools to:
1. Convert chemical names to SMILES strings
2. Generate molecular fingerprints
3. Create similarity matrices for use as priors in Bayesian models
"""

from .priors import (
    PriorData,
    load_metabolites_from_excel,
    get_smiles,
    generate_fingerprints,
    create_similarity_matrix,
    get_kernel,
    get_hmdb_contexts,
    get_metabolite_context_for_llm,
    get_llm_differential_priors,
    get_network_priors,
    save_results,
    run_pipeline,
    pipe,
)

from .literature import clean_pathway_list_with_llm, DSPyPathwayCleaner

# RAG components (optional imports)
try:
    from .rag import HMDBParser, HMDBVectorBuilder, HMDBRetriever
    from .rag.simple_hmdb_scraper import SimpleHMDBScraper

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Visualization components (optional imports)
try:
    from .visualization import (
        MarkovFieldVisualizer,
        LLMPriorVisualizer,
        plot_signed_network,
        plot_bayesian_scores,
        analyze_priors,
        create_comprehensive_report,
    )

    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    "PriorData",
    "load_metabolites_from_excel",
    "get_smiles",
    "generate_fingerprints",
    "create_similarity_matrix",
    "get_kernel",
    "get_hmdb_contexts",
    "get_metabolite_context_for_llm",
    "get_llm_differential_priors",
    "get_network_priors",
    "save_results",
    "run_pipeline",
    "pipe",
    "clean_pathway_list_with_llm",
    "DSPyPathwayCleaner",
    "RAG_AVAILABLE",
    "VIZ_AVAILABLE",
]

# Add RAG components to __all__ if available
if RAG_AVAILABLE:
    __all__.extend(
        ["HMDBParser", "HMDBVectorBuilder", "HMDBRetriever", "SimpleHMDBScraper"]
    )

# Add visualization components to __all__ if available
if VIZ_AVAILABLE:
    __all__.extend(
        [
            "MarkovFieldVisualizer",
            "LLMPriorVisualizer",
            "plot_signed_network",
            "plot_bayesian_scores",
            "analyze_priors",
            "create_comprehensive_report",
        ]
    )
