"""
macOS-specific fixes for multiprocessing and warning suppression.

This module provides utilities to handle macOS-specific issues with
multiprocessing libraries like sentence-transformers and FAISS.
"""

import os
import sys
import warnings
import platform
from typing import Optional
import contextlib
import io


def suppress_multiprocessing_warnings():
    """Completely suppress multiprocessing resource tracker warnings on macOS."""
    
    if platform.system() == "Darwin":  # macOS only
        # Method 1: Filter specific warnings
        warnings.filterwarnings("ignore", message=".*resource_tracker.*")
        warnings.filterwarnings("ignore", message=".*leaked semaphore.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
        
        # Method 2: Monkey patch the resource tracker warning function
        try:
            import multiprocessing.resource_tracker
            original_warn = warnings.warn
            
            def silent_warn(message, category=None, filename='', lineno=-1, file=None, stacklevel=1):
                if 'resource_tracker' in str(message) or 'leaked semaphore' in str(message):
                    return  # Suppress these specific warnings
                return original_warn(message, category, stacklevel=stacklevel)
            
            warnings.warn = silent_warn
            
        except ImportError:
            pass
        
        # Method 3: Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        # Method 4: Set multiprocessing start method
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to completely suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextlib.contextmanager 
def capture_warnings():
    """Context manager to capture and suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def setup_clean_environment():
    """Set up a clean environment for ML libraries on macOS."""
    
    # Suppress all multiprocessing warnings
    suppress_multiprocessing_warnings()
    
    # Additional environment setup for sentence-transformers
    if platform.system() == "Darwin":
        os.environ.update({
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_VERBOSITY": "error",
            "PYTHONWARNINGS": "ignore::UserWarning:multiprocessing.resource_tracker",
        })


# Auto-setup when module is imported
setup_clean_environment()