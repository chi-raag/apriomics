# apriomics

A Python package for LLM-based prior elicitation in Bayesian metabolomics models.

## Overview

`apriomics` provides tools for generating informative priors for Bayesian metabolomics analysis using Large Language Models (LLMs). The package enables:

1. **LLM-based prior elicitation**: Query LLMs to predict metabolite effect sizes and uncertainties for different experimental conditions
2. **Dual approach support**: Both categorical mapping (qualitative predictions → numerical priors) and direct numerical estimation
3. **Database-enhanced predictions**: Optional integration with HMDB (Human Metabolome Database) for context-aware predictions

The package focuses on prior generation for metabolomics studies, leveraging LLM knowledge for informed Bayesian inference.

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/chi-raag/apriomics.git
```

## Requirements

- Python >= 3.13
- OpenAI API key (for LLM-based priors)
- Core dependencies: pandas, numpy, requests, openai, pymc

## Usage

### Basic Usage

```python
from apriomics.priors.base import get_llm_priors, get_llm_quantitative_priors
from apriomics.priors.base import PriorData

# Define metabolites and experimental condition
metabolites = ["glucose", "lactate", "acetoacetate", "pyruvate"]
condition = "type 2 diabetes vs healthy controls"

# Create PriorData object
priors = PriorData(metabolites=metabolites)

# Option 1: Categorical mapping approach (qualitative → numerical)
llm_priors = get_llm_priors(
    priors=priors,
    condition=condition,
    model_name="gpt-4o-2024-08-06",
    use_hmdb_context=True
)

# Option 2: Direct numerical estimation
quantitative_priors = get_llm_quantitative_priors(
    priors=priors,
    condition=condition,
    model_name="gpt-4o-2024-08-06",
    use_hmdb_context=True
)

# Access prior parameters
for metabolite in metabolites:
    mu = llm_priors[metabolite]['mu']  # Prior mean
    sigma = llm_priors[metabolite]['sigma']  # Prior std
    print(f"{metabolite}: μ={mu:.3f}, σ={sigma:.3f}")
```

### Using Priors in PyMC Models

The generated priors can be used directly in Bayesian models:

```python
import pymc as pm

with pm.Model() as model:
    # Use LLM-informed priors
    for i, metabolite in enumerate(metabolites):
        pm.Normal(f"beta_{metabolite}", 
                 mu=llm_priors[metabolite]['mu'],
                 sigma=llm_priors[metabolite]['sigma'])
```

## Examples

The `examples` directory contains implementations showing how to use LLM-generated priors in Bayesian metabolomics models:

1. **Gaussian Process Example**: Demonstrates using LLM priors in a GP regression model for metabolomics data

Run the example with:

```bash
uv run python examples/gp_example.py
```

## Key Components

- `get_llm_priors()`: Categorical mapping approach (qualitative → numerical priors)
- `get_llm_quantitative_priors()`: Direct numerical prior elicitation
- `PriorData`: Data structure for metabolite information
- HMDB integration utilities for enhanced context

## How It Works

### Categorical Mapping Approach
1. **LLM Query**: Ask LLM for qualitative predictions (increase/decrease, small/moderate/large, confidence)
2. **Numerical Mapping**: Convert categorical responses to numerical prior parameters
3. **Prior Generation**: Use mapped values as μ and σ in Normal priors

### Direct Numerical Approach  
1. **LLM Query**: Directly ask LLM for numerical estimates (mean log fold change, uncertainty)
2. **Prior Generation**: Use LLM outputs directly as Normal prior parameters

### Optional HMDB Integration
- Query Human Metabolome Database for additional metabolite context
- Enhance LLM predictions with biochemical pathway information

## License

MIT License
