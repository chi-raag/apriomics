"""
LLM‑Lasso for Metabolomics (Gemini‑weighted)
===========================================
*Weighted Lasso* feature selection where Google Gemini assigns one importance
score (0‑1) **and** a direction (up / down / ambiguous) to every metabolite.

Why this file matters
---------------------
* **Retrieval‑augmented** prompts (optional FAISS + SBERT over HMDB / KEGG text).
* **γ–λ grid search** ⇒ per‑feature penalty factors like Tibshirani's LLM‑Lasso.
* **Pure‑Python stack**: `google‑generativeai`, `skglm`, `faiss‑cpu`, no R.
* **`--demo` flag** — fabricates a tiny breast‑cancer dataset, yet still calls
  **Gemini** so you can test your API key end‑to‑end.

Quickstart
----------
```bash
export GOOGLE_API_KEY="your‑key"
uv pip install google-generativeai faiss-cpu sentence-transformers skglm \
               scikit-learn pandas numpy tqdm

# synthetic smoke‑test (still hits Gemini)
python llm_lasso_metabolomics.py --demo

# real data
python llm_lasso_metabolomics.py \\
  --design "ER+ breast cancer vs healthy" \\
  --sample_matrix  data/lcms_matrix.feather \\
  --metadata_matrix data/metadata.csv \\
  --outcome_column "outcome_variable_name" \\
  --embeddings data/biomed_faiss/ \\
  --output  results/llm_lasso.tsv
```
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# modelling
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from skglm import GeneralizedLinearEstimator
from skglm.penalties import WeightedL1

# retrieval (optional)
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # demo mode works without
    faiss = SentenceTransformer = None  # type: ignore

# Gemini
import google.generativeai as genai  # type: ignore

# ------------------------------------------------------------------
# dataclasses
# ------------------------------------------------------------------
@dataclass
class RetrievalResult:
    text: str
    score: float
    source: str

@dataclassafds
class LLMMetaboliteScore:
    metabolite: str
    score: float       # importance 0‑1
    direction: str     # up / down / ambiguous
    rationale: str
    provenance: List[str] = field(default_factory=list)

# ------------------------------------------------------------------
# FAISS retriever (optional)
# ------------------------------------------------------------------
class TextRetriever:
    """Simple FAISS + SBERT retriever over literature snippets."""
    def __init__(self, path: Path, model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli"):
        if faiss is None or SentenceTransformer is None:
            raise ImportError("Install faiss-cpu & sentence-transformers for retrieval")
        self.model = SentenceTransformer(model, device="cpu")
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.meta = json.loads((path / "meta.json").read_text())

    def query(self, q: str, k: int = 5) -> List[RetrievalResult]:
        emb = self.model.encode([q])
        scores, ids = self.index.search(emb, k)
        return [RetrievalResult(self.meta[i]["text"], float(scores[0, j]), self.meta[i]["source"]) for j, i in enumerate(ids[0])]

# ------------------------------------------------------------------
# Gemini scorer
# ------------------------------------------------------------------
class GeminiScorer:
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.2):
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("Set GOOGLE_API_KEY env var.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    @staticmethod
    def _prompt(condition: str, snippets: str, mets: List[str]) -> str:
        return (
            "You are a metabolomics expert. For each metabolite, rate how likely it shows any "
            "differential change (increase OR decrease) in the study. Return JSON list of objects: "
            "{name, importance (0‑1), direction (up|down|ambiguous), rationale}.\n\n" +
            f"STUDY: {condition}\n\nEVIDENCE:\n{snippets}\n\nMETABOLITES: {', '.join(mets)}\n")

    def score_batch(self, condition: str, batch_snips: Dict[str, str]) -> List[LLMMetaboliteScore]:
        prompt = self._prompt(condition, "\n".join([f"### {m}\n{s}" for m, s in batch_snips.items()]), list(batch_snips))
        resp = self.model.generate_content(prompt, generation_config={"temperature": self.temperature, "max_output_tokens": 4096})
        try:
            # It's good practice to check if the response has parts and text before trying to parse.
            if not resp.parts:
                error_message = "Gemini response is empty or malformed. "
                if hasattr(resp, 'candidates') and resp.candidates:
                    error_message += f"Candidates: {resp.candidates}. "
                if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback:
                    error_message += f"Prompt Feedback: {resp.prompt_feedback}. "
                if hasattr(resp, 'text'): # Check if text attribute exists
                     error_message += f"Response Text: '{resp.text}'"
                else:
                    error_message += "No text attribute in response."
                raise ValueError(error_message)
            
            response_text = resp.text
            if not response_text:
                 error_message = "Gemini response text is empty. "
                 if hasattr(resp, 'candidates') and resp.candidates:
                    error_message += f"Candidates: {resp.candidates}. "
                 if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback:
                    error_message += f"Prompt Feedback: {resp.prompt_feedback}. "
                 raise ValueError(error_message)

            # Strip markdown fences if present
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json\n")
                response_text = response_text.strip("\n```")
            elif response_text.startswith("```"):
                response_text = response_text.strip("```\n")
                response_text = response_text.strip("\n```")

            parsed = json.loads(response_text)
            
            # Validate that the number of returned scores matches the number of requested metabolites
            if len(parsed) != len(batch_snips):
                raise ValueError(
                    f"Gemini response returned {len(parsed)} metabolite scores, "
                    f"but {len(batch_snips)} were requested for this batch. "
                    f"This may indicate a truncated or incomplete response. Response text: '{resp.text}'"
                )

        except json.JSONDecodeError as e:
            error_message = f"Gemini response not JSON. Original error: {e}. Response Text: '{resp.text}'. "
            if hasattr(resp, 'candidates') and resp.candidates:
                error_message += f"Candidates: {resp.candidates}. "
            if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback:
                error_message += f"Prompt Feedback: {resp.prompt_feedback}. "
            raise ValueError(error_message) from e
        except Exception as e: # Catch other potential errors from accessing resp.text or resp.parts
            error_message = f"Error processing Gemini response. Original error: {e}. "
            if hasattr(resp, 'candidates') and resp.candidates:
                error_message += f"Candidates: {resp.candidates}. "
            if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback:
                error_message += f"Prompt Feedback: {resp.prompt_feedback}. "
            if hasattr(resp, 'text'):
                 error_message += f"Response Text: '{resp.text}'"
            raise ValueError(error_message) from e
        return [LLMMetaboliteScore(d["name"], float(d["importance"]), d.get("direction", "ambiguous"), d.get("rationale", "")) for d in parsed]

# ------------------------------------------------------------------
# Weighted Lasso wrapper
# ------------------------------------------------------------------
class WeightedLasso:
    def __init__(self, alpha: float, weights: np.ndarray):
        self.alpha = alpha
        self.weights = weights
        self.model: Optional[GeneralizedLinearEstimator] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = GeneralizedLinearEstimator(
            penalty=WeightedL1(1.0, self.weights),
        ).fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    @property
    def coef_(self):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.coef_

# ------------------------------------------------------------------
# End‑to‑end pipeline
# ------------------------------------------------------------------
class LLMLassoPipeline:
    def __init__(self, llm: GeminiScorer, retriever: Optional[TextRetriever] = None,
                 gamma_grid: Iterable[float] = (0, 0.25, 0.5, 0.75, 1),
                 alpha_grid: Iterable[float] = (0.01, 0.1, 1, 10),
                 cv: int = 5, seed: int = 42):
        self.llm, self.retriever = llm, retriever
        self.gamma_grid, self.alpha_grid = list(gamma_grid), list(alpha_grid)
        self.cv, self.seed = cv, seed
        self._scores: List[LLMMetaboliteScore] | None = None
        self.best_gamma_: float | None = None
        self.best_alpha_: float | None = None
        self.weights_: np.ndarray | None = None
        self.best_model_: WeightedLasso | None = None

    # ---- scoring helpers ----
    def _evidence(self, mets: List[str], cond: str, k: int = 5) -> Dict[str, str]:
        if self.retriever is None:
            return {m: "" for m in mets}
        return {m: " ".join(r.text for r in self.retriever.query(f"{m} {cond}", k))[:750] for m in mets}

    def _score(self, mets: List[str], cond: str) -> List[LLMMetaboliteScore]:
        batch = max(10, int(np.sqrt(len(mets))))
        out: List[LLMMetaboliteScore] = []
        for i in range(0, len(mets), batch):
            subset = mets[i:i + batch]
            out.extend(self.llm.score_batch(cond, self._evidence(subset, cond)))
        return out

    # ---- weights ----
    @staticmethod
    def _w(scores: List[LLMMetaboliteScore], γ: float) -> np.ndarray:
        imp = np.array([s.score for s in scores])
        return (1 - γ) + γ * (1 - imp)

    # ---- fit ----
    def fit(self, X: pd.DataFrame, y: pd.Series, cond: str):
        mets = list(X.columns)
        if self._scores is None:
            self._scores = self._score(mets, cond)
        Xv, yv = X.values, y.values
        cv_split = KFold(self.cv, shuffle=True, random_state=self.seed)
        scorer = get_scorer("r2")
        best = -np.inf
        for γ in tqdm(self.gamma_grid, desc="γ grid"):
            w = self._w(self._scores, γ)
            for α in self.alpha_grid:
                cv_scores = []
                for tr, te in cv_split.split(Xv):
                    mdl = WeightedLasso(α, w).fit(Xv[tr], yv[tr])
                    cv_scores.append(scorer._score_func(yv[te], mdl.predict(Xv[te])))
                m = float(np.mean(cv_scores))
                if m > best:
                    best, self.best_gamma_, self.best_alpha_ = m, γ, α
        assert self.best_gamma_ is not None and self.best_alpha_ is not None
        self.weights_ = self._w(self._scores, self.best_gamma_)
        self.best_model_ = WeightedLasso(self.best_alpha_, self.weights_).fit(Xv, yv)
        return self

    def coefficients(self) -> pd.DataFrame:
        if self.best_model_ is None:
            raise RuntimeError("Run fit() first")
        return pd.DataFrame({
            "metabolite": [s.metabolite for s in self._scores],
            "coef": self.best_model_.coef_,
            "importance": [s.score for s in self._scores],
            "direction": [s.direction for s in self._scores],
            "weight": self.weights_,
        }).sort_values("coef", key=np.abs, ascending=False)

# ------------------------------------------------------------------
# Synthetic demo (still hits Gemini)
# ------------------------------------------------------------------

def demo():
    print("Running synthetic breast‑cancer demo (Gemini‑scored)…")
    rng = np.random.default_rng(0)
    mets = ["Choline", "Serine", "Glutamine", "Palmitic acid"]
    n = 40
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    X = pd.DataFrame({m: rng.normal(0, 1, n) + y * rng.normal(1.0, 0.5) for m in mets})
    y_ser = pd.Series(y)
    pipe = LLMLassoPipeline(llm=GeminiScorer(), retriever=None, gamma_grid=[0, 0.5, 1], alpha_grid=[0.01, 0.1, 1])
    pipe.fit(X, y_ser, "ER+ breast cancer vs healthy control")
    print(pipe.coefficients())

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    """Command‑line entry point.

    Usage
    -----
    Synthetic check (still queries Gemini):
        python llm_lasso_metabolomics.py --demo

    Real data:
        python llm_lasso_metabolomics.py \\
            --design "ER+ breast cancer vs healthy" \\
            --sample_matrix  data/lcms_matrix.feather \\
            --metadata_matrix data/metadata.csv \\
            --outcome_column "outcome_variable_name" \\
            --output results/llm_lasso.tsv \\
            --embeddings data/biomed_faiss/     # optional
    """
    import argparse

    parser = argparse.ArgumentParser(description="LLM‑weighted Lasso for metabolomics (Gemini)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic breast‑cancer smoke‑test")

    # Real‑run flags (ignored in demo)
    parser.add_argument("--design", help="Condition description, e.g. 'ER+ breast cancer vs healthy'")
    parser.add_argument("--sample_matrix",  help="CSV or Feather file with sample × feature matrix.")
    parser.add_argument("--metadata_matrix", help="CSV or Feather file with sample metadata, including the outcome column.")
    parser.add_argument("--outcome_column", help="Name of the column in the metadata_matrix to use as the outcome (y).")
    parser.add_argument("--output", help="Where to write TSV of coefficients (default: stdout)")
    parser.add_argument("--embeddings", help="Folder containing FAISS index.faiss & meta.json for retrieval")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    # ------- sanity checks for real run -------
    if not args.design or not args.sample_matrix or not args.metadata_matrix or not args.outcome_column:
        parser.error("--design, --sample_matrix, --metadata_matrix, and --outcome_column are required unless --demo is set")

    # Load data
    X_df = pd.read_feather(args.sample_matrix) if args.sample_matrix.endswith(".feather") else pd.read_csv(args.sample_matrix)
    meta_df = pd.read_feather(args.metadata_matrix) if args.metadata_matrix.endswith(".feather") else pd.read_csv(args.metadata_matrix)

    if args.outcome_column not in meta_df.columns:
        raise ValueError(f"Outcome column '{args.outcome_column}' not found in metadata_matrix columns: {meta_df.columns.tolist()}")
    y = meta_df[args.outcome_column]

    # Assuming the first column of X_df might be sample IDs, remove it.
    # If X_df has no sample ID column and all columns are features, this might need adjustment based on user's data.
    # For now, we'll keep the previous logic of dropping the first column if it's not a feature.
    # A more robust way would be to require features to be explicitly named or sample IDs explicitly marked.
    X = X_df
    # Check if the first column is likely an ID column (e.g., non-numeric or named 'ID', 'Sample')
    # This is a heuristic and might need refinement based on common data formats.
    if X_df.iloc[:, 0].dtype == 'object' or X_df.columns[0].lower() in ['id', 'sample', 'sample_id', 'sampleid']:
        X = X_df.drop(columns=[X_df.columns[0]])
    else:
        # If the first column doesn't look like an ID, assume all columns are features.
        # This handles cases where the sample matrix contains only feature data.
        pass

    # Components
    retriever = TextRetriever(Path(args.embeddings)) if args.embeddings else None
    pipeline = LLMLassoPipeline(llm=GeminiScorer(), retriever=retriever)

    # Fit
    pipeline.fit(X, y, cond=args.design)
    coef_df = pipeline.coefficients()

    # Output
    if args.output:
        coef_df.to_csv(args.output, sep="	", index=False)
        print(f"Saved coefficients → {args.output}")
    else:
        print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()
