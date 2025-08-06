"""
Quick test to compare LLM prior strengths and see if stronger priors improve performance.

Tests conservative, moderate, and strong LLM priors on a single train/test split
to quickly assess optimal strength calibration.
"""

import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, ".")

from predictive_performance_benchmark import (
    load_and_prepare_data,
    map_qualitative_to_classification_priors,
    preprocess_features,
    fit_uninformative_bayesian_classifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from apriomics.priors import PriorData, get_llm_qualitative_predictions
import warnings

warnings.filterwarnings("ignore")


def test_llm_prior_strength():
    """Test different LLM prior strengths on a single split."""

    print("Loading data...")
    abundance_data, group_labels, metabolite_names, sample_data = (
        load_and_prepare_data()
    )

    X = abundance_data.values
    y = group_labels

    # Single train/test split for speed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize
    X_train_proc, X_test_proc, _ = preprocess_features(X_train, X_test, "standardize")

    print(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"Train class balance: {np.bincount(y_train)}")
    print(f"Test class balance: {np.bincount(y_test)}")

    # Generate LLM qualitative predictions (cached if available)
    condition = """
Study: Type 2 diabetes mellitus vs healthy control
Context: Type 2 diabetes mellitus is the result of a combination of impaired insulin secretion with reduced insulin sensitivity of target tissues. In this study, NMR-based metabolomic analysis in conjunction with uni- and multivariate statistics was applied to examine the urinary metabolic changes in Human type 2 diabetes mellitus patients compared to the control group.
Sample type: Urine samples analyzed by NMR spectroscopy
Patient population: Unmedicated Type 2 diabetes patients with good dietary control vs healthy controls
Expected changes: Look for metabolites altered in diabetes pathophysiology, particularly those related to glucose metabolism, insulin sensitivity, and urinary excretion patterns.
"""

    priors_data = PriorData(metabolite_names=metabolite_names)

    try:
        # Try to load cached predictions
        cache_path = "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gemini_2_0_flash_*metabolites.pkl"
        import glob
        import pickle

        cached_files = glob.glob(cache_path)

        if cached_files:
            with open(cached_files[0], "rb") as f:
                qualitative_predictions = pickle.load(f)
            print("✅ Using cached LLM predictions")
        else:
            print("🔄 Generating fresh LLM predictions...")
            qualitative_predictions = get_llm_qualitative_predictions(
                priors=priors_data,
                condition=condition,
                use_hmdb_context=False,
                model_name="gemini-2.0-flash",
            )
    except Exception as e:
        print(f"⚠️ Error with LLM predictions: {e}")
        # Create dummy predictions to test the mapping
        qualitative_predictions = {
            name: {
                "prediction": np.random.choice(["increase", "decrease", "unchanged"]),
                "magnitude": "moderate",
                "confidence": np.random.uniform(0.3, 0.9),
                "reasoning": "Test",
            }
            for name in metabolite_names
        }

    # Test different prior strengths
    strengths = ["conservative", "moderate", "strong"]
    results = []

    print("\\nTesting different LLM prior strengths...")

    for strength in strengths:
        print(f"\\nTesting {strength} priors...")

        # Map qualitative to numerical priors
        numerical_priors = map_qualitative_to_classification_priors(
            qualitative_predictions, strength
        )

        # Check prior statistics
        prior_means = [
            numerical_priors[name]["expected_coef"]
            for name in metabolite_names
            if name in numerical_priors
        ]
        prior_sds = [
            numerical_priors[name]["prior_sd"]
            for name in metabolite_names
            if name in numerical_priors
        ]

        print(
            f"  Prior means: {np.mean(np.abs(prior_means)):.3f} ± {np.std(prior_means):.3f}"
        )
        print(f"  Prior SDs: {np.mean(prior_sds):.3f} ± {np.std(prior_sds):.3f}")

        try:
            # Fit LLM Bayesian model with this strength
            y_pred, y_prob = fit_llm_bayesian_classifier_strength(
                X_train_proc, y_train, X_test_proc, metabolite_names, numerical_priors
            )

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            results.append(
                {
                    "strength": strength,
                    "accuracy": acc,
                    "auc": auc,
                    "mean_abs_prior": np.mean(np.abs(prior_means)),
                    "mean_prior_sd": np.mean(prior_sds),
                }
            )

            print(f"  Accuracy: {acc:.3f}, AUC: {auc:.3f}")

        except Exception as e:
            print(f"  ❌ Error with {strength}: {e}")
            results.append(
                {
                    "strength": strength,
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "mean_abs_prior": np.mean(np.abs(prior_means)),
                    "mean_prior_sd": np.mean(prior_sds),
                }
            )

    # Test uninformative baseline for comparison
    print("\\nTesting uninformative baseline...")
    try:
        y_pred_uninf, y_prob_uninf = fit_uninformative_bayesian_classifier(
            X_train_proc, y_train, X_test_proc
        )

        acc_uninf = accuracy_score(y_test, y_pred_uninf)
        auc_uninf = roc_auc_score(y_test, y_prob_uninf)

        results.append(
            {
                "strength": "uninformative",
                "accuracy": acc_uninf,
                "auc": auc_uninf,
                "mean_abs_prior": 0.0,
                "mean_prior_sd": 0.5,
            }
        )

        print(f"  Accuracy: {acc_uninf:.3f}, AUC: {auc_uninf:.3f}")

    except Exception as e:
        print(f"  ❌ Error with uninformative: {e}")

    # Summary
    results_df = pd.DataFrame(results)
    print("\\n" + "=" * 50)
    print("PRIOR STRENGTH COMPARISON")
    print("=" * 50)
    print(results_df.to_string(index=False, float_format="%.3f"))

    if len(results_df.dropna()) > 1:
        best_acc = results_df.loc[results_df["accuracy"].idxmax()]
        best_auc = results_df.loc[results_df["auc"].idxmax()]

        print(f"\\nBest accuracy: {best_acc['strength']} ({best_acc['accuracy']:.3f})")
        print(f"Best AUC: {best_auc['strength']} ({best_auc['auc']:.3f})")

        # Check if stronger priors help
        llm_results = results_df[
            results_df["strength"].isin(["conservative", "moderate", "strong"])
        ].dropna()
        if len(llm_results) > 1:
            corr_strength_acc = np.corrcoef(
                llm_results["mean_abs_prior"], llm_results["accuracy"]
            )[0, 1]
            corr_strength_auc = np.corrcoef(
                llm_results["mean_abs_prior"], llm_results["auc"]
            )[0, 1]

            print(
                f"\\nCorrelation (prior strength vs accuracy): {corr_strength_acc:.3f}"
            )
            print(f"Correlation (prior strength vs AUC): {corr_strength_auc:.3f}")

            if corr_strength_acc > 0.5:
                print("✅ Stronger priors improve accuracy")
            elif corr_strength_acc < -0.5:
                print("⚠️ Weaker priors are better for accuracy")
            else:
                print("🤷 Prior strength has little effect on accuracy")

    return results_df


def fit_llm_bayesian_classifier_strength(
    X_train, y_train, X_test, metabolite_names, numerical_priors
):
    """Simplified LLM classifier that takes pre-computed numerical priors."""
    import pymc as pm

    n_features = X_train.shape[1]

    # Get prior means and sds (aligned with features)
    prior_means = np.array(
        [
            numerical_priors.get(metabolite_names[i], {"expected_coef": 0.0})[
                "expected_coef"
            ]
            for i in range(n_features)
        ]
    )
    prior_sds = np.array(
        [
            numerical_priors.get(metabolite_names[i], {"prior_sd": 1.0})["prior_sd"]
            for i in range(n_features)
        ]
    )

    with pm.Model() as model:
        # LLM-informed priors for coefficients (add regularization for stability)
        beta = pm.Normal(
            "beta", mu=prior_means, sigma=np.minimum(prior_sds, 2.0), shape=n_features
        )
        alpha = pm.Normal("alpha", mu=0, sigma=1.0)

        # Logistic regression
        logit_p = alpha + pm.math.dot(X_train, beta)
        p = pm.math.sigmoid(logit_p)

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

        # Sample posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                500,
                tune=500,
                cores=2,
                progressbar=False,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": False},
            )

    # Get posterior means
    alpha_mean = idata.posterior["alpha"].mean().values
    beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values

    # Predict on test set
    logit_pred = alpha_mean + np.dot(X_test, beta_mean)
    y_pred_prob = 1 / (1 + np.exp(-logit_pred))
    y_pred = (y_pred_prob > 0.5).astype(int)

    return y_pred, y_pred_prob


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set! Will use dummy predictions.")

    results = test_llm_prior_strength()
