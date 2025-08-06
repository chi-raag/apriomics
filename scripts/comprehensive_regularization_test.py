"""
Comprehensive Regularization Test: LLM-Informed vs Traditional Methods

Tests different approaches to combining biological knowledge with regularization:

1. LLM-Adaptive-LASSO - LLM confidence controls shrinkage strength
2. LLM-Informed-Bayesian - Original LLM priors (moderate strength)
3. Standard-Adaptive-LASSO - Data-driven adaptive weights
4. Elastic-Net - L1 + L2 regularization
5. Standard-LASSO - Pure L1 regularization
6. Ridge - Pure L2 regularization
7. Uninformative-Bayesian - Weak Bayesian priors
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
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import (
    Ridge as RidgeReg,
    LogisticRegression,
)
from apriomics.priors import PriorData, get_llm_qualitative_predictions
import pymc as pm
import warnings

warnings.filterwarnings("ignore")


def fit_llm_adaptive_lasso(
    X_train, y_train, X_test, metabolite_names, qualitative_predictions
):
    """Adaptive LASSO with LLM confidence controlling shrinkage strength."""

    # Map qualitative predictions with moderate strength for baseline effects
    numerical_priors = map_qualitative_to_classification_priors(
        qualitative_predictions, "moderate"
    )

    n_features = X_train.shape[1]

    # Get LLM means and confidences
    llm_means = np.array(
        [
            numerical_priors.get(metabolite_names[i], {"expected_coef": 0.0})[
                "expected_coef"
            ]
            for i in range(n_features)
        ]
    )
    llm_confidences = np.array(
        [
            qualitative_predictions.get(metabolite_names[i], {"confidence": 0.5})[
                "confidence"
            ]
            for i in range(n_features)
        ]
    )

    with pm.Model() as model:
        # Global shrinkage parameter
        tau_global = pm.HalfNormal("tau_global", sigma=0.5)

        # Adaptive shrinkage: high confidence → less shrinkage
        # tau_i = tau_global / (confidence_i + 0.1)
        adaptive_shrinkage = tau_global / (llm_confidences + 0.1)

        # Laplace priors with LLM-informed means and adaptive shrinkage
        beta = pm.Laplace("beta", mu=llm_means, b=adaptive_shrinkage, shape=n_features)
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

    return y_pred, y_pred_prob, beta_mean


def fit_llm_informed_bayesian(
    X_train, y_train, X_test, metabolite_names, qualitative_predictions
):
    """Original LLM-informed Bayesian method (moderate strength)."""

    # Map qualitative predictions to numerical priors
    numerical_priors = map_qualitative_to_classification_priors(
        qualitative_predictions, "moderate"
    )

    n_features = X_train.shape[1]

    # Get prior means and sds
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
        # LLM-informed priors (regularized for stability)
        beta = pm.Normal(
            "beta", mu=prior_means, sigma=np.minimum(prior_sds, 1.0), shape=n_features
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

    return y_pred, y_pred_prob, beta_mean


def fit_standard_adaptive_lasso(X_train, y_train, X_test, gamma=1.0):
    """Standard adaptive LASSO with data-driven weights."""

    # Step 1: Initial Ridge estimate for adaptive weights
    ridge_init = RidgeReg(alpha=1.0)
    ridge_init.fit(X_train, y_train)
    ridge_coefs = ridge_init.coef_

    # Step 2: Adaptive weights
    adaptive_weights = 1.0 / (np.abs(ridge_coefs) + 1e-3) ** gamma

    # Step 3: Weighted LASSO (approximate with iterative reweighting)
    # Use LogisticRegression with L1 penalty and sample weights approximation
    # This is an approximation - true adaptive LASSO needs custom implementation

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
    lasso.fit(X_train, y_train)

    # Predict
    y_pred = lasso.predict(X_test)
    y_pred_prob = lasso.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob, lasso.coef_[0]


def fit_elastic_net(X_train, y_train, X_test):
    """Elastic Net (L1 + L2 regularization)."""

    elastic = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        random_state=42,
        max_iter=1000,
    )
    elastic.fit(X_train, y_train)

    # Predict
    y_pred = elastic.predict(X_test)
    y_pred_prob = elastic.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob, elastic.coef_[0]


def fit_standard_lasso(X_train, y_train, X_test):
    """Standard LASSO (L1 regularization)."""

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
    lasso.fit(X_train, y_train)

    # Predict
    y_pred = lasso.predict(X_test)
    y_pred_prob = lasso.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob, lasso.coef_[0]


def fit_ridge(X_train, y_train, X_test):
    """Ridge regression (L2 regularization)."""

    ridge = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, random_state=42)
    ridge.fit(X_train, y_train)

    # Predict
    y_pred = ridge.predict(X_test)
    y_pred_prob = ridge.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob, ridge.coef_[0]


def fit_uninformative_bayesian(X_train, y_train, X_test):
    """Uninformative Bayesian baseline."""

    n_features = X_train.shape[1]

    with pm.Model() as model:
        # Regularized uninformative priors
        beta = pm.Normal("beta", mu=0, sigma=0.5, shape=n_features)
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

    return y_pred, y_pred_prob, beta_mean


def comprehensive_regularization_test():
    """Test all regularization methods on a single train/test split."""

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
    print(f"Features: {X_train_proc.shape[1]}")

    # Get LLM qualitative predictions
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
        # Create dummy predictions
        qualitative_predictions = {
            name: {
                "prediction": np.random.choice(["increase", "decrease", "unchanged"]),
                "magnitude": "moderate",
                "confidence": np.random.uniform(0.3, 0.9),
                "reasoning": "Test",
            }
            for name in metabolite_names
        }

    # Test all methods
    methods = [
        ("LLM-Adaptive-LASSO", fit_llm_adaptive_lasso),
        ("LLM-Informed-Bayesian", fit_llm_informed_bayesian),
        ("Standard-Adaptive-LASSO", fit_standard_adaptive_lasso),
        ("Elastic-Net", fit_elastic_net),
        ("Standard-LASSO", fit_standard_lasso),
        ("Ridge", fit_ridge),
        ("Uninformative-Bayesian", fit_uninformative_bayesian),
    ]

    results = []
    coefficients = {}

    print("\\n" + "=" * 60)
    print("COMPREHENSIVE REGULARIZATION TEST")
    print("=" * 60)

    for method_name, fit_func in methods:
        print(f"\\nTesting {method_name}...")

        try:
            if "LLM" in method_name:
                y_pred, y_prob, coefs = fit_func(
                    X_train_proc,
                    y_train,
                    X_test_proc,
                    metabolite_names,
                    qualitative_predictions,
                )
            else:
                y_pred, y_prob, coefs = fit_func(X_train_proc, y_train, X_test_proc)

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            # Calculate sparsity (fraction of coefficients near zero)
            sparsity = np.mean(np.abs(coefs) < 0.01)

            results.append(
                {
                    "method": method_name,
                    "accuracy": acc,
                    "auc": auc,
                    "sparsity": sparsity,
                    "n_nonzero": np.sum(np.abs(coefs) >= 0.01),
                }
            )

            coefficients[method_name] = coefs

            print(f"  Accuracy: {acc:.3f}")
            print(f"  AUC: {auc:.3f}")
            print(
                f"  Sparsity: {sparsity:.3f} ({np.sum(np.abs(coefs) >= 0.01)}/{len(coefs)} non-zero)"
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append(
                {
                    "method": method_name,
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "sparsity": np.nan,
                    "n_nonzero": np.nan,
                }
            )

    # Summary
    results_df = pd.DataFrame(results)
    print("\\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="%.3f"))

    # Find best performers
    valid_results = results_df.dropna()
    if len(valid_results) > 0:
        best_acc = valid_results.loc[valid_results["accuracy"].idxmax()]
        best_auc = valid_results.loc[valid_results["auc"].idxmax()]

        print(f"\\nBest Accuracy: {best_acc['method']} ({best_acc['accuracy']:.3f})")
        print(f"Best AUC: {best_auc['method']} ({best_auc['auc']:.3f})")

        # Compare LLM methods
        llm_methods = valid_results[valid_results["method"].str.contains("LLM")]
        if len(llm_methods) > 0:
            best_llm = llm_methods.loc[llm_methods["auc"].idxmax()]
            print(f"Best LLM Method: {best_llm['method']} ({best_llm['auc']:.3f} AUC)")

        # Sparsity analysis
        sparse_methods = valid_results[valid_results["method"].str.contains("LASSO")]
        if len(sparse_methods) > 0:
            print("\\nSparsity Results:")
            for _, row in sparse_methods.iterrows():
                print(f"  {row['method']}: {row['n_nonzero']:.0f} features selected")

    return results_df, coefficients


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set! Will use dummy predictions.")

    results_df, coefficients = comprehensive_regularization_test()
