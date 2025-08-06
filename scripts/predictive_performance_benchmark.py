"""
Predictive Performance Benchmark: LLM Priors vs Traditional Methods

This script evaluates how well LLM-informed priors improve classification
performance compared to traditional Bayesian and regularization approaches.

Compares:
1. Gemini-Flash (no HMDB context) - LLM-informed priors
2. Uninformative Bayesian - weak priors
3. Horseshoe prior - automatic sparsity
4. Ridge regression - L2 regularization baseline

Uses stratified cross-validation for robust evaluation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import logging
from tqdm import tqdm
from apriomics.priors import PriorData, get_llm_qualitative_predictions

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.WARNING)
logging.getLogger("pytensor").setLevel(logging.WARNING)


def map_qualitative_to_classification_priors(
    qualitative_predictions, prior_strength="moderate"
):
    """
    Map qualitative LLM predictions to numerical priors for logistic regression.

    In logistic regression, coefficients represent log-odds ratios, not log fold changes.
    We want:
    - "increase" → positive coefficient (higher odds of being case class)
    - "decrease" → negative coefficient (lower odds of being case class)
    - "unchanged" → coefficient near zero
    """

    def conservative_classification_prior(prediction, confidence):
        """Conservative priors for classification - small log-odds effects."""
        base_effect = 0.3  # Small log-odds effect

        if prediction == "increase":
            prior_mean = base_effect * confidence
        elif prediction == "decrease":
            prior_mean = -base_effect * confidence
        else:
            prior_mean = 0.0

        # Conservative: wide uncertainty
        prior_sd = 1.0
        return prior_mean, prior_sd

    def moderate_classification_prior(prediction, confidence):
        """Moderate priors for classification - medium log-odds effects."""
        base_effect = 0.7  # Medium log-odds effect

        if prediction == "increase":
            prior_mean = base_effect * confidence
        elif prediction == "decrease":
            prior_mean = -base_effect * confidence
        else:
            prior_mean = 0.0

        # Moderate: confidence affects uncertainty
        if confidence < 0.4:
            prior_sd = 1.5
        elif confidence < 0.7:
            prior_sd = 1.0
        else:
            prior_sd = 0.7

        return prior_mean, prior_sd

    def strong_classification_prior(prediction, confidence):
        """Strong priors for classification - large log-odds effects."""
        base_effect = 1.2  # Large log-odds effect

        if prediction == "increase":
            prior_mean = base_effect * confidence
        elif prediction == "decrease":
            prior_mean = -base_effect * confidence
        else:
            prior_mean = 0.0

        # Strong: tight uncertainty for high confidence
        if confidence < 0.4:
            prior_sd = 1.0
        elif confidence < 0.7:
            prior_sd = 0.7
        else:
            prior_sd = 0.5

        return prior_mean, prior_sd

    numerical_priors = {}

    for metabolite, qual_pred in qualitative_predictions.items():
        prediction = qual_pred["prediction"]
        confidence = qual_pred["confidence"]

        if prior_strength == "conservative":
            expected_coef, prior_sd = conservative_classification_prior(
                prediction, confidence
            )
        elif prior_strength == "moderate":
            expected_coef, prior_sd = moderate_classification_prior(
                prediction, confidence
            )
        elif prior_strength == "strong":
            expected_coef, prior_sd = strong_classification_prior(
                prediction, confidence
            )
        else:
            raise ValueError(f"Unknown prior_strength: {prior_strength}")

        numerical_priors[metabolite] = {
            "prediction": prediction,
            "confidence": confidence,
            "expected_coef": expected_coef,  # Log-odds coefficient
            "prior_sd": prior_sd,
        }

    return numerical_priors


def load_and_prepare_data():
    """Load MTBLS1 data for predictive modeling."""
    # Import the data loading function
    import sys

    sys.path.insert(0, ".")
    from benchmark_prior_recovery import load_mtbls1_data

    abundance_data, group_labels, metabolite_names, sample_data = load_mtbls1_data()

    print(
        f"Dataset: {len(abundance_data)} samples, {len(metabolite_names)} metabolites"
    )
    print(
        f"Class distribution: Control={np.sum(group_labels == 0)}, Case={np.sum(group_labels == 1)}"
    )

    return abundance_data, group_labels, metabolite_names, sample_data


def fit_llm_bayesian_classifier(
    X_train, y_train, X_test, metabolite_names, sample_data
):
    """Fit Bayesian logistic regression with LLM-informed priors."""

    # Generate LLM priors (cached if available)
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

        cached_files = glob.glob(cache_path)

        if cached_files:
            import pickle

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
        # Fallback to uninformative
        qualitative_predictions = {
            name: {
                "prediction": "unchanged",
                "magnitude": "small",
                "confidence": 0.5,
                "reasoning": "Fallback",
            }
            for name in metabolite_names
        }

    # Map to classification-specific numerical priors
    numerical_priors = map_qualitative_to_classification_priors(
        qualitative_predictions, "moderate"
    )

    # Fit Bayesian logistic regression
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
            "beta", mu=prior_means, sigma=np.minimum(prior_sds, 1.0), shape=n_features
        )
        alpha = pm.Normal("alpha", mu=0, sigma=1.0)  # Tighter prior

        # Logistic regression
        logit_p = alpha + pm.math.dot(X_train, beta)
        p = pm.math.sigmoid(logit_p)

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

        # Sample posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
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


def fit_uninformative_bayesian_classifier(X_train, y_train, X_test):
    """Fit Bayesian logistic regression with uninformative priors."""

    n_features = X_train.shape[1]

    with pm.Model() as model:
        # Regularized uninformative priors for stability
        beta = pm.Normal(
            "beta", mu=0, sigma=0.5, shape=n_features
        )  # Strong regularization
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
                1000,
                tune=1000,
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


def fit_horseshoe_classifier(X_train, y_train, X_test):
    """Fit Bayesian logistic regression with horseshoe prior for sparsity."""

    n_features = X_train.shape[1]

    with pm.Model() as model:
        # Horseshoe prior for automatic relevance determination
        tau = pm.HalfCauchy("tau", beta=1)  # Global shrinkage
        lam = pm.HalfCauchy("lam", beta=1, shape=n_features)  # Local shrinkage
        sigma_beta = tau * lam

        beta = pm.Normal("beta", mu=0, sigma=sigma_beta, shape=n_features)
        alpha = pm.Normal("alpha", mu=0, sigma=2)

        # Logistic regression
        logit_p = alpha + pm.math.dot(X_train, beta)
        p = pm.math.sigmoid(logit_p)

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

        # Sample posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                1000,
                tune=1000,
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


def fit_ridge_classifier(X_train, y_train, X_test):
    """Fit Ridge (L2 regularized) logistic regression."""

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit ridge classifier
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Predict
    y_pred = ridge.predict(X_test_scaled)
    # For consistency with other methods, return dummy probabilities
    decision_scores = ridge.decision_function(X_test_scaled)
    y_pred_prob = 1 / (1 + np.exp(-decision_scores))  # Convert to probabilities

    return y_pred, y_pred_prob


def preprocess_features(X_train, X_test, method="standardize"):
    """Preprocess features to avoid numerical issues."""
    if method == "standardize":
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        X_test_proc = scaler.transform(X_test)
        return X_train_proc, X_test_proc, None

    elif method == "pca":
        # Use PCA to reduce dimensionality and remove multicollinearity
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Keep 90% of variance but limit to reasonable number of components
        pca = PCA(n_components=min(20, X_train.shape[1] - 1), random_state=42)
        X_train_proc = pca.fit_transform(X_train_scaled)
        X_test_proc = pca.transform(X_test_scaled)

        return X_train_proc, X_test_proc, (scaler, pca)

    else:
        return X_train, X_test, None


def run_cv_benchmark(
    abundance_data, group_labels, metabolite_names, sample_data, n_folds=5
):
    """Run cross-validation benchmark comparing all methods."""

    X = abundance_data.values
    y = group_labels

    # Always use standardization only - regularization will handle multicollinearity
    try:
        cond_num = np.linalg.cond(X)
        print(
            f"Condition number: {cond_num:.2e}, using standardization + regularization"
        )
    except:
        print(
            "Could not compute condition number, using standardization + regularization"
        )

    use_pca = False  # Never use PCA - preserve metabolite identity

    # Initialize results storage
    results = []
    methods = ["LLM-Bayesian", "Uninformative-Bayesian", "Horseshoe", "Ridge"]

    # Stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    total_folds = n_folds
    pbar = tqdm(
        total=total_folds,
        desc="CV Benchmark",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        pbar.set_description(f"Fold {fold + 1}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features (always, no PCA)
        X_train_proc, X_test_proc, _ = preprocess_features(
            X_train, X_test, "standardize"
        )

        try:
            # LLM-Bayesian
            y_pred_llm, y_prob_llm = fit_llm_bayesian_classifier(
                X_train_proc, y_train, X_test_proc, metabolite_names, sample_data
            )

            acc_llm = accuracy_score(y_test, y_pred_llm)
            auc_llm = roc_auc_score(y_test, y_prob_llm)

            results.append(
                {
                    "fold": fold,
                    "method": "LLM-Bayesian",
                    "accuracy": acc_llm,
                    "auc": auc_llm,
                }
            )

            # Uninformative Bayesian
            y_pred_uninf, y_prob_uninf = fit_uninformative_bayesian_classifier(
                X_train_proc, y_train, X_test_proc
            )

            acc_uninf = accuracy_score(y_test, y_pred_uninf)
            auc_uninf = roc_auc_score(y_test, y_prob_uninf)

            results.append(
                {
                    "fold": fold,
                    "method": "Uninformative-Bayesian",
                    "accuracy": acc_uninf,
                    "auc": auc_uninf,
                }
            )

            # Horseshoe
            y_pred_horse, y_prob_horse = fit_horseshoe_classifier(
                X_train_proc, y_train, X_test_proc
            )

            acc_horse = accuracy_score(y_test, y_pred_horse)
            auc_horse = roc_auc_score(y_test, y_prob_horse)

            results.append(
                {
                    "fold": fold,
                    "method": "Horseshoe",
                    "accuracy": acc_horse,
                    "auc": auc_horse,
                }
            )

            # Ridge (handles its own preprocessing)
            y_pred_ridge, y_prob_ridge = fit_ridge_classifier(X_train, y_train, X_test)

            acc_ridge = accuracy_score(y_test, y_pred_ridge)
            auc_ridge = roc_auc_score(y_test, y_prob_ridge)

            results.append(
                {
                    "fold": fold,
                    "method": "Ridge",
                    "accuracy": acc_ridge,
                    "auc": auc_ridge,
                }
            )

        except Exception as e:
            pbar.write(f"⚠️ Error in fold {fold}: {e}")
            continue

        pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


def analyze_predictive_results(results_df):
    """Analyze and visualize predictive performance results."""

    print("\n" + "=" * 60)
    print("PREDICTIVE PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)

    # Summary statistics
    summary = (
        results_df.groupby("method")
        .agg({"accuracy": ["mean", "std"], "auc": ["mean", "std"]})
        .round(3)
    )

    print("\nAccuracy (Mean ± Std):")
    for method in results_df["method"].unique():
        method_data = results_df[results_df["method"] == method]
        acc_mean = method_data["accuracy"].mean()
        acc_std = method_data["accuracy"].std()
        print(f"  {method:20s}: {acc_mean:.3f} ± {acc_std:.3f}")

    print("\nAUC (Mean ± Std):")
    for method in results_df["method"].unique():
        method_data = results_df[results_df["method"] == method]
        auc_mean = method_data["auc"].mean()
        auc_std = method_data["auc"].std()
        print(f"  {method:20s}: {auc_mean:.3f} ± {auc_std:.3f}")

    # Statistical comparisons
    print(f"\n{'STATISTICAL COMPARISONS':^60}")
    print("-" * 60)

    from scipy.stats import ttest_ind

    llm_acc = results_df[results_df["method"] == "LLM-Bayesian"]["accuracy"]
    llm_auc = results_df[results_df["method"] == "LLM-Bayesian"]["auc"]

    for method in ["Uninformative-Bayesian", "Horseshoe", "Ridge"]:
        method_acc = results_df[results_df["method"] == method]["accuracy"]
        method_auc = results_df[results_df["method"] == method]["auc"]

        if len(method_acc) > 0 and len(llm_acc) > 0:
            acc_stat, acc_p = ttest_ind(llm_acc, method_acc)
            auc_stat, auc_p = ttest_ind(llm_auc, method_auc)

            print(f"\nLLM-Bayesian vs {method}:")
            print(f"  Accuracy improvement: t={acc_stat:.3f}, p={acc_p:.4f}")
            print(f"  AUC improvement: t={auc_stat:.3f}, p={auc_p:.4f}")

    return summary


def create_predictive_visualizations(results_df):
    """Create visualizations for predictive performance results."""

    plt.style.use("default")
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    sns.boxplot(data=results_df, x="method", y="accuracy", ax=axes[0])
    axes[0].set_title("Classification Accuracy by Method")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    # AUC comparison
    sns.boxplot(data=results_df, x="method", y="auc", ax=axes[1])
    axes[1].set_title("AUC by Method")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("AUC")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/predictive_performance_benchmark.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    """Main benchmark execution."""

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY not set! LLM method will use fallback.")

    print("Loading MTBLS1 dataset...")
    abundance_data, group_labels, metabolite_names, sample_data = (
        load_and_prepare_data()
    )

    print("Running cross-validation benchmark...")
    results_df = run_cv_benchmark(
        abundance_data, group_labels, metabolite_names, sample_data, n_folds=5
    )

    if len(results_df) == 0:
        print("❌ No results generated! All folds failed.")
        return None

    print("Analyzing results...")
    summary = analyze_predictive_results(results_df)

    print("Creating visualizations...")
    create_predictive_visualizations(results_df)

    # Save results
    output_dir = "/Users/chiraag/Projects/gwu/lab/apriomics/output"
    results_df.to_csv(f"{output_dir}/predictive_performance_results.csv", index=False)
    summary.to_csv(f"{output_dir}/predictive_performance_summary.csv")

    print(f"\n📊 Results saved to {output_dir}/")
    print("Predictive performance benchmark complete!")

    return results_df, summary


if __name__ == "__main__":
    result = main()
    if result is not None:
        results_df, summary = result
