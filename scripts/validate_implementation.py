"""
Validate the predictive performance implementation to check for potential issues.
"""

import numpy as np
import sys

sys.path.insert(0, ".")

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from comprehensive_regularization_test import load_and_prepare_data
import warnings

warnings.filterwarnings("ignore")


def validate_implementation():
    """Run validation checks on the implementation."""

    print("🔍 IMPLEMENTATION VALIDATION")
    print("=" * 60)

    # Load data
    print("\n1. Loading and examining data...")
    abundance_data, group_labels, metabolite_names, sample_data = (
        load_and_prepare_data()
    )

    X = abundance_data.values
    y = group_labels

    print(f"   Dataset shape: {X.shape}")
    print(
        f"   Class distribution: {np.bincount(y)} (Control={np.sum(y == 0)}, Case={np.sum(y == 1)})"
    )
    print(f"   Class imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}:1")
    print(f"   Feature-to-sample ratio: {X.shape[1] / X.shape[0]:.2f}")

    # Check for obvious data issues
    print(f"   Missing values: {np.sum(np.isnan(X))}")
    print(f"   Infinite values: {np.sum(np.isinf(X))}")
    print(f"   Feature value ranges: min={np.min(X):.3f}, max={np.max(X):.3f}")

    # 2. Baseline checks
    print("\n2. Baseline performance checks...")

    # Dummy classifier (should be around class majority = 84/132 = 0.636)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring="accuracy")
    print(
        f"   Dummy classifier (most frequent): {dummy_scores.mean():.3f} ± {dummy_scores.std():.3f}"
    )

    # Dummy classifier (stratified)
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat_scores = cross_val_score(dummy_strat, X, y, cv=5, scoring="accuracy")
    print(
        f"   Dummy classifier (stratified): {dummy_strat_scores.mean():.3f} ± {dummy_strat_scores.std():.3f}"
    )

    # 3. Simple logistic regression baseline
    print("\n3. Simple logistic regression baseline...")

    # Standardize and fit simple logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Default logistic regression
    lr_default = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr_default, X_scaled, y, cv=5, scoring="accuracy")
    lr_auc_scores = cross_val_score(lr_default, X_scaled, y, cv=5, scoring="roc_auc")

    print(
        f"   Logistic Regression (default): Acc={lr_scores.mean():.3f}±{lr_scores.std():.3f}, AUC={lr_auc_scores.mean():.3f}±{lr_auc_scores.std():.3f}"
    )

    # L2 regularized (similar to Ridge)
    lr_l2 = LogisticRegression(penalty="l2", C=1.0, random_state=42, max_iter=1000)
    lr_l2_scores = cross_val_score(lr_l2, X_scaled, y, cv=5, scoring="accuracy")
    lr_l2_auc_scores = cross_val_score(lr_l2, X_scaled, y, cv=5, scoring="roc_auc")

    print(
        f"   Logistic Regression (L2, C=1): Acc={lr_l2_scores.mean():.3f}±{lr_l2_scores.std():.3f}, AUC={lr_l2_auc_scores.mean():.3f}±{lr_l2_auc_scores.std():.3f}"
    )

    # Stronger regularization
    lr_strong = LogisticRegression(penalty="l2", C=0.1, random_state=42, max_iter=1000)
    lr_strong_scores = cross_val_score(lr_strong, X_scaled, y, cv=5, scoring="accuracy")
    lr_strong_auc_scores = cross_val_score(
        lr_strong, X_scaled, y, cv=5, scoring="roc_auc"
    )

    print(
        f"   Logistic Regression (L2, C=0.1): Acc={lr_strong_scores.mean():.3f}±{lr_strong_scores.std():.3f}, AUC={lr_strong_auc_scores.mean():.3f}±{lr_strong_auc_scores.std():.3f}"
    )

    # 4. Check data separability
    print("\n4. Data separability analysis...")

    # Fit simple model on full data to see coefficients
    lr_full = LogisticRegression(penalty="l2", C=1.0, random_state=42, max_iter=1000)
    lr_full.fit(X_scaled, y)

    # Look at coefficient magnitudes
    coef_magnitudes = np.abs(lr_full.coef_[0])
    print(
        f"   Coefficient statistics: mean={coef_magnitudes.mean():.3f}, max={coef_magnitudes.max():.3f}"
    )
    print(
        f"   Large coefficients (>1.0): {np.sum(coef_magnitudes > 1.0)}/{len(coef_magnitudes)}"
    )

    # Look at top discriminative features
    top_features_idx = np.argsort(coef_magnitudes)[-5:]
    print("   Top 5 discriminative features:")
    for idx in reversed(top_features_idx):
        print(f"     {metabolite_names[idx]}: coef={lr_full.coef_[0][idx]:.3f}")

    # 5. Check if results are too good to be true
    print("\n5. Sanity checks...")

    expected_dummy = max(np.sum(y == 0), np.sum(y == 1)) / len(y)
    improvement_over_dummy = lr_scores.mean() - expected_dummy

    print(f"   Expected dummy accuracy: {expected_dummy:.3f}")
    print(f"   Logistic regression improvement: +{improvement_over_dummy:.3f}")

    if lr_scores.mean() > 0.95:
        print(
            "   ⚠️  WARNING: Accuracy > 0.95 suggests potential data leakage or very easy problem"
        )
    elif lr_scores.mean() > 0.85:
        print(
            "   ⚠️  CAUTION: High accuracy - check if this is reasonable for the problem"
        )
    else:
        print("   ✅ Accuracy seems reasonable")

    if lr_auc_scores.mean() > 0.99:
        print("   ⚠️  WARNING: AUC > 0.99 suggests potential issues")
    elif lr_auc_scores.mean() > 0.95:
        print("   ⚠️  CAUTION: Very high AUC - verify this is expected")
    else:
        print("   ✅ AUC seems reasonable")

    # 6. Check cross-validation stability
    print("\n6. Cross-validation stability...")

    cv_results = []
    for i in range(5):  # Multiple random seeds
        lr_temp = LogisticRegression(penalty="l2", C=1.0, random_state=i, max_iter=1000)
        scores_temp = cross_val_score(lr_temp, X_scaled, y, cv=5, scoring="accuracy")
        cv_results.append(scores_temp.mean())

    cv_stability = np.std(cv_results)
    print(
        f"   CV stability across random seeds: {np.mean(cv_results):.3f} ± {cv_stability:.3f}"
    )

    if cv_stability > 0.05:
        print("   ⚠️  WARNING: High CV instability suggests small dataset issues")
    else:
        print("   ✅ CV results are stable")

    # 7. Dataset-specific considerations
    print("\n7. Dataset considerations...")
    print("   - MTBLS1: Type 2 diabetes vs healthy controls")
    print("   - NMR metabolomics data from urine samples")
    print("   - Diabetes can have strong metabolomic signatures")
    print("   - Small dataset (n=132) with moderate feature count (p=53)")

    if X.shape[0] < 200 and X.shape[1] > 30:
        print("   ⚠️  Small sample size with many features - risk of overfitting")

    return {
        "dummy_accuracy": dummy_scores.mean(),
        "lr_accuracy": lr_scores.mean(),
        "lr_auc": lr_auc_scores.mean(),
        "cv_stability": cv_stability,
        "improvement_over_dummy": improvement_over_dummy,
    }


def check_data_leakage():
    """Additional checks for data leakage."""

    print("\n" + "=" * 60)
    print("🔍 DATA LEAKAGE CHECKS")
    print("=" * 60)

    abundance_data, group_labels, metabolite_names, sample_data = (
        load_and_prepare_data()
    )
    X = abundance_data.values
    y = group_labels

    # Check if any samples are duplicated
    print("\n1. Checking for duplicate samples...")
    unique_samples = np.unique(X, axis=0)
    if len(unique_samples) < X.shape[0]:
        print(
            f"   ⚠️  WARNING: Found {X.shape[0] - len(unique_samples)} duplicate samples!"
        )
    else:
        print("   ✅ No duplicate samples found")

    # Check correlation between features and target
    print("\n2. Feature-target correlation analysis...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    correlations = []
    for i in range(X_scaled.shape[1]):
        corr = np.corrcoef(X_scaled[:, i], y)[0, 1]
        correlations.append(abs(corr))

    max_corr = max(correlations)
    high_corr_features = sum(1 for c in correlations if c > 0.7)

    print(f"   Maximum feature-target correlation: {max_corr:.3f}")
    print(f"   Features with |correlation| > 0.7: {high_corr_features}")

    if max_corr > 0.9:
        print(
            "   ⚠️  WARNING: Very high feature-target correlation suggests potential leakage"
        )
    elif max_corr > 0.8:
        print("   ⚠️  CAUTION: High feature-target correlation")
    else:
        print("   ✅ Feature-target correlations seem reasonable")

    # Show top correlated features
    top_corr_idx = np.argsort(correlations)[-3:]
    print("   Top 3 correlated features:")
    for idx in reversed(top_corr_idx):
        print(
            f"     {metabolite_names[idx]}: r={np.corrcoef(X_scaled[:, idx], y)[0, 1]:.3f}"
        )


if __name__ == "__main__":
    results = validate_implementation()
    check_data_leakage()

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Dummy accuracy: {results['dummy_accuracy']:.3f}")
    print(
        f"Logistic regression: {results['lr_accuracy']:.3f} (AUC: {results['lr_auc']:.3f})"
    )
    print(f"Improvement over dummy: +{results['improvement_over_dummy']:.3f}")
    print(f"CV stability: ±{results['cv_stability']:.3f}")
