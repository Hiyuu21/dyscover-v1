import warnings
warnings.filterwarnings("ignore",message=".*No early stopping will be performed.*",category=UserWarning)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier

from nhanes import NHANES

def build_preprocessor(X, scale_type):
    """
    The bare minimum translation required for scikit-learn to not crash.
    Zero feature selection, zero dimensionality reduction, etc.
    """

    valid_X = X.dropna(axis=1, how='all')

    # Find column types
    numeric_cols = [col for col in valid_X.select_dtypes(include=['number']).columns if X[col].nunique() > 10]
    categorical_cols = [col for col in  valid_X.select_dtypes(include=['object', 'category']).columns if X[col].nunique() <= 10]

    # 1. Numeric: ONLY impute missing values so the math doesn't break
    num_steps = [('imputer', SimpleImputer(strategy='median'))]

    if scale_type == 'standard':
        num_steps.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    # 2. Categorical: ONLY impute and translate to numbers
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )


class TabNet(ClassifierMixin, BaseEstimator):
    """A wrapper to force PyTorch TabNet to conform to Scikit-Learn's strict scoring API."""

    def __init__(self, seed=42, verbose=0):
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        y = np.asarray(y).astype(int).ravel()
        X = np.asarray(X, dtype=np.float32)

        self.model = TabNetClassifier(seed = self.seed, verbose=self.verbose)
        self.model.fit(X, y, **fit_params)

        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

def run_evaluation():
    RANDOM_STATE = 42
    nh = NHANES()

    # Using your new loader that outputs flat matrices
    X_train, X_test, y_train, y_test, df_1720, df_2123 = nh.load_data_a(force_rebuild=False)
    model_1 = MLPClassifier(random_state=RANDOM_STATE, max_iter=5000)
    model_2 = TabNet(seed=RANDOM_STATE, verbose=0)

    scale_1 = "standard"
    scale_2 = "standard"

    my_models = [(model_1, scale_1), (model_2, scale_2)]
    results = []

    # Part A Requirement: Robust Validation Strategy
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    for current_model, current_scale in my_models:
        if current_model is None:
            continue

        model_name = current_model.__class__.__name__
        print(f"\nEvaluating {model_name} on the full feature space...")

        # Build the dynamic pipeline for this specific model's scaling needs
        preprocessor = build_preprocessor(X_train, scale_type=current_scale)
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', current_model)
        ])

        # Execute cross-validation on the entire X_train matrix
        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv_strategy,
            scoring=["roc_auc", "average_precision"],
            n_jobs=max(1, os.cpu_count() - 2)
        )

        roc_auc_mean = cv_scores["test_roc_auc"].mean()
        roc_auc_std = cv_scores["test_roc_auc"].std()
        pr_auc_mean = cv_scores["test_average_precision"].mean()
        pr_auc_std = cv_scores["test_average_precision"].std()

        print(
            f"  Result: ROC-AUC = {roc_auc_mean:.4f} (±{roc_auc_std:.4f}) | PR-AUC = {pr_auc_mean:.4f} (±{pr_auc_std:.4f})")

        results.append({
            "Model": model_name,
            "ROC_AUC_mean": roc_auc_mean,
            "ROC_AUC_std": roc_auc_std,
            "PR_AUC_mean": pr_auc_mean,
            "PR_AUC_std": pr_auc_std
        })

    # Save to CSV for team aggregation
    if results:
        scorecard = pd.DataFrame(results).sort_values(by="ROC_AUC_mean", ascending=False)
        tag = "_".join(r["Model"][:10] for r in results if r["Model"] is not None)
        filename = f"{tag}_scorecard_A.csv"
        scorecard.to_csv(filename, index=False)
        print(f"\nSaved '{filename}'.")


if __name__ == "__main__":
    run_evaluation()