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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

""" 
TODO:
 1. robustness across distributions (cycle shift, subgroups)
 2. permutation importance, SHAP, visualisations
 3. confusion matrices, errors by subgroup (cross_val_predict)
 4. choose model by performance, effectiveness, interpretability and adaptability
"""

from nhanes import NHANES

def build_preprocessor(X, scale_type):
    """
    The bare minimum translation required for scikit-learn to not crash.
    Zero feature selection, zero dimensionality reduction, etc.
    """

    valid_X = X.dropna(axis=1, how='all')

    # Find column types
    numeric_cols, categorical_cols = [], []
    for col in valid_X.columns:
        is_numeric = pd.api.types.is_numeric_dtype(valid_X[col])
        if is_numeric and valid_X[col].nunique() > 10:
            numeric_cols.append(col)
        elif valid_X[col].nunique() <= 10:
            categorical_cols.append(col)

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

    print(f"numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, total features: {X.shape[1]}")
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

def run_evaluation(force_rebuild = False):
    RANDOM_STATE = 42
    nh = NHANES()

    # Using your new loader that outputs flat matrices
    X_train, X_test, y_train, y_test, df_1720, df_2123 = nh.load_data_a(force_rebuild=force_rebuild)

    models = {
        MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True):"standard",
        TabNet(seed=RANDOM_STATE, verbose=0):"standard",
        LogisticRegression(random_state=RANDOM_STATE, verbose=0, max_iter=1000):"standard",
        GaussianNB():"standard",
        KNeighborsClassifier():"standard",
        SVC(random_state=RANDOM_STATE, verbose=0):"standard",
        DecisionTreeClassifier(random_state=RANDOM_STATE):"none",
        RandomForestClassifier(random_state=RANDOM_STATE, verbose=0):"none",
    }

    results = []

    # Part A Requirement: Robust Validation Strategy
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    all_folds = []

    for current_model, current_scale in models.items():

        model_name = current_model.__class__.__name__
        print(f"\nEvaluating {model_name} on the full feature space...")

        # Build the dynamic pipeline for this specific model's scaling needs
        preprocessor = build_preprocessor(X_train, scale_type=current_scale)
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', current_model)
        ])

        # Execute cross-validation on the entire X_train matrix
        scoring = ["roc_auc", "average_precision", "precision", "recall", "f1"]

        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=max(1, os.cpu_count() - 2)
        )

        roc_auc_mean = cv_scores["test_roc_auc"].mean()
        roc_auc_std = cv_scores["test_roc_auc"].std()
        pr_auc_mean = cv_scores["test_average_precision"].mean()
        pr_auc_std = cv_scores["test_average_precision"].std()

        print(
            f"--> Result: ROC-AUC = {roc_auc_mean:.4f} (±{roc_auc_std:.4f}) | PR-AUC = {pr_auc_mean:.4f} (±{pr_auc_std:.4f})")

        row = {"Model": model_name}
        for m in scoring:
            vals = cv_scores[f"test_{m}"]
            row[f"{m}_mean"] = vals.mean().round(4)
            row[f"{m}_std"] = vals.std().round(4)
        results.append(row)

        fold_scores = pd.DataFrame({m: cv_scores[f"test_{m}"] for m in scoring})
        fold_scores.insert(0, "Model", model_name)
        all_folds.append(fold_scores)

    # Save to CSV for team aggregation
    if results:
        scorecard = pd.DataFrame(results).sort_values(by="roc_auc_mean", ascending=False)
        scorecard.to_csv("scorecard_A.csv", index=False)
        pd.concat(all_folds, ignore_index=True).to_csv("per_fold_scores.csv", index=False)
        print("\nSaved 'scorecard_A.csv' and 'per_fold_scores.csv'.")


if __name__ == "__main__":
    run_evaluation(force_rebuild=False)