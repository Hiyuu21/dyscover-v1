"""
EDA TODO TEMPLATE  (skeleton only, you write the bodies)

Two classes:
    PreprocessEDA   -> run on the RAW feature frame, BEFORE imputation / encoding / scaling
    PostprocessEDA  -> run on the matrix AFTER your ColumnTransformer, to verify it did what you think

Rules of thumb:
    * Every method should RETURN something (a DataFrame, a number, a figure) so you can save it
      for the report. Printing alone is not reproducible.
    * Fit anything data-dependent on TRAIN only. Never look at X_test to make decisions.
    * Save figures to a folder (e.g. ./eda_out/) instead of only plt.show().
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns                      # optional
# from scipy import stats                    # ks_2samp, chi2_contingency
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# 1. PRE-PROCESSING EDA  (raw data: NaNs, odd codes, wrong dtypes still present)
# =============================================================================
class PreprocessEDA:
    """Questions answered here: What is in the data? What is wrong with it? What must the
    preprocessor handle?"""

    def __init__(self, X: pd.DataFrame, y: pd.Series, out_dir: str = "eda_out/pre"):
        self.X = X
        self.y = y
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        # TODO: decide how you will tell numeric from categorical. Write the rule down once
        #       (e.g. nunique <= 10) and store the two column lists as attributes here.
        self.numeric_cols = []
        self.categorical_cols = []

    # ---- 1.1 Structure ------------------------------------------------------
    def overview(self) -> pd.DataFrame:
        """One row per column: dtype, n_unique, n_missing, % missing, min, max, example value.
        TODO: build it from X.dtypes, X.nunique(), X.isna().sum(), X.describe().T
        Questions: any column with the wrong dtype? any constant column?"""
        raise NotImplementedError

    def duplicates_and_ids(self) -> dict:
        """TODO: count duplicated rows (DataFrame.duplicated). Is there an ID column left over?"""
        raise NotImplementedError

    # ---- 1.2 Missing data ---------------------------------------------------
    def missing_summary(self) -> pd.DataFrame:
        """TODO: % missing per column, sorted descending. Flag columns above a threshold you choose
        (and justify it)."""
        raise NotImplementedError

    def missing_pattern(self):
        """TODO: is missingness random? Ideas:
            - heatmap of X.isna() (rows sorted by number of NaNs)
            - correlation between missing-indicators of different columns
            - compare target rate for rows with vs without a NaN in each column
        Why it matters: decides median/mode imputation vs a 'missing' indicator vs dropping."""
        raise NotImplementedError

    def special_codes_check(self, codes=(7, 9, 77, 99, 777, 999, 7777, 9999)) -> pd.DataFrame:
        """NHANES uses 7/9/77/99... for 'Refused / Don't know'.
        TODO: for each column, count how often these values still appear, and compare against the
        column's real range. Anything left over means the cleaning step missed it."""
        raise NotImplementedError

    # ---- 1.3 Distributions --------------------------------------------------
    def numeric_distributions(self):
        """TODO: histogram (or KDE) per numeric column in a grid. Report skew (Series.skew()).
        Questions: heavy tails? zero-inflation (e.g. alcohol)? impossible values (negative age)?"""
        raise NotImplementedError

    def outliers(self, method: str = "iqr") -> pd.DataFrame:
        """TODO: count outliers per numeric column using the IQR rule (Q1-1.5*IQR, Q3+1.5*IQR).
        Decide: are they errors, or real extreme patients you should keep?"""
        raise NotImplementedError

    def categorical_distributions(self):
        """TODO: value_counts(normalize=True) per categorical column, as bar charts.
        Questions: rare levels (<1-2%)? levels that mean 'unknown'?"""
        raise NotImplementedError

    # ---- 1.4 Target ---------------------------------------------------------
    def target_balance(self) -> pd.Series:
        """TODO: y.value_counts(normalize=True). This prevalence is the no-skill baseline for
        PR-AUC. Write the number down for the report."""
        raise NotImplementedError

    def feature_vs_target(self):
        """TODO: numeric -> boxplot/violin of feature split by y; categorical -> target rate per
        level (groupby(level)[y].mean()). Questions: which features separate the classes at all?"""
        raise NotImplementedError

    # ---- 1.5 Relationships between features --------------------------------
    def correlation(self, method: str = "spearman") -> pd.DataFrame:
        """TODO: correlation matrix of numeric columns + heatmap.
        List pairs with |r| above a threshold you choose (derived-vs-raw columns will show up)."""
        raise NotImplementedError

    def categorical_association(self):
        """TODO (optional): Cramer's V between categorical pairs, built on chi2_contingency."""
        raise NotImplementedError

    # ---- 1.6 Leakage and fairness sanity checks ----------------------------
    def leakage_check(self, suspicious_substrings=("LBX", "DIQ", "target")) -> list:
        """TODO: list columns whose name suggests they define or directly encode the target.
        Also: any single feature with a suspiciously high univariate AUC against y?"""
        raise NotImplementedError

    # ---- 1.7 Cross-cycle comparison (needed for Part B) --------------------
    def cycle_column_comparison(self, X_other: pd.DataFrame) -> dict:
        """TODO: columns only in one frame, columns that are all-NaN in one frame, dtype mismatches.
        Compare with set operations on the column lists."""
        raise NotImplementedError

    # ---- 1.8 Wrap-up --------------------------------------------------------
    def run_all(self):
        """TODO: call the methods above in order, save tables to CSV, figures to PNG, and write a
        short 'decisions' list (e.g. 'impute median', 'drop col X because ...')."""
        raise NotImplementedError


# =============================================================================
# 2. POST-PROCESSING EDA  (after imputation + encoding + scaling)
# =============================================================================
class PostprocessEDA:
    """Questions answered here: Did the preprocessor do what I intended? Is the matrix safe to
    hand to a model? Does train look like test?"""

    def __init__(self, X_proc: np.ndarray, feature_names: list, y: pd.Series,
                 X_proc_other: np.ndarray = None, out_dir: str = "eda_out/post"):
        # X_proc       : output of preprocessor.fit_transform(X_train)
        # X_proc_other : output of preprocessor.transform(X_test)   (transform ONLY, never fit)
        # feature_names: from preprocessor.get_feature_names_out()
        self.X = pd.DataFrame(X_proc, columns=feature_names)
        self.Xo = None if X_proc_other is None else pd.DataFrame(X_proc_other, columns=feature_names)
        self.y = y
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ---- 2.1 Integrity ------------------------------------------------------
    def integrity_checks(self) -> dict:
        """TODO: shape, any NaN or inf left (np.isfinite), dtypes all numeric, duplicate column names.
        These should all pass before any model is fitted."""
        raise NotImplementedError

    def expected_columns_check(self, n_numeric: int, categorical_levels: dict) -> bool:
        """TODO: does the output width equal n_numeric + sum(levels per categorical)?
        Mismatch means a column was dropped or an unseen level appeared."""
        raise NotImplementedError

    # ---- 2.2 Scaling --------------------------------------------------------
    def scaling_check(self) -> pd.DataFrame:
        """TODO: mean and std of each numeric column. After StandardScaler they should be about
        0 and 1 on TRAIN. On test they will differ slightly, which is normal."""
        raise NotImplementedError

    def constant_or_near_constant(self, threshold: float = 0.99) -> list:
        """TODO: one-hot columns where one value covers more than `threshold` of rows carry almost
        no information. List them."""
        raise NotImplementedError

    # ---- 2.3 Collinearity ---------------------------------------------------
    def correlation_after_encoding(self, threshold: float = 0.9) -> pd.DataFrame:
        """TODO: absolute correlation matrix, list pairs above threshold.
        Note: one-hot columns of the same feature are perfectly dependent by design (dummy trap)."""
        raise NotImplementedError

    def vif(self) -> pd.DataFrame:
        """TODO (optional): variance inflation factor per feature. Pay attention to which features
        are highly inflated and why (BMI vs weight vs waist?)."""
        raise NotImplementedError

    # ---- 2.4 Target relationship -------------------------------------------
    def univariate_signal(self) -> pd.DataFrame:
        """TODO: per-feature univariate ROC-AUC against y (roc_auc_score on the column itself, flip
        if < 0.5), or mutual information (mutual_info_classif). Gives a feature-ranking baseline
        before any model."""
        raise NotImplementedError

    # ---- 2.5 Train vs test shift (key for Part B) --------------------------
    def distribution_shift(self) -> pd.DataFrame:
        """TODO: for each feature compare train vs test.
            - continuous: two-sample KS test (scipy.stats.ks_2samp), or standardized mean difference
            - binary/one-hot: difference in proportion
        Sort by effect size, not just p-value (with large n everything is 'significant').
        Questions: which features drift? does it explain a drop in test performance?"""
        raise NotImplementedError

    def adversarial_validation(self) -> float:
        """TODO (optional): train a simple classifier to tell train rows from test rows. AUC near
        0.5 means no shift; near 1.0 means a strong shift. Use cross-validation."""
        raise NotImplementedError

    # ---- 2.6 Wrap-up --------------------------------------------------------
    def run_all(self):
        """TODO: same idea as the pre-processing version. Save outputs, then write down what changed
        your plan for modeling."""
        raise NotImplementedError


# =============================================================================
# 3. HOW TO USE IT  (fill in the order yourself)
# =============================================================================
if __name__ == "__main__":
    # TODO 1: load X_train, X_test, y_train, y_test with your NHANES loader
    # TODO 2: pre = PreprocessEDA(X_train, y_train); call methods; read the output; note decisions
    # TODO 3: build the preprocessor, fit_transform on train, transform on test
    # TODO 4: post = PostprocessEDA(X_train_proc, names, y_train, X_test_proc); call methods
    # TODO 5: write 5-10 bullet 'EDA findings' for the report, each linked to a modeling decision
    pass