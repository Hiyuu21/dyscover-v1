"""
Two classes:
    PreprocessEDA   -> run on the RAW feature frame, BEFORE imputation / encoding / scaling
    PostprocessEDA  -> run on the matrix AFTER your ColumnTransformer, to verify it did what you think

Rules of thumb:
    * Every method should RETURN something (a DataFrame, a number, a figure) so you can save it
      for the report. Printing alone is not reproducible.
    * Fit anything data-dependent on TRAIN only. Never look at X_test to make decisions.
    * Save figures to a folder (e.g. ./eda_out/) instead of only plt.show().
"""
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from nhanes import NHANES

# =============================================================================
# 1. PRE-PROCESSING EDA
# =============================================================================
class PreprocessEDA:
    """Questions answered here: What is in the data? What is wrong with it? What must the preprocessor handle?"""

    def __init__(self, X: pd.DataFrame, y: pd.Series, out_dir: str = "eda_out/pre"):
        self.X = X
        self.y = y
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.numeric_cols = [col for col in self.X.columns if pd.api.types.is_numeric_dtype(self.X[col]) and self.X[col].nunique() > 10]
        self.categorical_cols = [col for col in self.X.columns if self.X[col].nunique() <= 10]

    # ---- 1.1 Structure ------------------------------------------------------
    def overview(self) -> pd.DataFrame:
        """One row per column: dtype, n_unique, n_missing, % missing, min, max, example value."""
        df = pd.DataFrame({
            "dtype":self.X.dtypes,
            "n_unique":self.X.nunique(),
            "n_missing":self.X.isna().sum(),
            "%_missing":(self.X.isna().mean() * 100).round(4),
            "min": self.X.min(numeric_only=True),
            "max": self.X.max(numeric_only=True)
        })

        df['example'] = [self.X[col].dropna().iloc[0] if self.X[col].notna().any() else np.nan for col in self.X.columns]
        df.to_csv(os.path.join(self.out_dir, "overview.csv"))
        return df

    def duplicates_and_ids(self) -> dict:
        n_dupes = self.X.duplicated().sum()
        suspicious_ids = [col for col in self.X.columns if "id" in col.lower() or "seqn" in col.lower()]
        result = {"duplicate_rows": int(n_dupes), "leftover_id_columns": suspicious_ids}
        with open(os.path.join(self.out_dir, "duplicates.txt"), "w") as f:
            f.write(str(result))

        return result

    # ---- 1.2 Missing data ---------------------------------------------------
    def missing_summary(self, threshold = 5.0) -> pd.DataFrame:
        miss = pd.DataFrame({
            "missing_count": self.X.isna().sum(),
            "missing_pct": self.X.isna().mean() * 100
        }).sort_values(by="missing_pct", ascending=False)

        miss = miss[miss["missing_count"] > 0]
        miss['flag_above_threshold'] = miss['missing_pct'] > threshold
        miss.to_csv(os.path.join(self.out_dir, "missing_summary.csv"))
        return miss

    def missing_pattern(self):
        cols_with_nan = [c for c in self.X.columns if self.X[c].isna().any()]
        if not cols_with_nan:
            return pd.DataFrame()

        missing_counts = self.X[cols_with_nan].isna().sum(axis=1)
        sorted_X = self.X[cols_with_nan].loc[missing_counts.sort_values(ascending=False).index]
        plt.figure(figsize=(10, 6))
        sns.heatmap(sorted_X.isna().astype(int), cbar=False, cmap="viridis", yticklabels=False)
        plt.title("Missing Data Pattern (Yellow = Missing)")
        plt.savefig(os.path.join(self.out_dir, "missing_heatmap.png"), dpi=200, bbox_inches="tight")
        plt.close()

        records = []
        for col in cols_with_nan:
            rate_when_missing = self.y[self.X[col].isna()].mean()
            rate_when_present = self.y[self.X[col].notna()].mean()
            records.append({
                "feature": col,
                "target_rate_if_missing": rate_when_missing,
                "target_rate_if_present": rate_when_present,
                "absolute_diff": abs(rate_when_missing - rate_when_present),
        })

        df=pd.DataFrame(records).sort_values(by="absolute_diff", ascending=False)
        df.to_csv(os.path.join(self.out_dir, "missing_informative_pattern.csv"), index=False)
        return df

    def special_codes_check(self, codes=(7, 9, 77, 99, 777, 999, 7777, 9999)) -> pd.DataFrame:
        """NHANES uses 7/9/77/99... for 'Refused / Don't know'."""
        records = []
        for col in self.categorical_cols:
            counts = self.X[col].isin(codes).sum()
            if counts > 0:
                records.append({
                    'feature':col,
                    'special_code_count':counts
                })
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.out_dir, "special_codes.csv"), index=False)
        return df

    # ---- 1.3 Distributions --------------------------------------------------
    def numeric_distributions(self):
        skewness = self.X[self.numeric_cols].skew()
        skewness.to_csv(os.path.join(self.out_dir, "numeric_skewness.csv"))

        cols = 3
        rows = math.ceil(len(self.numeric_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(self.numeric_cols):
            sns.histplot(self.X[col].dropna(), kde=True, bins=30, color = 'steelblue', ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")

        for j in range(len(self.numeric_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "numeric_distribution.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        return skewness


    def outliers(self, method: str = "iqr") -> pd.DataFrame:
        records = []
        for col in self.numeric_cols:
            s = self.X[col].dropna()
            if len(s) == 0: continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1-1.5*iqr, q3+1.5*iqr
            n_outliers = ((s < lower) | (s > upper)).sum()
            records.append({
                'feature': col,
                'outliers_count': n_outliers,
                'outliers_pct':(n_outliers/len(self.X)) * 100
            })
        df = pd.DataFrame(records).sort_values(by='outliers_pct', ascending=False)
        df.to_csv(os.path.join(self.out_dir, "outliers.csv"), index=False)
        return df


    def categorical_distributions(self):
        cols = 3
        rows = math.ceil(len(self.categorical_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(self.categorical_cols):
            data = self.X[col].value_counts(dropna=False, normalize=True)
            sns.barplot(x = data.index.astype(str), y = data.values,ax=axes[i],hue=data.index.astype(str), legend=False)
            axes[i].set_title(f"Proportions of {col}")
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(self.categorical_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "categorical_distribution.png"), bbox_inches='tight')
        plt.close()

    # ---- 1.4 Target ---------------------------------------------------------
    def target_balance(self) -> pd.Series:
        balance = self.y.value_counts(normalize=True)
        balance.to_csv(os.path.join(self.out_dir, "target_balance.csv"))
        return balance

    def feature_vs_target(self):
        cols = 3
        rows = math.ceil(len(self.numeric_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(self.numeric_cols):
            sns.boxplot(x = self.y.astype(str), y = self.X[col], hue = self.y.astype(str), legend=False, ax=axes[i])
            axes[i].set_title(f"{col} split by target")

        for j in range(len(self.numeric_cols), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"target_vs_numerical.png"), bbox_inches='tight')
        plt.close()

        records = []
        for col in self.categorical_cols:
            rates = self.y.groupby(self.X[col].fillna(-1)).mean()  # -1 handles missing categories
            for level, rate in rates.items():
                records.append({"feature": col, "level": level, "target_rate": rate})
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.out_dir, "target_vs_categorical.csv"), index=False)
        return df

    # ---- 1.5 Relationships between features --------------------------------
    def correlation(self, method: str = "spearman", threshold = 0.5) -> pd.DataFrame:
        """Spearman correlation  heatmap and list of highly correlated pairs."""
        corr_matrix = self.X[self.numeric_cols].corr(method=method)

        plt.figure(figsize=(12, 12))
        sns.heatmap(corr_matrix, cmap="coolwarm", center = 0, annot = True, vmin = -1, vmax = 1)
        plt.title(f"{method} correlation heatmap")
        plt.savefig(os.path.join(self.out_dir, "correlation_heatmap.png"), bbox_inches='tight')
        plt.close()

        corr_unstacked = corr_matrix.unstack().reset_index()
        corr_unstacked.columns = ['feature_1', 'feature_2', 'spearman_rho']
        corr_unstacked = corr_unstacked[corr_unstacked['feature_1'] < corr_unstacked['feature_2']]
        high_corr = corr_unstacked[corr_unstacked['spearman_rho'].abs() > threshold]

        high_corr.sort_values(by='spearman_rho', key=abs, ascending=False).to_csv(
            os.path.join(self.out_dir, "high_correlations.csv"), index=False
        )

        return high_corr

    # ---- 1.6 Leakage and fairness sanity checks ----------------------------
    def leakage_check(self, suspicious_substrings=("LBX", "DIQ", "target", "seqn")) -> list:
        """Flags columns that might secretly hold the answer key."""
        leaks = [col for col in self.X.columns if any(sub.lower() in col.lower() for sub in suspicious_substrings)]
        with open(os.path.join(self.out_dir, "leakage_warnings.txt"), "w") as f:
            f.write(str(leaks))
        return leaks

    # ---- 1.7 Cross-cycle comparison (needed for Part B) --------------------
    def cycle_column_comparison(self, X_other: pd.DataFrame) -> dict:
        """Ensures the 2017-2020 training set features match the 2021-2023 testing set features."""
        train_cols = set(self.X.columns)
        test_cols = set(X_other.columns)
        result = {
            "in_train_not_test": list(train_cols - test_cols),
            "in_test_not_train": list(test_cols - train_cols)
        }
        with open(os.path.join(self.out_dir, "cross_cycle_diff.txt"), "w") as f:
            f.write(str(result))
        return result

    # ---- 1.8 Wrap-up --------------------------------------------------------
    def run_all(self, X_other: pd.DataFrame):
        print("Running Pre-processing EDA...")
        self.overview()
        self.duplicates_and_ids()
        self.missing_summary()
        self.missing_pattern()
        self.special_codes_check()
        self.numeric_distributions()
        self.outliers()
        self.categorical_distributions()
        self.target_balance()
        self.feature_vs_target()
        self.correlation()
        self.leakage_check()
        self.cycle_column_comparison(X_other)
        print(f"Done. Check dir: {self.out_dir}")


# =============================================================================
# 2. POST-PROCESSING EDA  (after imputation + encoding + scaling)
# =============================================================================
class PostprocessEDA:
    def __init__(self, X_proc: np.ndarray, feature_names: list, y: pd.Series,
                 X_proc_other: np.ndarray = None, out_dir: str = "eda_out/post"):
        # X_proc       : output of preprocessor.fit_transform(X_train)
        # X_proc_other : output of preprocessor.transform(X_test)   (transform ONLY, never fit)
        # feature_names: from preprocessor.get_feature_names_out()
        self.X = pd.DataFrame(X_proc, columns=feature_names)
        self.Xo = None if X_proc_other is None else pd.DataFrame(X_proc_other, columns=feature_names)
        self.y = y.reset_index(drop=True)
        self.out_dir = out_dir
        self.numeric_cols = [col for col in self.X.columns if pd.api.types.is_numeric_dtype(self.X[col]) and self.X[col].nunique() > 10]
        self.categorical_cols = [col for col in self.X.columns if self.X[col].nunique() <= 10]
        os.makedirs(out_dir, exist_ok=True)

    # ---- 2.1 Integrity ------------------------------------------------------
    def integrity_checks(self) -> dict:
        """Verifies shape, NaNs, infinities, and datatypes."""
        result = {
            "shape_train": self.X.shape,
            "has_nan":bool(self.X.isna().any().any()),
            "has_inf":bool(np.isinf(self.X.to_numpy()).any()),
            "all_numeric":all(pd.api.types.is_numeric_dtype(dtype) for dtype in self.X.dtypes),
            "duplicate_columns":list(self.X.columns[self.X.columns.duplicated()])
        }
        with open(os.path.join(self.out_dir, "integrity.txt"), "w") as f:
            f.write(str(result))
        return result

    def expected_columns_check(self, n_numeric: int, categorical_levels: dict) -> bool:
        expected_width = n_numeric + sum(categorical_levels.values())
        actual_width = self.X.shape[1]
        passed = (expected_width == actual_width)

        with open(os.path.join(self.out_dir, "expected_columns.txt"), "w") as f:
            f.write(f"Expected width: {expected_width}\nActual width: {actual_width}\nStatus:{passed}")
        return passed

    # ---- 2.2 Scaling --------------------------------------------------------
    def scaling_check(self) -> pd.DataFrame:
        scaled_cols = [c for c in self.X.columns if self.X[c].nunique() > 2]

        stats = pd.DataFrame({
            'mean':self.X[scaled_cols].mean().round(4),
            'std':self.X[scaled_cols].std().round(4),
        })
        stats.to_csv(os.path.join(self.out_dir, "scaling_verification.txt"))
        return stats

    def constant_or_near_constant(self, threshold: float = 0.99) -> list:
        """Flags one-hot columns that offer zero variance"""
        near_constant = []
        for col in self.X.columns:
            top_share = self.X[col].value_counts(normalize=True).iloc[0]
            if top_share >= threshold:
                near_constant.append({'feature':col, 'top_value_share':top_share})

        df=pd.DataFrame(near_constant)
        if not df.empty:
            df.to_csv(os.path.join(self.out_dir, "near_constant_features.txt"))
        return near_constant

    # ---- 2.3 Collinearity ---------------------------------------------------
    def _parent(self, name:str) -> str:
        if name.startswith("cat__"):return name.rsplit("_", 1)[0]
        return name

    def correlation_after_encoding(self, threshold: float = 0.9) -> pd.DataFrame:
        corr_matrix = self.X.corr().abs()
        corr_unstacked = corr_matrix.unstack().reset_index()
        corr_unstacked.columns = ['feature_1','feature_2','abs_r']

        # Remove self-correlations and duplicates
        corr_unstacked = corr_unstacked[corr_unstacked['feature_1'] < corr_unstacked['feature_2']]
        same_source = corr_unstacked['feature_1'].map(self._parent) == corr_unstacked['feature_2'].map(self._parent)
        corr_unstacked = corr_unstacked[~same_source]
        high_corr = corr_unstacked[corr_unstacked['abs_r'] >= threshold].sort_values('abs_r', ascending=False)

        high_corr.to_csv(os.path.join(self.out_dir, "post_encoding_collinearity.csv"), index=False)
        return high_corr

    def vif(self) -> pd.DataFrame:
        """Variance Inflation Factor to detect severe multicollinearity."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant

            X_num = add_constant(self.X[self.numeric_cols])
            vif_data = pd.DataFrame({
                "feature": self.numeric_cols,
                "VIF": [variance_inflation_factor(X_num.values, i+1) for i in range(len(self.numeric_cols))]
            }).sort_values("VIF", ascending=False)

            vif_data.to_csv(os.path.join(self.out_dir, "vif_scores.csv"), index=False)
            return vif_data
        except ImportError:
            print("statsmodels not installed; skipping VIF.")
            return pd.DataFrame()

    # ---- 2.4 Target relationship -------------------------------------------
    def univariate_signal(self) -> pd.DataFrame:
        """The baseline predictive power of every individual feature."""
        records = []
        for col in self.X.columns:
            valid_idx = self.X[col].notna()
            if valid_idx.sum() == 0: continue

            auc = roc_auc_score(self.y[valid_idx], self.X.loc[valid_idx, col])
            if auc < 0.5: auc = 1.0- auc

            records.append({'feature':col, 'univariate_roc_auc':auc})
        df = pd.DataFrame(records).sort_values("univariate_roc_auc", ascending=False)
        df.to_csv(os.path.join(self.out_dir, "univariate_signals.csv"), index=False)
        return df

    # ---- 2.5 Train vs test shift (key for Part B) --------------------------
    def distribution_shift(self) -> pd.DataFrame:
        """Identifies features that drastically change between the 17-20 and 21-23 cycles."""
        if self.Xo is None:
            return pd.DataFrame()

        records = []
        for col in self.X.columns:
            train_vals = self.X[col].dropna()
            test_vals = self.Xo[col].dropna()

            # Continuous features: Kolmogorov-Smirnov test for distribution shape drift
            if train_vals.nunique() > 2:
                stat, pval = ks_2samp(train_vals, test_vals)
                metric_name = "ks_statistic"
            # Binary/One-Hot features: Absolute difference in proportions
            else:
                stat = abs(train_vals.mean() - test_vals.mean())
                pval = np.nan
                metric_name = "abs_prop_diff"

            records.append({
                "feature": col,
                "shift_metric": metric_name,
                "effect_size": stat,
                "p_value": pval
            })

        df = pd.DataFrame(records).sort_values("effect_size", ascending=False)
        df.to_csv(os.path.join(self.out_dir, "distribution_shift.csv"), index=False)
        return df

    def adversarial_validation(self) -> float:
        """Trains a model to distinguish Train (17-20) from Test (21-23).
        AUC ~0.50 means no drift. AUC >0.70 means massive data drift."""
        if self.Xo is None:
            return np.nan

        # Label train as 0 and test as 1
        X_adv = pd.concat([self.X, self.Xo], axis=0, ignore_index=True)
        y_adv = np.array([0] * len(self.X) + [1] * len(self.Xo))

        # HistGradientBoosting is fast and handles complex interactions
        clf = HistGradientBoostingClassifier(random_state=42, max_iter=50)
        scores = cross_val_score(clf, X_adv, y_adv, cv=3, scoring='roc_auc', n_jobs=-1)

        mean_auc = scores.mean()
        with open(os.path.join(self.out_dir, "adversarial_validation.txt"), "w") as f:
            f.write(
                f"Adversarial ROC-AUC: {mean_auc:.4f}\n(0.50 = perfect overlap, 1.00 = completely disjointed cycles)")

        return mean_auc

    # ---- 2.6 Wrap-up --------------------------------------------------------
    def run_all(self):
        print("Running Post-processing EDA (Matrix Validation)...")
        self.integrity_checks()
        self.scaling_check()
        self.constant_or_near_constant()
        self.correlation_after_encoding()
        self.vif()
        self.univariate_signal()
        self.distribution_shift()
        self.adversarial_validation()
        print(f"Done. Check dir: {self.out_dir}.")


# =============================================================================
# Run the EDAs
# =============================================================================
if __name__ == "__main__":
    # TODO 1: load X_train, X_test, y_train, y_test with your NHANES loader
    # TODO 2: pre = PreprocessEDA(X_train, y_train); call methods; read the output; note decisions
    # TODO 3: build the preprocessor, fit_transform on train, transform on test
    # TODO 4: post = PostprocessEDA(X_train_proc, names, y_train, X_test_proc); call methods
    # TODO 5: write 5-10 bullet 'EDA findings' for the report, each linked to a modeling decision

    from baseline_partA import build_preprocessor

    nh = NHANES()
    X_train, X_test, y_train, y_test, _, _ = nh.load_data_a()

    pre = PreprocessEDA(X_train, y_train)
    pre.run_all(X_test)

    prep = build_preprocessor(X_train, scale_type='standard')
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)
    names = list(prep.get_feature_names_out())

    post = PostprocessEDA(X_tr, names, y_train, X_proc_other = X_te)
    post.run_all()