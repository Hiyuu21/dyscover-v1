"""
main.py  --  single entry point for the BMDS2114 glycaemic-risk project.

RUN ORDER (do these in sequence the first time)
------------------------------------------------
Step 0  Put the raw NHANES .XPT files in  ./data/raw/   (see nhanes.py -> Config.file_map_*)
Step 1  Build the cleaned cache (slow, run once, or after changing nhanes.py):
            python main.py data --rebuild
Step 2  Pre-processing EDA on the raw features:
            python main.py eda-pre
Step 3  Post-processing EDA (after imputation / encoding / scaling):
            python main.py eda-post
Step 4  Part A baseline (repeated stratified 5x3 CV on 2017-2020 only):
            python main.py baseline-a
Step 5  Part B (train 2017-2020 -> test 2021-2023). Not implemented yet:
            python main.py part-b

Other examples
--------------
    python main.py data                 # load from cache, print shapes (fast sanity check)
    python main.py baseline-a --rebuild # rebuild cache first, then run the baseline
    python main.py all                  # steps 1-4 in order (cache is reused unless --rebuild)

OUTPUTS
-------
    data/cache/*.feather        cleaned data per cycle
    scorecard_A.csv             mean/std per metric per model
    per_fold_scores.csv         raw per-fold scores (use for boxplots / paired comparisons)
    eda_out/pre, eda_out/post   EDA tables and figures

RULES FOR THE TEAM
------------------
    * Never edit the feature list (nhanes.py -> get_xy -> forbidden) without telling everyone,
      otherwise our baselines are not comparable.
    * X_test is NOT touched in Part A or in EDA decisions. It is only used in Part B.
    * Fit preprocessing on train only, then transform test.
"""

import argparse
import sys
from pathlib import Path

from nhanes import NHANES

RAW_DIR = Path("./data/raw")
CACHE_DIR = Path("./data/cache")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def check_raw_files():
    """Fail early with a clear message instead of a stack trace deep inside nhanes.py."""
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.XPT")):
        sys.exit(f"[ERROR] No .XPT files found in {RAW_DIR.resolve()}. See Step 0 in the docstring.")


def load(rebuild=False):
    """Returns X_train, X_test, y_train, y_test, df_1720, df_2123."""
    if rebuild:
        check_raw_files()
    elif not (CACHE_DIR / "df_1720.feather").exists():
        print("[INFO] No cache found, building from raw data.")
        check_raw_files()
        rebuild = True
    return NHANES().load_data_a(force_rebuild=rebuild)


# -----------------------------------------------------------------------------
# steps
# -----------------------------------------------------------------------------
def step_data(args):
    X_train, X_test, y_train, y_test, *_ = load(args.rebuild)
    print(f"train: X={X_train.shape}, prevalence={y_train.mean():.3f}")
    print(f"test : X={X_test.shape}, prevalence={y_test.mean():.3f}")


def step_eda_pre(args):
    from eda import PreprocessEDA            # rename if your file has another name

    X_train, _, y_train, *_ = load(args.rebuild)
    eda = PreprocessEDA(X_train, y_train, out_dir="eda_out/pre")
    eda.run_all()


def step_eda_post(args):
    from eda import PostprocessEDA
    from baseline_partA import build_preprocessor     # same preprocessor as the baseline

    X_train, X_test, y_train, *_ = load(args.rebuild)
    prep = build_preprocessor(X_train, scale_type="standard")
    X_tr = prep.fit_transform(X_train)                # fit on TRAIN only
    X_te = prep.transform(X_test)                     # transform only
    names = list(prep.get_feature_names_out())

    eda = PostprocessEDA(X_tr, names, y_train, X_proc_other=X_te, out_dir="eda_out/post")
    eda.run_all()


def step_baseline_a(args):
    from baseline_partA import run_evaluation         # file name of the Part A script
    run_evaluation(force_rebuild=args.rebuild)


def step_part_b(args):
    # TODO (Part B): fit on full 2017-2020 train, evaluate once on 2021-2023 test.
    raise NotImplementedError("Part B is not designed yet.")


def step_all(args):
    step_data(args)
    step_eda_pre(args)
    step_eda_post(args)
    step_baseline_a(args)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
STEPS = {
    "data": step_data,
    "eda-pre": step_eda_pre,
    "eda-post": step_eda_post,
    "baseline-a": step_baseline_a,
    "part-b": step_part_b,
    "all": step_all,
}


def main():
    parser = argparse.ArgumentParser(
        description="Project runner. See the docstring at the top of this file for the run order.")
    parser.add_argument("step", choices=STEPS.keys(), help="which step to run")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the cleaned cache from raw files (slow)")
    args = parser.parse_args()
    STEPS[args.step](args)


if __name__ == "__main__":
    main()