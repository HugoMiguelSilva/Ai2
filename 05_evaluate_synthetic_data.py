"""
Evaluate augmented credit risk data.

Outputs:
- figures comparing real vs augmented distributions;
- reports/synthetic_validation_summary.md with statistical, privacy, and training utility metrics.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

REAL_PATH = DATA_DIR / "credit_risk_clean.csv"
SYNTHETIC_ADDED_PATH = DATA_DIR / "credit_risk_synthetic_added_rows.csv"
AUGMENTED_PATH = DATA_DIR / "credit_risk_augmented.csv"
SUMMARY_PATH = REPORTS_DIR / "synthetic_validation_summary.md"

TARGET = "loan_status"
NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not REAL_PATH.exists() or not SYNTHETIC_ADDED_PATH.exists() or not AUGMENTED_PATH.exists():
        raise SystemExit(
            "Missing generated files. Run 04_generate_synthetic_data.py first."
        )
    return (
        pd.read_csv(REAL_PATH),
        pd.read_csv(SYNTHETIC_ADDED_PATH),
        pd.read_csv(AUGMENTED_PATH),
    )


def plot_numeric_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    combined = pd.concat(
        [
            real_df.assign(dataset="Real"),
            synthetic_df.assign(dataset="Synthetic"),
        ],
        ignore_index=True,
    )

    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=combined, x=col, hue="dataset", stat="density", common_norm=False, bins=35)
        plt.title(f"Real vs Synthetic - {col}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"synthetic_numeric_{col}.png", dpi=200)
        plt.close()


def plot_categorical_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    for col in CATEGORICAL_COLUMNS + [TARGET]:
        real_counts = real_df[col].value_counts(normalize=True).rename("Real")
        synthetic_counts = synthetic_df[col].value_counts(normalize=True).rename("Synthetic")
        plot_df = pd.concat([real_counts, synthetic_counts], axis=1).fillna(0).reset_index()
        category_col = plot_df.columns[0]
        plot_df = plot_df.melt(
            id_vars=category_col,
            var_name="dataset",
            value_name="proportion",
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(data=plot_df, x=category_col, y="proportion", hue="dataset")
        plt.title(f"Real vs Synthetic - {col}")
        plt.xlabel(col)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"synthetic_categorical_{col}.png", dpi=200)
        plt.close()


def plot_correlations(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    real_corr = real_df[NUMERIC_COLUMNS + [TARGET]].corr()
    synthetic_corr = synthetic_df[NUMERIC_COLUMNS + [TARGET]].corr()
    corr_delta = (real_corr - synthetic_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Real correlations")
    sns.heatmap(synthetic_corr, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Synthetic correlations")
    sns.heatmap(corr_delta, ax=axes[2], cmap="mako_r")
    axes[2].set_title("Absolute difference")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_correlation_comparison.png", dpi=200)
    plt.close()

    return float(corr_delta.where(~np.eye(corr_delta.shape[0], dtype=bool)).stack().mean())


def numeric_tests(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLUMNS:
        stat, p_value = ks_2samp(real_df[col], synthetic_df[col])
        rows.append(
            {
                "column": col,
                "real_mean": real_df[col].mean(),
                "synthetic_mean": synthetic_df[col].mean(),
                "real_median": real_df[col].median(),
                "synthetic_median": synthetic_df[col].median(),
                "ks_statistic": stat,
                "ks_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def categorical_tests(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_COLUMNS + [TARGET]:
        categories = sorted(set(real_df[col].unique()) | set(synthetic_df[col].unique()))
        observed = pd.DataFrame(
            {
                "real": real_df[col].value_counts().reindex(categories, fill_value=0),
                "synthetic": synthetic_df[col].value_counts().reindex(categories, fill_value=0),
            }
        )
        chi2, p_value, _, _ = chi2_contingency(observed.T)
        rows.append(
            {
                "column": col,
                "chi2_statistic": chi2,
                "chi2_p_value": p_value,
                "real_top": real_df[col].mode().iloc[0],
                "synthetic_top": synthetic_df[col].mode().iloc[0],
            }
        )
    return pd.DataFrame(rows)


def mean_risk_by_group(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    real = real_df.groupby(group_col)[TARGET].mean().rename("real_risk")
    synthetic = synthetic_df.groupby(group_col)[TARGET].mean().rename("synthetic_risk")
    return pd.concat([real, synthetic], axis=1).fillna(0).reset_index()


def exact_duplicate_count(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> int:
    real_rows = set(map(tuple, real_df.astype(str).to_numpy()))
    return int(sum(tuple(row) in real_rows for row in synthetic_df.astype(str).to_numpy()))


def added_rows_rule_checks(real_df: pd.DataFrame, synthetic_added_df: pd.DataFrame) -> pd.DataFrame:
    """Validate basic domain rules for the synthetic rows that were added."""
    real_rows = set(map(tuple, real_df.astype(str).to_numpy()))
    recomputed_ratio = (
        synthetic_added_df["loan_amnt"] / synthetic_added_df["person_income"]
    ).clip(real_df["loan_percent_income"].min(), real_df["loan_percent_income"].max()).round(2)

    checks = [
        ("all_rows_are_target_1", bool((synthetic_added_df[TARGET] == 1).all())),
        ("missing_values", int(synthetic_added_df.isna().sum().sum())),
        ("duplicates_inside_added_rows", int(synthetic_added_df.duplicated().sum())),
        (
            "exact_duplicates_with_real_clean",
            int(sum(tuple(row) in real_rows for row in synthetic_added_df.astype(str).to_numpy())),
        ),
        ("invalid_home_ownership_categories", int((~synthetic_added_df["person_home_ownership"].isin(real_df["person_home_ownership"].unique())).sum())),
        ("invalid_loan_intent_categories", int((~synthetic_added_df["loan_intent"].isin(real_df["loan_intent"].unique())).sum())),
        ("invalid_loan_grade_categories", int((~synthetic_added_df["loan_grade"].isin(real_df["loan_grade"].unique())).sum())),
        ("invalid_default_file_categories", int((~synthetic_added_df["cb_person_default_on_file"].isin(real_df["cb_person_default_on_file"].unique())).sum())),
        ("non_positive_income", int((synthetic_added_df["person_income"] <= 0).sum())),
        ("non_positive_loan_amount", int((synthetic_added_df["loan_amnt"] <= 0).sum())),
        ("non_positive_interest_rate", int((synthetic_added_df["loan_int_rate"] <= 0).sum())),
        (
            "age_outside_real_clean_range",
            int((~synthetic_added_df["person_age"].between(real_df["person_age"].min(), real_df["person_age"].max())).sum()),
        ),
        (
            "loan_percent_income_mismatch_gt_0_01",
            int(((synthetic_added_df["loan_percent_income"] - recomputed_ratio).abs() > 0.011).sum()),
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def added_numeric_fidelity(real_minority_df: pd.DataFrame, synthetic_added_df: pd.DataFrame) -> pd.DataFrame:
    """Compare added synthetic rows with real rows from the same target class."""
    rows = []
    for col in NUMERIC_COLUMNS:
        stat, p_value = ks_2samp(real_minority_df[col], synthetic_added_df[col])
        rows.append(
            {
                "column": col,
                "real_target_1_mean": real_minority_df[col].mean(),
                "added_mean": synthetic_added_df[col].mean(),
                "mean_delta": synthetic_added_df[col].mean() - real_minority_df[col].mean(),
                "real_target_1_median": real_minority_df[col].median(),
                "added_median": synthetic_added_df[col].median(),
                "ks_statistic": stat,
                "ks_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def added_categorical_fidelity(real_minority_df: pd.DataFrame, synthetic_added_df: pd.DataFrame) -> pd.DataFrame:
    """Compare categorical proportions against real rows from the same target class."""
    rows = []
    for col in CATEGORICAL_COLUMNS:
        categories = sorted(set(real_minority_df[col].unique()) | set(synthetic_added_df[col].unique()))
        real_props = real_minority_df[col].value_counts(normalize=True).reindex(categories, fill_value=0)
        added_props = synthetic_added_df[col].value_counts(normalize=True).reindex(categories, fill_value=0)
        total_variation = float((real_props - added_props).abs().sum() / 2)
        rows.append(
            {
                "column": col,
                "real_target_1_top": real_minority_df[col].mode().iloc[0],
                "added_top": synthetic_added_df[col].mode().iloc[0],
                "total_variation_distance": total_variation,
                "max_category_delta": float((real_props - added_props).abs().max()),
            }
        )
    return pd.DataFrame(rows)


def added_category_detail_tables(real_minority_df: pd.DataFrame, synthetic_added_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build detailed real-vs-added proportion tables for each categorical column."""
    tables = {}
    for col in CATEGORICAL_COLUMNS:
        categories = sorted(set(real_minority_df[col].unique()) | set(synthetic_added_df[col].unique()))
        table = pd.DataFrame(
            {
                col: categories,
                "real_target_1_proportion": real_minority_df[col].value_counts(normalize=True).reindex(categories, fill_value=0).values,
                "added_proportion": synthetic_added_df[col].value_counts(normalize=True).reindex(categories, fill_value=0).values,
            }
        )
        table["absolute_delta"] = (
            table["real_target_1_proportion"] - table["added_proportion"]
        ).abs()
        tables[col] = table
    return tables


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float | list[list[int]]]:
    feature_columns = [col for col in train_df.columns if col != TARGET]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def augmented_training_evaluation(
    real_df: pd.DataFrame,
    synthetic_added_df: pd.DataFrame,
) -> tuple[dict, dict]:
    real_train, real_test = train_test_split(
        real_df, test_size=0.2, random_state=42, stratify=real_df[TARGET]
    )
    augmented_train = pd.concat([real_train, synthetic_added_df], ignore_index=True)
    augmented_train = augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)

    real_baseline = evaluate_model(real_train, real_test)
    augmented_result = evaluate_model(augmented_train, real_test)
    return real_baseline, augmented_result


def write_summary(
    real_df: pd.DataFrame,
    synthetic_added_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    added_rule_df: pd.DataFrame,
    added_numeric_df: pd.DataFrame,
    added_categorical_df: pd.DataFrame,
    added_category_tables: dict[str, pd.DataFrame],
    corr_delta: float,
    duplicate_count: int,
    real_baseline: dict,
    augmented_result: dict,
) -> None:
    def markdown_table(df: pd.DataFrame | pd.Series, index: bool = False) -> str:
        if isinstance(df, pd.Series):
            df = df.to_frame("value")
        table = df.reset_index() if index else df.copy()
        table = table.astype(str)
        headers = list(table.columns)
        rows = table.values.tolist()
        header = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, separator] + body)

    risk_tables = {
        "loan_grade": mean_risk_by_group(real_df, augmented_df, "loan_grade"),
        "loan_intent": mean_risk_by_group(real_df, augmented_df, "loan_intent"),
        "person_home_ownership": mean_risk_by_group(real_df, augmented_df, "person_home_ownership"),
    }

    lines = [
        "# Synthetic Data Validation Summary",
        "",
        "## Files",
        "- Clean real data: `data/credit_risk_clean.csv`",
        "- Added synthetic rows: `data/credit_risk_synthetic_added_rows.csv`",
        "- Augmented data: `data/credit_risk_augmented.csv`",
        "- Figures: `figures/`",
        "",
        "## Row Counts",
        f"- Real clean rows: {len(real_df)}",
        f"- Added synthetic rows: {len(synthetic_added_df)}",
        f"- Augmented rows: {len(augmented_df)}",
        "",
        "## Target Distribution",
        markdown_table(real_df[TARGET].value_counts(normalize=True).sort_index().round(4), index=True),
        "",
        markdown_table(augmented_df[TARGET].value_counts(normalize=True).sort_index().round(4), index=True),
        "",
        "## Numeric KS Tests",
        markdown_table(numeric_df.round(4)),
        "",
        "## Categorical Chi-Square Tests",
        markdown_table(categorical_df.round(4)),
        "",
        "## Added Synthetic Rows Fidelity",
        "The added rows should be compared with real clean rows where `loan_status = 1`, because they were generated only for that class.",
        "",
        "### Basic Rule Checks",
        markdown_table(added_rule_df),
        "",
        "### Numeric Fidelity vs Real Target Class",
        markdown_table(added_numeric_df.round(4)),
        "",
        "### Categorical Fidelity vs Real Target Class",
        markdown_table(added_categorical_df.round(4)),
        "",
        "### Categorical Proportion Details",
    ]

    for name, table in added_category_tables.items():
        lines.extend(["", f"#### {name}", markdown_table(table.round(4))])

    lines.extend([
        "## Correlation Preservation",
        f"- Mean absolute off-diagonal correlation difference: {corr_delta:.4f}",
        "",
        "## Risk by Group",
    ])

    for name, table in risk_tables.items():
        lines.extend(["", f"### {name}", markdown_table(table.round(4))])

    lines.extend(
        [
            "",
            "## Privacy Check",
            f"- Exact added synthetic rows copied from real clean data: {duplicate_count}",
            "",
            "## Training Utility Evaluation",
            "### Real train -> Real test",
            markdown_table(pd.DataFrame([real_baseline]).drop(columns=["confusion_matrix"]).round(4)),
            f"- Confusion matrix: `{real_baseline['confusion_matrix']}`",
            "",
            "### Augmented train -> Real test",
            markdown_table(pd.DataFrame([augmented_result]).drop(columns=["confusion_matrix"]).round(4)),
            f"- Confusion matrix: `{augmented_result['confusion_matrix']}`",
            "",
            "## Notes",
            "- The generator is Gaussian Copula, trained only on the minority `loan_status` class.",
            "- The augmented dataset keeps all clean real rows and adds synthetic minority-class rows.",
            "- `loan_percent_income` is recalculated after sampling as `loan_amnt / person_income`.",
            "- Use this dataset when the goal is to reduce class imbalance, not to replace the real dataset.",
        ]
    )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    real_df, synthetic_added_df, augmented_df = load_data()
    print("1. Plotting distributions...")
    plot_numeric_distributions(real_df, augmented_df)
    plot_categorical_distributions(real_df, augmented_df)
    corr_delta = plot_correlations(real_df, augmented_df)

    print("2. Running statistical tests...")
    numeric_df = numeric_tests(real_df, augmented_df)
    categorical_df = categorical_tests(real_df, augmented_df)
    real_minority_df = real_df[real_df[TARGET] == 1].copy()
    added_rule_df = added_rows_rule_checks(real_df, synthetic_added_df)
    added_numeric_df = added_numeric_fidelity(real_minority_df, synthetic_added_df)
    added_categorical_df = added_categorical_fidelity(real_minority_df, synthetic_added_df)
    added_category_tables = added_category_detail_tables(real_minority_df, synthetic_added_df)

    print("3. Running privacy check...")
    duplicate_count = exact_duplicate_count(real_df, synthetic_added_df)

    print("4. Running augmented training evaluation...")
    real_baseline, augmented_result = augmented_training_evaluation(real_df, synthetic_added_df)

    write_summary(
        real_df,
        synthetic_added_df,
        augmented_df,
        numeric_df,
        categorical_df,
        added_rule_df,
        added_numeric_df,
        added_categorical_df,
        added_category_tables,
        corr_delta,
        duplicate_count,
        real_baseline,
        augmented_result,
    )

    print("Evaluation complete")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Exact duplicate rows: {duplicate_count}")
    print(f"Augmented train F1: {augmented_result['f1']:.4f}")
    print(f"Real baseline F1: {real_baseline['f1']:.4f}")


if __name__ == "__main__":
    main()
