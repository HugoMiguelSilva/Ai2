"""
Generate synthetic rows to augment the credit risk data with SDV.

Augmentation strategy:
- clean the original dataset using the same missing-value policy as model training;
- identify the minority class in the critical feature loan_status;
- train a GaussianCopulaSynthesizer on that class only;
- sample enough rows to match the majority class;
- post-process simple domain rules;
- save the clean data, added synthetic rows, augmented data, and fitted generator.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
except ImportError as exc:
    raise SystemExit(
        "SDV is required. Install it with: python -m pip install sdv"
    ) from exc

from scipy.stats import ks_2samp


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "credit_risk_dataset.csv"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


CLEAN_PATH = DATA_DIR / "credit_risk_clean.csv"
SYNTHETIC_ADDED_PATH = DATA_DIR / "credit_risk_synthetic_added_rows.csv"
AUGMENTED_PATH = DATA_DIR / "credit_risk_augmented.csv"
LEGACY_SYNTHETIC_PATH = DATA_DIR / "credit_risk_synthetic.csv"
SCORES_PATH = DATA_DIR / "synthetic_candidate_scores.csv"

TARGET = "loan_status"
BALANCE_RATIO = 1.0
SYNTHESIZER_CANDIDATES = ("gaussian_copula", "tvae", "ctgan")
NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
INTEGER_COLUMNS = ["person_age", "cb_person_cred_hist_length"]
CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_and_clean_dataset() -> pd.DataFrame:
    """Load the real data and apply conservative project-level cleaning."""
    df = pd.read_csv(DATASET_PATH)
    original_rows = len(df)

    df = df.dropna().copy()
    df = df[df["person_age"].between(18, 100)]
    df = df[df["person_income"] > 0]
    df = df[df["person_emp_length"].between(0, 80)]
    df = df[df["loan_amnt"] > 0]
    df = df[df["loan_int_rate"] > 0]
    df = df[df["cb_person_cred_hist_length"] >= 0]
    df[TARGET] = df[TARGET].astype(int)

    print("1. Dataset cleaning")
    print(f"   Original rows: {original_rows}")
    print(f"   Clean rows:    {len(df)}")
    print(f"   Removed rows:  {original_rows - len(df)}")

    return df.reset_index(drop=True)


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Create SDV metadata with explicit column types."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    for col in NUMERIC_COLUMNS:
        metadata.update_column(col, sdtype="numerical")
    for col in CATEGORICAL_COLUMNS:
        metadata.update_column(col, sdtype="categorical")

    return metadata


def make_synthesizer(name: str, metadata: SingleTableMetadata):
    """Create a synthesizer candidate with conservative training settings."""
    if name == "gaussian_copula":
        return GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
        )
    if name == "tvae":
        return TVAESynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=300,
            verbose=False,
        )
    if name == "ctgan":
        return CTGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=300,
            verbose=False,
        )
    raise ValueError(f"Unknown synthesizer: {name}")


def total_variation_distance(real_series: pd.Series, synthetic_series: pd.Series) -> float:
    """Compare categorical proportions. Lower is better."""
    categories = sorted(set(real_series.dropna().unique()) | set(synthetic_series.dropna().unique()))
    real_props = real_series.value_counts(normalize=True).reindex(categories, fill_value=0)
    synthetic_props = synthetic_series.value_counts(normalize=True).reindex(categories, fill_value=0)
    return float((real_props - synthetic_props).abs().sum() / 2)


def score_candidate(real_class_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float]:
    """Score candidate fidelity against the real minority class. Lower is better."""
    numeric_ks = [
        ks_2samp(real_class_df[col], synthetic_df[col]).statistic
        for col in NUMERIC_COLUMNS
    ]
    categorical_tv = [
        total_variation_distance(real_class_df[col], synthetic_df[col])
        for col in CATEGORICAL_COLUMNS
    ]
    real_corr = real_class_df[NUMERIC_COLUMNS].corr()
    synthetic_corr = synthetic_df[NUMERIC_COLUMNS].corr()
    corr_delta = (real_corr - synthetic_corr).abs()
    mean_corr_delta = float(
        corr_delta.where(~pd.DataFrame(
            [[i == j for j in range(len(NUMERIC_COLUMNS))] for i in range(len(NUMERIC_COLUMNS))],
            index=NUMERIC_COLUMNS,
            columns=NUMERIC_COLUMNS,
        )).stack().mean()
    )

    mean_ks = float(pd.Series(numeric_ks).mean())
    mean_tv = float(pd.Series(categorical_tv).mean())
    weighted_score = (0.60 * mean_ks) + (0.25 * mean_tv) + (0.15 * mean_corr_delta)
    return {
        "mean_numeric_ks": mean_ks,
        "mean_categorical_tv": mean_tv,
        "mean_numeric_corr_delta": mean_corr_delta,
        "weighted_score": weighted_score,
    }


def quantile_calibrate_numeric_columns(
    synthetic_df: pd.DataFrame,
    real_class_df: pd.DataFrame,
    full_real_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match synthetic numeric marginal distributions to the real minority class.

    This keeps each synthetic row's rank order for a column but replaces values with
    empirical quantiles from the real class. It improves fidelity for skewed columns
    such as employment length while avoiding direct row copying.
    """
    calibrated = synthetic_df.copy()
    columns_to_calibrate = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "cb_person_cred_hist_length",
    ]

    for col in columns_to_calibrate:
        real_values = np.sort(real_class_df[col].to_numpy(dtype=float))
        synthetic_values = calibrated[col].to_numpy(dtype=float)
        ranks = pd.Series(synthetic_values).rank(method="first").to_numpy()
        quantile_positions = (ranks - 0.5) / len(synthetic_values)
        mapped_values = np.quantile(real_values, quantile_positions)
        calibrated[col] = mapped_values

    calibrated["loan_percent_income"] = (
        calibrated["loan_amnt"] / calibrated["person_income"]
    ).clip(
        full_real_df["loan_percent_income"].min(),
        full_real_df["loan_percent_income"].max(),
    )

    for col in INTEGER_COLUMNS:
        calibrated[col] = calibrated[col].round().astype(int)

    for col in [c for c in NUMERIC_COLUMNS if c not in INTEGER_COLUMNS]:
        calibrated[col] = calibrated[col].round(2)

    return calibrated[synthetic_df.columns]


def train_and_sample_minority_class(df: pd.DataFrame) -> pd.DataFrame:
    """Train one synthesizer for the minority target class and sample added rows."""
    feature_columns = [col for col in df.columns if col != TARGET]
    class_counts = df[TARGET].value_counts().sort_index()
    majority_count = int(class_counts.max())
    minority_status = int(class_counts.idxmin())
    minority_count = int(class_counts.min())
    target_count = int(round(majority_count * BALANCE_RATIO))
    rows_to_add = max(0, target_count - minority_count)

    print("\n2. Targeted synthetic augmentation")
    print(class_counts.to_string())
    print(f"   Critical feature: {TARGET}")
    print(f"   Minority class:   {minority_status}")
    print(f"   Rows to add:      {rows_to_add}")

    if rows_to_add == 0:
        print("   No rows needed: the critical feature is already balanced enough.")
        return pd.DataFrame(columns=df.columns)

    class_df = df[df[TARGET] == minority_status].reset_index(drop=True)
    class_features = class_df[feature_columns].reset_index(drop=True)
    metadata = build_metadata(class_features)

    candidate_rows = []
    candidate_scores = []

    for candidate_name in SYNTHESIZER_CANDIDATES:
        print(f"   Training {candidate_name} for loan_status={minority_status}...")
        synthesizer = make_synthesizer(candidate_name, metadata)
        synthesizer.fit(class_features)

        sampled = synthesizer.sample(num_rows=rows_to_add).reset_index(drop=True)
        sampled[TARGET] = minority_status
        sampled = post_process(sampled[df.columns], df)
        sampled = quantile_calibrate_numeric_columns(sampled, class_df, df)
        sampled = post_process(sampled[df.columns], df)

        scores = score_candidate(class_df, sampled)
        scores["candidate"] = candidate_name
        candidate_scores.append(scores)
        candidate_rows.append((candidate_name, sampled, synthesizer))

        print(
            "      score={weighted_score:.4f} "
            "ks={mean_numeric_ks:.4f} "
            "cat_tv={mean_categorical_tv:.4f} "
            "corr={mean_numeric_corr_delta:.4f}".format(**scores)
        )

    scores_df = pd.DataFrame(candidate_scores).sort_values("weighted_score")
    scores_df.to_csv(SCORES_PATH, index=False)
    best_name = scores_df.iloc[0]["candidate"]
    best_sampled = next(rows for name, rows, _ in candidate_rows if name == best_name)
    best_synthesizer = next(model for name, _, model in candidate_rows if name == best_name)

    model_path = MODELS_DIR / f"synthetic_generator_loan_status_{minority_status}_{best_name}.pkl"
    best_synthesizer.save(filepath=str(model_path))
    print(f"   Best candidate: {best_name}")
    print(f"   Candidate scores saved: {SCORES_PATH}")
    print(f"   Saved best generator: {model_path}")

    return best_sampled[df.columns]


def post_process(synthetic_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Enforce simple domain rules after sampling."""
    synthetic_df = synthetic_df.copy()

    for col in NUMERIC_COLUMNS:
        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors="coerce")
        synthetic_df[col] = synthetic_df[col].clip(real_df[col].min(), real_df[col].max())

    synthetic_df["person_income"] = synthetic_df["person_income"].clip(lower=1)
    synthetic_df["loan_amnt"] = synthetic_df["loan_amnt"].clip(lower=1)
    synthetic_df["loan_int_rate"] = synthetic_df["loan_int_rate"].clip(lower=0.01)
    synthetic_df["person_emp_length"] = synthetic_df["person_emp_length"].clip(lower=0)

    synthetic_df["loan_percent_income"] = (
        synthetic_df["loan_amnt"] / synthetic_df["person_income"]
    ).clip(real_df["loan_percent_income"].min(), real_df["loan_percent_income"].max())

    for col in INTEGER_COLUMNS:
        synthetic_df[col] = synthetic_df[col].round().astype(int)

    float_columns = [col for col in NUMERIC_COLUMNS if col not in INTEGER_COLUMNS]
    for col in float_columns:
        synthetic_df[col] = synthetic_df[col].round(2)

    valid_categories = {col: set(real_df[col].dropna().unique()) for col in CATEGORICAL_COLUMNS}
    for col, allowed in valid_categories.items():
        fallback = real_df[col].mode().iloc[0]
        synthetic_df[col] = synthetic_df[col].where(synthetic_df[col].isin(allowed), fallback)

    synthetic_df[TARGET] = synthetic_df[TARGET].round().clip(0, 1).astype(int)
    return synthetic_df[real_df.columns]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    real_clean = load_and_clean_dataset()
    real_clean.to_csv(CLEAN_PATH, index=False)
    print(f"   Saved clean dataset: {CLEAN_PATH}")

    synthetic_added = train_and_sample_minority_class(real_clean)
    synthetic_added.to_csv(SYNTHETIC_ADDED_PATH, index=False)

    augmented = pd.concat([real_clean, synthetic_added], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    augmented.to_csv(AUGMENTED_PATH, index=False)

    # Keep the older filename as a compatibility alias for notebooks or open tabs.
    augmented.to_csv(LEGACY_SYNTHETIC_PATH, index=False)

    print("\n3. Augmented dataset saved")
    print(f"   Added synthetic rows: {len(synthetic_added)}")
    print(f"   Augmented rows:       {len(augmented)}")
    print(f"   Added rows path:      {SYNTHETIC_ADDED_PATH}")
    print(f"   Augmented path:       {AUGMENTED_PATH}")
    print("\nOriginal clean loan_status distribution:")
    print(real_clean[TARGET].value_counts().sort_index().to_string())
    print("\nAugmented loan_status distribution:")
    print(augmented[TARGET].value_counts().sort_index().to_string())
    print("\nAugmented loan_status proportions:")
    print(augmented[TARGET].value_counts(normalize=True).sort_index().round(4).to_string())


if __name__ == "__main__":
    main()
