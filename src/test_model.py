from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "retail_cleaned_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "retail_demand_model_bundle.joblib"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(data):

    df = data.copy()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Week_of_Year"] = (
        df["Date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )
    df["Quarter"] = df["Date"].dt.quarter

    df["Is_Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

    # Pricing features
    df["Effective_Price"] = (
        df["Price"]
        *
        (
            1
            -
            df["Discount"] / 100
        )
    )

    df["Competitor_Price_Gap"] = (
        df["Price"]
        -
        df["Competitor Pricing"]
    )

    df["Competitor_Price_Ratio"] = (
        df["Price"]
        /
        df["Competitor Pricing"].replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 72)
    print("SMART RETAIL SUPPLY CHAIN - LOCAL MODEL TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found:\n{MODEL_PATH}"
        )

    print("\n[1/6] Required files found.")


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"[2/6] Dataset loaded: "
        f"{len(data):,} rows"
    )


    # --------------------------------------------------------
    # Load model bundle
    # --------------------------------------------------------

    bundle = joblib.load(
        MODEL_PATH
    )

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    feature_columns = bundle[
        "feature_columns"
    ]

    target_column = bundle[
        "target_column"
    ]

    print(
        f"[3/6] Model loaded: "
        f"{type(model).__name__}"
    )

    print(
        f"      Expected features: "
        f"{len(feature_columns)}"
    )


    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    prepared_data = prepare_features(
        data
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in prepared_data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    X = prepared_data[
        feature_columns
    ].copy()

    print(
        "[4/6] Feature engineering successful."
    )


    # --------------------------------------------------------
    # Use the known chronological test period
    # --------------------------------------------------------

    test_mask = (
        (
            prepared_data["Date"]
            >= pd.Timestamp("2023-10-01")
        )
        &
        (
            prepared_data["Date"]
            <= pd.Timestamp("2023-12-31")
        )
    )

    X_test = X.loc[
        test_mask
    ].copy()

    actual_test = prepared_data.loc[
        test_mask,
        target_column
    ].copy()


    if len(X_test) != 9200:
        raise ValueError(
            "Expected 9,200 test records, "
            f"but found {len(X_test):,}."
        )

    print(
        f"[5/6] Test period selected: "
        f"{len(X_test):,} rows"
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X_test
        )
    )

    predictions = model.predict(
        X_processed
    )


    if len(predictions) != len(X_test):
        raise ValueError(
            "Prediction count does not match "
            "test record count."
        )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    errors = (
        actual_test.to_numpy()
        -
        predictions
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )


    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            actual_test.to_numpy()
            -
            actual_test.mean()
        )
        ** 2
    )

    r2 = (
        1
        -
        ss_res / ss_tot
    )


    print(
        "[6/6] Predictions generated successfully."
    )

    print("\n" + "=" * 72)
    print("LOCAL MODEL TEST RESULTS")
    print("=" * 72)

    print(
        f"Test records : "
        f"{len(predictions):,}"
    )

    print(
        f"MAE          : "
        f"{mae:.3f}"
    )

    print(
        f"RMSE         : "
        f"{rmse:.3f}"
    )

    print(
        f"R²           : "
        f"{r2:.4f}"
    )


    # --------------------------------------------------------
    # Expected Kaggle results
    # --------------------------------------------------------

    expected_mae = 68.946
    expected_rmse = 87.923
    expected_r2 = 0.3390

    mae_match = (
        abs(mae - expected_mae)
        < 0.01
    )

    rmse_match = (
        abs(rmse - expected_rmse)
        < 0.01
    )

    r2_match = (
        abs(r2 - expected_r2)
        < 0.001
    )


    print("\nKAGGLE RESULT VALIDATION")
    print("-" * 72)

    print(
        f"MAE match  : {mae_match}"
    )

    print(
        f"RMSE match : {rmse_match}"
    )

    print(
        f"R² match   : {r2_match}"
    )


    if not (
        mae_match
        and rmse_match
        and r2_match
    ):

        raise ValueError(
            "Local predictions do not reproduce "
            "the expected Kaggle test metrics."
        )


    print(
        "\n✓ MODEL PORTABILITY TEST PASSED"
    )

    print(
        "✓ Saved Kaggle model works independently "
        "from the training notebook."
    )

    print(
        "✓ This project is ready for the "
        "VS Code validation stage."
    )


if __name__ == "__main__":
    main()
