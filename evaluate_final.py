
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_final():
    print("Loading data and model...")

    # 1. Load the same processed dataset used for training.
    df = pd.read_csv("data/processed_shipments.csv")

    try:
        model = joblib.load("models/improved_shipment_model.pkl")
    except FileNotFoundError:
        print("Model not found. Run improve_model.py first.")
        return

    # 2. Reproduce the EXACT feature engineering from improve_model.py.
    datetime_cols = [
        col for col in df.columns
        if "scheduled" in col and "duration" not in col
    ]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
        df[f"{col}_timestamp"] = (
            df[col].astype("int64") // 10**9
        )

    # Drop original datetime columns, as done during training.
    df = df.drop(columns=datetime_cols)

    # 3. Recreate the same target and test split.
    target = "final_delay_hours"

    X = df.drop(
        columns=["shipment_id", target],
        errors="ignore"
    )
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Check the model's expected input columns.
    expected = list(
        model.named_steps["preprocessor"].feature_names_in_
    )

    if X_test.columns.tolist() != expected:
        missing = set(expected) - set(X_test.columns)
        extra = set(X_test.columns) - set(expected)

        raise ValueError(
            f"Feature mismatch.\nMissing: {missing}\nExtra: {extra}"
        )

    # 4. Calculate final metrics.
    print("Calculating final metrics...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    report = (
        "--- Final Evaluation Report ---\n"
        f"Mean Absolute Error (MAE): {mae:.4f} hours\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f} hours\n"
        f"R-squared (R2): {r2:.4f}\n"
    )

    os.makedirs("results", exist_ok=True)

    with open("results/final_evaluation.txt", "w") as f:
        f.write(report)

    print(report)

    # 5. Save actual and predicted values.
    predictions_df = pd.DataFrame({
    "Actual_Delay_Hours": y_test.to_numpy(),
    "Predicted_Delay_Hours": predictions,
}, index=y_test.index)

# Recover the scheduled arrival timestamps from the original dataset.
    scheduled_arrival = pd.to_datetime(
        pd.read_csv("data/processed_shipments.csv")
        .loc[y_test.index, "scheduled_final_arrival"],
        errors="coerce"
)

    predictions_df["Scheduled_Final_Arrival"] = (
        scheduled_arrival.to_numpy()
    )

    predictions_df["Predicted_Final_ETA"] = (
        scheduled_arrival
        + pd.to_timedelta(predictions, unit="h")
    ).to_numpy()

    predictions_df["Actual_Final_ETA"] = (
        scheduled_arrival
        + pd.to_timedelta(y_test.to_numpy(), unit="h")
    ).to_numpy()

    predictions_df.to_csv("results/predictions.csv", index=False)

    print("\n--- Sample Final ETA Predictions ---")
    print(predictions_df.head(10).to_string(index=False))
    print("\nPredictions saved to results/predictions.csv")


    # 6. Save the actual vs predicted plot.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())

    plt.plot([lower, upper], [lower, upper], "r--", linewidth=2)
    plt.xlabel("Actual Delay (Hours)")
    plt.ylabel("Predicted Delay (Hours)")
    plt.title("Actual vs Predicted Shipment Delays")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted.png")
    plt.close()

    print("All evaluation files saved in results/.")


if __name__ == "__main__":
    evaluate_final()
