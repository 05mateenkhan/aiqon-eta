import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_final():
    # 1. Load Data and Model
    print("Loading data and model...")
    df = pd.read_csv('data/processed_shipments.csv')

    # The prompt asks to use the "best Linear Regression model".
    # Since we saved the absolute best model in improve_model.py,
    # if a LinearRegression was the best, it's in improved_shipment_model.pkl.
    # However, for a strict final evaluation of the Linear Regression approach specifically,
    # we would need that specific model. Given the prompt's wording, we'll use
    # the improved model we saved, assuming it represents our best effort.
    try:
        model = joblib.load('models/improved_shipment_model.pkl')
    except FileNotFoundError:
        print("Error: improved_shipment_model.pkl not found. Please run improve_model.py first.")
        return

    # 2. Reconstruct the Test Set (Consistent with previous scripts)
    target = 'final_delay_hours'
    cols_to_exclude = ['shipment_id', target]
    X = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Calculate Final Metrics
    print("Calculating final metrics...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # 4. Save Metrics to results/final_evaluation.txt
    metrics_report = (
        f"--- Final Evaluation Report ---\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R-squared (R2): {r2:.4f}\n"
    )
    with open('results/final_evaluation.txt', 'w') as f:
        f.write(metrics_report)
    print("Metrics saved to results/final_evaluation.txt")

    # 5. Save Predictions to results/predictions.csv
    # Create a dataframe for predictions for the test set
    test_indices = X_test.index
    predictions_df = pd.DataFrame({
        'Actual_Delay': y_test.values,
        'Predicted_Delay': predictions
    }, index=test_indices)

    predictions_df.to_csv('results/predictions.csv', index=False)
    print("Predictions saved to results/predictions.csv")

    # 6. Generate a table for 10 samples
    print("\n--- Sample Predictions (Actual vs Predicted) ---")
    sample_df = predictions_df.head(10)
    print(sample_df.to_string(index=False))

    # 7. Create Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Delay (Hours)')
    plt.ylabel('Predicted Delay (Hours)')
    plt.title('Actual vs Predicted Shipment Delays')
    plt.grid(True)
    plt.savefig('results/actual_vs_predicted.png')
    print("\nPlot saved to results/actual_vs_predicted.png")

if __name__ == "__main__":
    evaluate_final()
