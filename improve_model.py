import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

def improve_model():
    # 1. Load Processed Data
    print("Loading processed data...")
    df = pd.read_csv('data/processed_shipments.csv')

    # 2. Enhanced Feature Engineering (Handling Datetimes)
    # The baseline kept 'scheduled_...' columns as strings/objects.
    # We convert them to numeric features to make them usable by Linear Regression.
    datetime_cols = [col for col in df.columns if 'scheduled' in col and 'duration' not in col]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
        # Convert to unix timestamp (seconds) to make it numeric
        df[f"{col}_timestamp"] = df[col].astype('int64') // 10**9

    # Drop original datetime columns to avoid issues with the pipeline
    df = df.drop(columns=datetime_cols)

    # 3. Define Target and Features
    target = 'final_delay_hours'
    cols_to_exclude = ['shipment_id', target]
    X = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
    y = df[target]

    # 4. Consistent Reproducible Split (Same as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Preprocessing Pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 6. Model Comparison
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_model = None
    best_mae = float('inf')

    print("\nComparing models...")
    for name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        pipeline.fit(X_train, y_train)
        mae, rmse, r2 = evaluate_model(pipeline, X_test, y_test)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = pipeline

    # 7. Save the Improved Model
    model_path = 'models/improved_shipment_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to {model_path}")

    # 8. Generate Comparison Report
    try:
        with open('base_model_report.txt', 'r') as f:
            baseline_report = f.read()
    except FileNotFoundError:
        baseline_report = "Baseline report not found."

    # Extract metrics from the best improved model
    best_name = max(results, key=lambda k: results[k]['R2']) # Simple heuristic for best
    best_metrics = results[best_name]

    comparison_report = (
        f"--- Model Comparison Report ---\n\n"
        f"BASELINE REPORT:\n{baseline_report}\n"
        f"----------------------------\n"
        f"IMPROVED MODEL ({best_name}):\n"
        f"MAE: {best_metrics['MAE']:.4f}\n"
        f"RMSE: {best_metrics['RMSE']:.4f}\n"
        f"R2: {best_metrics['R2']:.4f}\n"
        f"----------------------------\n"
        f"ALL TRIALS:\n"
    )

    for name, m in results.items():
        comparison_report += f"{name}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}\n"

    with open('model_comparison_report.txt', 'w') as f:
        f.write(comparison_report)
    print("Comparison report saved to model_comparison_report.txt")

if __name__ == "__main__":
    improve_model()
