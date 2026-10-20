import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model():
    # 1. Load Processed Data
    print("Loading processed data...")
    df = pd.read_csv('data/processed_shipments.csv')

    # 2. Define Target and Features
    target = 'final_delay_hours'

    # Exclude IDs and any other columns that aren't useful for prediction
    # We remove shipment_id as it's just a unique identifier
    cols_to_exclude = ['shipment_id', target]
    X = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
    y = df[target]

    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

    # 4. Build Preprocessing Pipeline
    # Separate numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Numeric pipeline: Impute missing -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing -> One-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 5. Create Full Pipeline with RandomForestRegressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 6. Train the Model
    print("Training RandomForestRegressor baseline model...")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluate the Model
    predictions = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    report = (
        f"--- Baseline Model Report ---\n"
        f"Training set size: {len(X_train)}\n"
        f"Testing set size: {len(X_test)}\n"
        f"Numerical features: {numeric_features}\n"
        f"Categorical features: {categorical_features}\n"
        f"----------------------------\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"R-squared (R2): {r2:.4f}\n"
        f"----------------------------\n"
    )

    with open('base_model_report.txt', 'w') as f:
        f.write(report)
    print("Evaluation metrics saved to base_model_report.txt")

    # 8. Save the Model
    model_path = 'models/shipment_model.pkl'
    joblib.dump(model_pipeline, model_path)
    print(f"\nTrained model saved to {model_path}")

if __name__ == "__main__":
    train_model()
