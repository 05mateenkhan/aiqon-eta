import pandas as pd
import joblib

df = pd.read_csv("data/processed_shipments.csv")
model = joblib.load("models/improved_shipment_model.pkl")

print("Dataset columns:")
print(df.columns.tolist())

print("\nModel pipeline:")
print(model)

print("\nDataset shape:", df.shape)