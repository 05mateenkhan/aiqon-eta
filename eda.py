from contextlib import redirect_stdout

def run_eda():

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    DATA_PATH = "data/shipments.csv"
    OUTPUT_DIR = "eda_plots"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid")

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe(include="all").T)

    # -----------------------------
    # 2. Missing values
    # -----------------------------
    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percent
    })

    print(missing_summary[missing_summary["Missing_Count"] > 0])

    print("\nDuplicate rows:", df.duplicated().sum())

    # -----------------------------
    # 3. Convert timestamp columns
    # -----------------------------
    date_columns = [
        "scheduled_departure",
        "actual_departure",
        "scheduled_port_arrival",
        "actual_port_arrival",
        "scheduled_final_arrival",
        "actual_customs_clearance",
        "scheduled_customs_clearance",
        "actual_final_delivery"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # -----------------------------
    # 4. Target variable analysis
    # -----------------------------
    TARGET = "final_delay_hours"

    if TARGET in df.columns:
        print("\n========== TARGET ANALYSIS ==========")
        print(df[TARGET].describe())

        print("On-time or early deliveries:",
            (df[TARGET] <= 0).sum())

        print("Late deliveries:",
            (df[TARGET] > 0).sum())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(df[TARGET].dropna(), kde=True, ax=axes[0])
        axes[0].set_title("Distribution of Final Delay")
        axes[0].set_xlabel("Final delay (hours)")

        sns.boxplot(x=df[TARGET], ax=axes[1])
        axes[1].set_title("Final Delay Outliers")
        axes[1].set_xlabel("Final delay (hours)")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=150)
        plt.show()

    # -----------------------------
    # 5. Missing values visualization
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=150)
    plt.show()

    # -----------------------------
    # 6. Numerical feature distributions
    # -----------------------------
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        numeric_df.hist(figsize=(16, 12), bins=25, edgecolor="black")
        plt.suptitle("Numerical Feature Distributions")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/numerical_distributions.png", dpi=150)
        plt.show()

    # -----------------------------
    # 7. Correlation analysis
    # -----------------------------
    if len(numeric_df.columns) > 1:
        correlation = numeric_df.corr()

        print("\n========== CORRELATION WITH TARGET ==========")
        if TARGET in correlation.columns:
            print(
                correlation[TARGET]
                .drop(TARGET)
                .sort_values(key=abs, ascending=False)
            )

        plt.figure(figsize=(13, 9))
        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            center=0
        )
        plt.title("Numerical Feature Correlation")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
        plt.show()

    # -----------------------------
    # 8. Categorical feature analysis
    # -----------------------------
    categorical_cols = [
        col for col in ["origin_port", "destination_port"]
        if col in df.columns
    ]

    for col in categorical_cols:
        print(f"\nAverage final delay by {col}:")
        print(
            df.groupby(col)[TARGET]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
            if TARGET in df.columns
            else df[col].value_counts()
        )

        if TARGET in df.columns:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=df, x=col, y=TARGET, errorbar=None)
            plt.title(f"Average Final Delay by {col}")
            plt.ylabel("Average delay (hours)")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{col}_delay.png", dpi=150)
            plt.show()

    # -----------------------------
    # 9. Feature relationships with target
    # -----------------------------
    features_to_check = [
        "route_distance_km",
        "departure_delay_hours",
        "port_arrival_delay_hours",
        "port_congestion_index",
        "weather_severity",
        "expected_port_processing_hours",
        "actual_port_processing_hours"
    ]

    for col in features_to_check:
        if col in df.columns and TARGET in df.columns:
            plt.figure(figsize=(7, 5))
            sns.regplot(
                data=df,
                x=col,
                y=TARGET,
                scatter_kws={"alpha": 0.5},
                line_kws={"color": "red"}
            )
            plt.title(f"{col} vs Final Delay")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{col}_vs_delay.png", dpi=150)
            plt.show()

    # -----------------------------
    # 10. Delay by departure month
    # -----------------------------
    if "scheduled_departure" in df.columns and TARGET in df.columns:
        df["departure_month"] = df["scheduled_departure"].dt.month

        monthly_delay = df.groupby("departure_month")[TARGET].mean()

        print("\nAverage delay by departure month:")
        print(monthly_delay)

        monthly_delay.plot(kind="bar", figsize=(9, 5))
        plt.title("Average Final Delay by Departure Month")
        plt.xlabel("Month")
        plt.ylabel("Average delay (hours)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/monthly_delay.png", dpi=150)
        plt.show()

    # -----------------------------
    # 11. Save numerical correlation
    # -----------------------------
    if len(numeric_df.columns) > 1:
        correlation.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")

    print("\nEDA completed.")
    print(f"Plots and correlation output saved in: {OUTPUT_DIR}/")
    pass

with open("eda_report.txt", "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        run_eda()

print("EDA report saved to eda_report.txt")