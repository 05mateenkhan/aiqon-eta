"""
Generate a reproducible synthetic shipment dataset for the Aiqon ETA assignment.

Prediction point:
    After the shipment reaches its intermediate port, using only information
    available at or before that point.

Output:
    data/shipments.csv (300 rows by default)

Run:
    python generate_data.py
    python generate_data.py --rows 300 --output data/shipments.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PORTS = [
    "Mumbai",
    "Chennai",
    "Singapore",
    "Rotterdam",
    "Hamburg",
    "Dubai",
    "Colombo",
    "Shanghai",
]

# Illustrative route distances in km. These are synthetic values for this demo,
# not authoritative shipping-route distances.
ROUTES = [
    ("Mumbai", "Rotterdam", 7200),
    ("Mumbai", "Singapore", 3900),
    ("Chennai", "Singapore", 2900),
    ("Chennai", "Rotterdam", 7900),
    ("Shanghai", "Rotterdam", 10400),
    ("Shanghai", "Hamburg", 10200),
    ("Dubai", "Rotterdam", 6100),
    ("Colombo", "Hamburg", 8300),
    ("Singapore", "Hamburg", 9300),
    ("Singapore", "Rotterdam", 8800),
]


def generate_dataset(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic shipment journeys with stage-dependent delays."""
    if rows < 30:
        raise ValueError("Use at least 30 rows so the dataset can be split for evaluation.")

    rng = np.random.default_rng(seed)
    records = []

    base_departure = pd.Timestamp("2025-01-01 00:00:00")

    for i in range(rows):
        origin, destination, distance_km = ROUTES[int(rng.integers(0, len(ROUTES)))]

        # Planned schedule: longer routes have longer transit times.
        departure_offset_days = int(rng.integers(0, 365))
        scheduled_departure = base_departure + pd.Timedelta(
            days=departure_offset_days,
            hours=int(rng.integers(0, 24)),
        )
        scheduled_transit_hours = distance_km / rng.uniform(24, 34)
        scheduled_port_arrival = scheduled_departure + pd.Timedelta(
            hours=scheduled_transit_hours
        )

        # Planned final delivery includes an illustrative post-port allowance.
        scheduled_processing_hours = float(rng.uniform(8, 28))
        scheduled_customs_clearance = scheduled_port_arrival + pd.Timedelta(
            hours=scheduled_processing_hours
        )
        scheduled_last_mile_hours = float(rng.uniform(12, 60))
        scheduled_final_arrival = (
            scheduled_customs_clearance + pd.Timedelta(hours=scheduled_last_mile_hours)
        )

        # External conditions known at/around port arrival.
        port_congestion_index = float(np.clip(rng.beta(2.2, 2.0), 0, 1))
        weather_severity = float(np.clip(rng.beta(1.6, 3.0), 0, 1))

        # Departure delay (hours): skewed positive with some early departures.
        departure_delay_hours = float(np.clip(rng.normal(3.0, 8.0), -8, 36))
        actual_departure = scheduled_departure + pd.Timedelta(
            hours=departure_delay_hours
        )

        # Transit disruption depends partly on weather and route length.
        transit_weather_effect = weather_severity * rng.uniform(0, 20)
        transit_noise = rng.normal(0, 7)
        transit_delay_hours = float(
            np.clip(
                departure_delay_hours * rng.uniform(0.45, 0.9)
                + transit_weather_effect
                + transit_noise,
                -8,
                60,
            )
        )
        actual_port_arrival = scheduled_port_arrival + pd.Timedelta(
            hours=transit_delay_hours
        )
        port_arrival_delay_hours = (
            actual_port_arrival - scheduled_port_arrival
        ).total_seconds() / 3600.0

        # At the prediction point, this is an estimate, not the future actual.
        expected_port_processing_hours = float(
            np.clip(
                scheduled_processing_hours
                + port_congestion_index * 30
                + weather_severity * 5
                + rng.normal(0, 4),
                4,
                80,
            )
        )

        # Future outcomes: actual processing, customs, and last-mile durations.
        # These must NOT be used as model features at port-arrival prediction time.
        actual_port_processing_hours = float(
            np.clip(
                scheduled_processing_hours
                + port_congestion_index * 42
                + weather_severity * 7
                + max(0, port_arrival_delay_hours) * 0.08
                + rng.normal(0, 5),
                3,
                100,
            )
        )

        # Delays can partially recover if schedule buffer is available.
        recovery_hours = float(
            np.clip(rng.normal(7, 5), 0, 18)
        )
        customs_extra_hours = float(
            np.clip(
                port_congestion_index * 10
                + weather_severity * 3
                + rng.normal(0, 5),
                -8,
                30,
            )
        )
        actual_customs_clearance = (
            actual_port_arrival
            + pd.Timedelta(hours=actual_port_processing_hours)
        )
        actual_customs_clearance += pd.Timedelta(hours=customs_extra_hours)

        actual_last_mile_hours = float(
            np.clip(
                scheduled_last_mile_hours
                + rng.normal(0, 8)
                + port_congestion_index * 4,
                6,
                90,
            )
        )

        # Final delay propagates from port arrival, processing, customs and delivery.
        actual_final_delivery = (
            actual_customs_clearance + pd.Timedelta(hours=actual_last_mile_hours)
        )
        actual_final_delivery -= pd.Timedelta(
            hours=min(recovery_hours, max(0, transit_delay_hours))
        )

        final_delay_hours = (
            actual_final_delivery - scheduled_final_arrival
        ).total_seconds() / 3600.0

        records.append(
            {
                "shipment_id": f"SHP-{i + 1:04d}",
                "origin_port": origin,
                "destination_port": destination,
                "route_distance_km": float(distance_km),
                "scheduled_departure": scheduled_departure,
                "actual_departure": actual_departure,
                "scheduled_port_arrival": scheduled_port_arrival,
                "actual_port_arrival": actual_port_arrival,
                "scheduled_final_arrival": scheduled_final_arrival,
                "departure_delay_hours": departure_delay_hours,
                "port_arrival_delay_hours": port_arrival_delay_hours,
                "port_congestion_index": port_congestion_index,
                "weather_severity": weather_severity,
                "expected_port_processing_hours": expected_port_processing_hours,
                "actual_customs_clearance": actual_customs_clearance,
                "actual_port_processing_hours": actual_port_processing_hours,
                "scheduled_customs_clearance": scheduled_customs_clearance,
                "actual_final_delivery": actual_final_delivery,
                "final_delay_hours": final_delay_hours,
            }
        )

    df = pd.DataFrame(records)

    # Keep timestamps ISO-formatted in the CSV for portability.
    timestamp_columns = [
        "scheduled_departure",
        "actual_departure",
        "scheduled_port_arrival",
        "actual_port_arrival",
        "scheduled_final_arrival",
        "actual_customs_clearance",
        "scheduled_customs_clearance",
        "actual_final_delivery",
    ]
    for column in timestamp_columns:
        df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic shipment ETA data.")
    parser.add_argument("--rows", type=int, default=300, help="Number of shipments (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shipments.csv"),
        help="Output CSV path (default: data/shipments.csv)",
    )
    args = parser.parse_args()

    df = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Generated {len(df)} shipments: {args.output}")
    print(f"Columns: {len(df.columns)}")
    print("\nPreview:")
    print(df.head(5).to_string(index=False))
    print("\nFinal delay summary (hours):")
    print(df["final_delay_hours"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
