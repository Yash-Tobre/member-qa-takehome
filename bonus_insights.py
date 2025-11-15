import requests
import pandas as pd
import numpy as np

API_URL = "https://november7-730026606190.europe-west1.run.app/messages?limit=1000"   # Replace if needed

def fetch_member_data():
    """Fetch member dataset from the API."""
    response = requests.get(API_URL)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def detect_anomalies(df):
    anomalies = []

    # 1. Missing values
    missing = df[df.isna().any(axis=1)]
    if not missing.empty:
        missing["anomaly_type"] = "missing_fields"
        anomalies.append(missing)

    # 2. Duplicate member IDs
    if "id" in df.columns:
        duplicates = df[df.duplicated("id", keep=False)]
        if not duplicates.empty:
            duplicates["anomaly_type"] = "duplicate_ids"
            anomalies.append(duplicates)

    # 3. Invalid email format
    if "email" in df.columns:
        invalid_email = df[~df["email"].str.contains(r"[^@]+@[^@]+\.[^@]+", na=False)]
        if not invalid_email.empty:
            invalid_email["anomaly_type"] = "invalid_email_format"
            anomalies.append(invalid_email)

    # 4. Impossible or negative numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        invalid = df[df[col] < 0]
        if not invalid.empty:
            invalid["anomaly_type"] = f"negative_value_in_{col}"
            anomalies.append(invalid)

    # 5. Timestamp anomalies
    if "created" in df.columns:
        # Non-parsable timestamps
        invalid_ts = df[pd.to_datetime(df["created"], errors="coerce").isna()]
        if not invalid_ts.empty:
            invalid_ts["anomaly_type"] = "invalid_timestamp"
            anomalies.append(invalid_ts)

    if anomalies:
        combined = pd.concat(anomalies).drop_duplicates()
    else:
        combined = pd.DataFrame()

    return combined

def main():
    df = fetch_member_data()
    print("Fetched member dataset with", len(df), "rows.")

    anomalies = detect_anomalies(df)

    if anomalies.empty:
        print("No anomalies detected.")
    else:
        anomalies.to_csv("anomalies.csv", index=False)
        print("Anomalies found and written to anomalies.csv:")
        print(anomalies.head())

if __name__ == "__main__":
    main()
