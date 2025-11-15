import requests
import pandas as pd
import numpy as np
import uuid
import datetime

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"

# -------------------------
# Fetch data with pagination
# -------------------------
# -------------------------
# Fetch data with pagination (all unique IDs)
# -------------------------
def fetch_member_data(limit=1000):
    """Fetch all messages using timestamp to paginate and avoid API duplicates."""
    all_items = []
    seen_ids = set()
    before_ts = None  # Fetch latest first

    print("Starting time-based fetch...")
    while True:
        params = {"limit": limit}
        if before_ts:
            params["before"] = before_ts  # API should support a timestamp filter

        try:
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

        items = data.get("items", [])
        if not items:
            break

        # Keep only new unique items
        new_items = [item for item in items if item['id'] not in seen_ids]
        if not new_items:
            print("No new unique items, stopping fetch.")
            break

        all_items.extend(new_items)
        for item in new_items:
            seen_ids.add(item['id'])

        # Update before_ts to oldest timestamp in this batch
        timestamps = [item['timestamp'] for item in new_items]
        before_ts = min(timestamps)

        print(f"-> Fetched {len(seen_ids)} unique items so far...")

    df = pd.DataFrame(all_items)
    print(f"\nFetched {len(df)} unique rows in total.\n")
    return df



# -------------------------
# UUID validation helper
# -------------------------
def is_valid_uuid(val):
    """Check if a string or object can be parsed as a valid UUID."""
    if pd.isna(val):
        return False
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

# -------------------------
# Anomaly detection
# -------------------------
def detect_anomalies(df):
    anomalies = []

    # If the DataFrame is empty, we can't perform checks
    if df.empty:
        print("DataFrame is empty, skipping anomaly detection.")
        return pd.DataFrame()

    # -------------------------
    # 0. Split user_name
    # -------------------------
    if 'user_name' in df.columns:
        # Use .loc to ensure we are modifying the DataFrame correctly
        df.loc[:, ['first_name', 'last_name']] = df['user_name'].str.split(' ', n=1, expand=True)

    # -------------------------
    # 1. Missing values
    # -------------------------
    core_cols = ['id', 'user_id', 'user_name', 'timestamp', 'message']
    missing_data_cols = [col for col in core_cols if col in df.columns]

    missing = df[df[missing_data_cols].isna().any(axis=1)]
    print(f"Missing values check: {len(missing)} rows with missing fields in core columns")
    if not missing.empty:
        missing_copy = missing.copy()
        missing_copy["anomaly_type"] = "missing_fields"
        anomalies.append(missing_copy)

    # -------------------------
    # 2. Duplicate message IDs
    # -------------------------
    if 'id' in df.columns:
        duplicate_msg_ids = df[df.duplicated("id", keep=False)]
        print(f"Duplicate message ID check: {len(duplicate_msg_ids)} rows with duplicate 'id'")
        if not duplicate_msg_ids.empty:
            duplicate_msg_ids_copy = duplicate_msg_ids.copy()
            duplicate_msg_ids_copy["anomaly_type"] = "duplicate_message_id"
            anomalies.append(duplicate_msg_ids_copy)

    # -------------------------
    # 3. Duplicate user IDs across names
    # -------------------------
    if 'user_id' in df.columns and 'user_name' in df.columns:
        user_name_counts = df.groupby('user_id')['user_name'].nunique()
        duplicate_user_id_names = user_name_counts[user_name_counts > 1]
        print(f"Users with same user_id but multiple names: {len(duplicate_user_id_names)}")
        if not duplicate_user_id_names.empty:
            anomalies.append(pd.DataFrame({
                "user_id": duplicate_user_id_names.index,
                "num_names": duplicate_user_id_names.values,
                "anomaly_type": "user_id_multiple_names"
            }))

        user_id_counts = df.groupby('user_name')['user_id'].nunique()
        duplicate_name_ids = user_id_counts[user_id_counts > 1]
        print(f"Names associated with multiple user_ids: {len(duplicate_name_ids)}")
        if not duplicate_name_ids.empty:
            anomalies.append(pd.DataFrame({
                "user_name": duplicate_name_ids.index,
                "num_user_ids": duplicate_name_ids.values,
                "anomaly_type": "name_multiple_user_ids"
            }))

    # -------------------------
    # 4. Case inconsistencies in first names
    # -------------------------
    if 'first_name' in df.columns:
        # Convert the column to string type before using .str accessor
        name_lower = df['first_name'].dropna().astype(str).str.lower()
        duplicated_lower = name_lower[name_lower.duplicated(keep=False)]
        print(f"Case inconsistencies in first names: {len(duplicated_lower.unique())}")
        if not duplicated_lower.empty:
            anomalies.append(pd.DataFrame({
                "name_lower": duplicated_lower.unique(),
                "anomaly_type": "case_inconsistency_first_name"
            }))
    else:
        print("Case inconsistency check skipped: 'first_name' column not found.")


    # -------------------------
    # 5. Empty messages
    # -------------------------
    if 'message' in df.columns:
        empty_messages = df[df['message'].astype(str).str.strip() == '']
        print(f"Empty messages: {len(empty_messages)}")
        if not empty_messages.empty:
            empty_messages_copy = empty_messages.copy()
            empty_messages_copy["anomaly_type"] = "empty_message"
            anomalies.append(empty_messages_copy)

    # -------------------------
    # 6. Timestamp sanity
    # -------------------------
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        
        invalid_ts = df[timestamps.isna()]
        print(f"Invalid timestamp parsing: {len(invalid_ts)} rows")
        if not invalid_ts.empty:
            invalid_ts_copy = invalid_ts.copy()
            invalid_ts_copy["anomaly_type"] = "invalid_timestamp"
            anomalies.append(invalid_ts_copy)

        # Filter out invalid timestamps before date range checks
        valid_ts_df = df[timestamps.notna()]
        valid_timestamps = timestamps[timestamps.notna()]

        # Use timezone-aware current time for accurate comparison
        current_time = pd.Timestamp.now(tz='utc') 

        old_msgs = valid_ts_df[valid_timestamps < pd.Timestamp('2020-01-01', tz='utc')]
        future_msgs = valid_ts_df[valid_timestamps > current_time]
        print(f"Old messages: {len(old_msgs)}, Future messages: {len(future_msgs)}")
        if not old_msgs.empty:
            old_msgs_copy = old_msgs.copy()
            old_msgs_copy["anomaly_type"] = "old_timestamp"
            anomalies.append(old_msgs_copy)
        if not future_msgs.empty:
            future_msgs_copy = future_msgs.copy()
            future_msgs_copy["anomaly_type"] = "future_timestamp"
            anomalies.append(future_msgs_copy)

    # -------------------------
    # 7. Invalid UUIDs
    # -------------------------
    if 'id' in df.columns:
        invalid_id = df[~df['id'].apply(is_valid_uuid)]
        print(f"Invalid message UUIDs: {len(invalid_id)}")
        if not invalid_id.empty:
            invalid_id_copy = invalid_id.copy()
            invalid_id_copy["anomaly_type"] = "invalid_message_uuid"
            anomalies.append(invalid_id_copy)
            
    if 'user_id' in df.columns:
        invalid_user_id = df[~df['user_id'].apply(is_valid_uuid)]
        print(f"Invalid user UUIDs: {len(invalid_user_id)}")
        if not invalid_user_id.empty:
            invalid_user_id_copy = invalid_user_id.copy()
            invalid_user_id_copy["anomaly_type"] = "invalid_user_uuid"
            anomalies.append(invalid_user_id_copy)

    # -------------------------
    # 8. High message counts per user (Dynamic Threshold)
    # -------------------------
    if 'user_id' in df.columns and 'user_name' in df.columns:
        user_msg_counts = df.groupby('user_id').size()
        
        total_messages = len(df)
        total_unique_users = user_msg_counts.shape[0]

        if total_unique_users > 0:
            avg_messages_per_user = total_messages / total_unique_users
            threshold = avg_messages_per_user
            
            # Print the average message count as requested
            print(f"\n--- Dynamic Threshold Insight ---")
            print(f"Total Messages: {total_messages}")
            print(f"Total Unique Users: {total_unique_users}")
            print(f"Average Messages per User (Threshold): {threshold:.2f}")
            
            # Filter users whose count is strictly greater than the average
            high_activity = user_msg_counts[user_msg_counts > threshold]

            print(f"Users with unusually high message counts (>{threshold:.2f}): {len(high_activity)}")
            
            if not high_activity.empty:
                # 1. Create the base anomaly dataframe
                high_activity_df = pd.DataFrame({
                    "user_id": high_activity.index,
                    "message_count": high_activity.values,
                    "anomaly_type": "high_message_count"
                })

                # 2. Get a unique mapping of user_id to user_name
                # This uses the first occurrence of the user_name for the user_id
                user_name_map = df[['user_id', 'user_name']].drop_duplicates(subset=['user_id'], keep='first')
                
                # 3. Merge the user name into the high activity report
                high_activity_df = pd.merge(
                    high_activity_df,
                    user_name_map,
                    on='user_id',
                    how='left'
                )

                # 4. Append to anomalies list
                anomalies.append(high_activity_df)
        else:
            print("High message count check skipped: No unique users found.")


    # Combine all anomalies
    if anomalies:
        combined = pd.concat(anomalies, ignore_index=True)
        # Drop duplicates that may have been flagged by multiple checks
        combined = combined.drop_duplicates(subset=combined.columns.difference(['anomaly_type']))
    else:
        combined = pd.DataFrame()
        print("\nAll checks passed: No anomalies detected.\n")

    return combined

# -------------------------
# Main
# -------------------------
def main():
    df = fetch_member_data()
    anomalies = detect_anomalies(df)

    if not anomalies.empty:
        # Define the set of columns to keep in the final output
        # Added 'message_count' to ensure it's captured in the final output file and printed head
        cols_to_keep = ['id', 'user_id', 'user_name', 'message_count', 'timestamp', 'message', 'anomaly_type']
        
        # Filter the combined DataFrame to only include present, desired columns
        final_anomalies = anomalies[[col for col in cols_to_keep if col in anomalies.columns]]
        
        final_anomalies.to_csv("anomalies.csv", index=False)
        print("\nAnomalies found and written to anomalies.csv:")
        print(final_anomalies.head(20)) # Print more head rows to show high-activity details
    else:
        print("Dataset passed all anomaly checks.")

if __name__ == "__main__":
    main()