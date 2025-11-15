# Member Messages Dataset Analysis

## Overview
I analyzed the member messages dataset from the API endpoint. The API reports a total of **3,349 messages**, but due to its current limitations, only **1,000 unique messages** were retrievable using time-based fetching with timestamps. The dataset includes 10 unique users.

---

## Anomaly Detection Approach
The following checks were applied to the dataset:

1. **Missing values**: Checked for missing fields in core columns (`id`, `user_id`, `user_name`, `timestamp`, `message`).  
2. **Duplicate message IDs**: Ensured each message `id` is unique.  
3. **Duplicate user IDs across names**: Verified that a single `user_id` isn’t associated with multiple names.  
4. **Duplicate names across user IDs**: Checked that a single `user_name` isn’t linked to multiple `user_id`s.  
5. **Case inconsistencies in first names**: Detected variations in capitalization.  
6. **Empty messages**: Checked for messages with no content.  
7. **Timestamp sanity**: Verified timestamps are valid, not in the distant past (before 2020) or future.  
8. **UUID validation**: Ensured `id` and `user_id` are valid UUIDs.  
9. **High message counts per user**: Flagged users exceeding the average number of messages (dynamic threshold based on dataset).

---

## Findings

| Anomaly Type               | Count | Notes |
|----------------------------|-------|-------|
| Missing values             | 0     | All core fields present |
| Duplicate message IDs      | 0     | All message IDs unique |
| Multiple names per user_id | 0     | No user_id maps to multiple names |
| Multiple user_ids per name | 0     | No name maps to multiple user_ids |
| Case inconsistencies        | 0     | No capitalization inconsistencies |
| Empty messages             | 0     | All messages contain content |
| Invalid timestamps         | 0     | All timestamps valid |
| Invalid UUIDs              | 0     | All IDs are valid UUIDs |
| High message counts         | 5     | Users sending more messages than average (100 messages/user) |

**High message count details:**

| user_id                               | user_name             | message_count |
|--------------------------------------|---------------------|---------------|
| 130f1fb9-2ddf-4049-ad0e-9a270f0cb561 | Vikram Desai        | 104           |
| 1a4b66ec-2fe6-46d8-9d6e-a81ec06bc5c5 | Lily O'Sullivan     | 107           |
| a1ac663a-277a-4782-a0ba-7efdca8ae2ee | Amina Van Den Berg  | 103           |
| cd3a350e-dbd2-408f-afa0-16a072f56d23 | Sophia Al-Farsi     | 106           |
| e35ed60a-5190-4a5f-b3cd-74ced7519b4a | Fatima El-Tahir     | 105           |

---

## Interpretation
- Overall, the dataset is **clean** with no missing values, duplicates, or invalid fields.  
- The **high message counts** indicate users with high activity, which may be worth further investigation. In this case though, the numbers were not signficently high. Even if we conside 1% of total users, the max difference was still 7. But this number could potentially be worth to be on the lookout for as the data increases.
- Due to API limitations, the analysis only includes **the latest 1,000 unique messages**. The `total` reported by the API (3,349) cannot be fully retrieved using the current pagination/limit.

---

## Recommendations

1. **API / Storage Improvements**
   - Support a **full export endpoint** or increase the `limit` beyond 1,000 messages.
   - Add support for **time-based pagination** (`start_time` / `end_time`) to allow sequential fetching of older messages.
   
2. **Data Quality**
   - Continue to validate UUIDs and timestamps as messages accumulate.
   - Track high-activity users to detect unusual behavior or system abuse.

3. **Future Analysis**
   - With full dataset access, recalculate average message thresholds to capture all high-activity users.
   - Consider additional anomaly detection: message length, repeated content, or sentiment outliers.

---

✅ **Conclusion:**  
The dataset is robust and well-structured. The main limitation is **API-imposed data retrieval constraints**, which currently restrict anomaly detection and analytics to 1,000 messages. High-activity users are clearly identifiable, and the data quality checks confirm overall consistency.
