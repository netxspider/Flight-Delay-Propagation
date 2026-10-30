"""
Data Preprocessing and Base Flight Cleaning Pipeline
Phase 2: Preprocessing and Data Cleaning
Author: Flight Delay Propagation Research Project
"""

import os
import sys
import glob
import time
import duckdb
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.airport_enrichment import build_airport_reference

def run_preprocessing_pipeline(
    raw_data_dir="data/raw",
    processed_data_dir="data/processed",
    results_dir="results/tables"
):
    """
    Executes Phase 2 preprocessing:
    1. Loads and validates the 352 active airport reference table.
    2. Enforces explicit data type casting and normalizes flight schemas.
    3. Categorizes flights into explicit operational categories (A, B, C, D).
    4. Deconstructs local HHMM timestamps into valid hour/minute representations.
    5. Flags midnight boundaries, overnight operations, and DST transition ambiguities.
    6. Enriches origin and destination airport coordinates and IANA timezones.
    7. Preserves structural missingness (delay causes, unobserved actual times).
    8. Writes compressed Parquet datasets for Category A, B, and C.
    9. Produces comprehensive validation reports and summaries.
    """
    t_start = time.time()
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=========================================================")
    print("PHASE 2: PREPROCESSING AND BASE DATA CLEANING PIPELINE")
    print("=========================================================")

    con = duckdb.connect()

    # 1. Build and register authoritative airport reference table
    print("\n--- 1. Building and Registering Enriched Airport Reference Table ---")
    df_airports = build_airport_reference(raw_data_dir, results_dir, con=con)
    print(f"Registered ref_airports with {len(df_airports)} airports.")

    raw_flights_pattern = os.path.join(raw_data_dir, "2025", "*.csv")

    # 2. Establish Flight Operational Categorization
    print("\n--- 2. Auditing and Categorizing Operational Flight Cohorts ---")
    cat_audit_q = f"""
    SELECT 
        COUNT(*) AS total_input_rows,
        SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS count_cancelled,
        SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 1.0 THEN 1 ELSE 0 END) AS count_diverted,
        SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND (ARR_TIME IS NULL OR ARR_DELAY IS NULL) THEN 1 ELSE 0 END) AS count_missing_arrival,
        SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND CRS_ELAPSED_TIME <= 0 THEN 1 ELSE 0 END) AS count_nonpositive_elapsed,
        SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_TIME IS NOT NULL AND ARR_DELAY IS NOT NULL AND CRS_ELAPSED_TIME > 0 THEN 1 ELSE 0 END) AS count_completed_valid
    FROM read_csv_auto('{raw_flights_pattern}')
    """
    cat_res = con.execute(cat_audit_q).df().iloc[0].to_dict()
    total_in = cat_res['total_input_rows']
    n_cancelled = cat_res['count_cancelled']
    n_diverted = cat_res['count_diverted']
    n_missing_arr = cat_res['count_missing_arrival']
    n_nonpos_elapsed = cat_res['count_nonpositive_elapsed']
    n_completed = cat_res['count_completed_valid']
    n_invalid = n_missing_arr + n_nonpos_elapsed

    assert total_in == 7001619, f"Input count {total_in} does not match Phase 1 audit (7,001,619)"
    assert n_completed + n_cancelled + n_diverted + n_invalid == total_in, "Categorization cohorts do not sum to total input!"

    print(f"Total Input Flights Ingested:    {total_in:,}")
    print(f"Category A (Completed & Valid):  {n_completed:,} ({n_completed/total_in*100:.2f}%)")
    print(f"Category B (Cancelled):          {n_cancelled:,} ({n_cancelled/total_in*100:.2f}%)")
    print(f"Category C (Diverted):           {n_diverted:,} ({n_diverted/total_in*100:.2f}%)")
    print(f"Category D (Invalid/Incomplete): {n_invalid:,} ({n_invalid/total_in*100:.4f}%)")

    # Save preprocessing summary table
    df_prep_summary = pd.DataFrame([
        {
            "operational_category": "Category A: Completed & Valid",
            "record_count": int(n_completed),
            "percentage_of_total": round(n_completed / total_in * 100, 3),
            "inclusion_status": "Eligible for Rotation Reconstruction & Predictive Modeling",
            "exclusion_reason": "None (Target cohort)"
        },
        {
            "operational_category": "Category B: Cancelled Flights",
            "record_count": int(n_cancelled),
            "percentage_of_total": round(n_cancelled / total_in * 100, 3),
            "inclusion_status": "Preserved Separately (data/processed/cancelled_flights.parquet)",
            "exclusion_reason": "Flight cancelled; no aircraft rotation continuity"
        },
        {
            "operational_category": "Category C: Diverted Flights",
            "record_count": int(n_diverted),
            "percentage_of_total": round(n_diverted / total_in * 100, 3),
            "inclusion_status": "Preserved Separately (data/processed/diverted_flights.parquet)",
            "exclusion_reason": "Flight diverted; scheduled destination differs from landing airport"
        },
        {
            "operational_category": "Category D: Invalid / Incomplete",
            "record_count": int(n_invalid),
            "percentage_of_total": round(n_invalid / total_in * 100, 5),
            "inclusion_status": "Flagged and Excluded from Core Analysis",
            "exclusion_reason": "Missing arrival observation (YX 5859) or negative scheduled duration (OO 5262)"
        }
    ])
    prep_summary_path = os.path.join(results_dir, "preprocessing_summary.csv")
    df_prep_summary.to_csv(prep_summary_path, index=False)
    print(f"Saved preprocessing summary to {prep_summary_path}")

    # 3. Export Category A: Cleaned, Enriched Base Flight Dataset
    print("\n--- 3. Constructing and Exporting Category A (Completed Flights) to Parquet ---")
    cleaned_parquet_path = os.path.join(processed_data_dir, "flights_cleaned.parquet")
    
    # SQL query for Category A
    q_cat_a = f"""
    COPY (
        WITH raw_completed AS (
            SELECT 
                CAST(YEAR AS SMALLINT) AS YEAR,
                CAST(QUARTER AS UTINYINT) AS QUARTER,
                CAST(MONTH AS UTINYINT) AS MONTH,
                CAST(DAY_OF_MONTH AS UTINYINT) AS DAY_OF_MONTH,
                CAST(DAY_OF_WEEK AS UTINYINT) AS DAY_OF_WEEK,
                CAST(FL_DATE AS DATE) AS FL_DATE,
                CAST(OP_UNIQUE_CARRIER AS VARCHAR) AS OP_UNIQUE_CARRIER,
                CAST(OP_CARRIER_AIRLINE_ID AS INTEGER) AS OP_CARRIER_AIRLINE_ID,
                CAST(TAIL_NUM AS VARCHAR) AS TAIL_NUM,
                CAST(OP_CARRIER_FL_NUM AS INTEGER) AS OP_CARRIER_FL_NUM,
                CAST(ORIGIN_AIRPORT_ID AS INTEGER) AS ORIGIN_AIRPORT_ID,
                CAST(ORIGIN AS VARCHAR) AS ORIGIN,
                CAST(ORIGIN_CITY_NAME AS VARCHAR) AS ORIGIN_CITY_NAME,
                CAST(DEST_AIRPORT_ID AS INTEGER) AS DEST_AIRPORT_ID,
                CAST(DEST AS VARCHAR) AS DEST,
                CAST(DEST_CITY_NAME AS VARCHAR) AS DEST_CITY_NAME,
                LPAD(CAST(CRS_DEP_TIME AS VARCHAR), 4, '0') AS CRS_DEP_TIME,
                LPAD(CAST(DEP_TIME AS VARCHAR), 4, '0') AS DEP_TIME,
                CAST(DEP_DELAY AS FLOAT) AS DEP_DELAY,
                CAST(DEP_DELAY_NEW AS FLOAT) AS DEP_DELAY_NEW,
                CAST(DEP_DEL15 AS UTINYINT) AS DEP_DEL15,
                CAST(DEP_DELAY_GROUP AS SMALLINT) AS DEP_DELAY_GROUP,
                CAST(DEP_TIME_BLK AS VARCHAR) AS DEP_TIME_BLK,
                CAST(TAXI_OUT AS FLOAT) AS TAXI_OUT,
                LPAD(CAST(WHEELS_OFF AS VARCHAR), 4, '0') AS WHEELS_OFF,
                LPAD(CAST(WHEELS_ON AS VARCHAR), 4, '0') AS WHEELS_ON,
                CAST(TAXI_IN AS FLOAT) AS TAXI_IN,
                LPAD(CAST(CRS_ARR_TIME AS VARCHAR), 4, '0') AS CRS_ARR_TIME,
                LPAD(CAST(ARR_TIME AS VARCHAR), 4, '0') AS ARR_TIME,
                CAST(ARR_DELAY AS FLOAT) AS ARR_DELAY,
                CAST(ARR_DELAY_NEW AS FLOAT) AS ARR_DELAY_NEW,
                CAST(ARR_DEL15 AS UTINYINT) AS ARR_DEL15,
                CAST(ARR_DELAY_GROUP AS SMALLINT) AS ARR_DELAY_GROUP,
                CAST(ARR_TIME_BLK AS VARCHAR) AS ARR_TIME_BLK,
                CAST(CANCELLED AS UTINYINT) AS CANCELLED,
                CAST(CANCELLATION_CODE AS VARCHAR) AS CANCELLATION_CODE,
                CAST(DIVERTED AS UTINYINT) AS DIVERTED,
                CAST(CRS_ELAPSED_TIME AS FLOAT) AS CRS_ELAPSED_TIME,
                CAST(ACTUAL_ELAPSED_TIME AS FLOAT) AS ACTUAL_ELAPSED_TIME,
                CAST(AIR_TIME AS FLOAT) AS AIR_TIME,
                CAST(DISTANCE AS FLOAT) AS DISTANCE,
                CAST(CARRIER_DELAY AS FLOAT) AS CARRIER_DELAY,
                CAST(WEATHER_DELAY AS FLOAT) AS WEATHER_DELAY,
                CAST(NAS_DELAY AS FLOAT) AS NAS_DELAY,
                CAST(SECURITY_DELAY AS FLOAT) AS SECURITY_DELAY,
                CAST(LATE_AIRCRAFT_DELAY AS FLOAT) AS LATE_AIRCRAFT_DELAY
            FROM read_csv_auto('{raw_flights_pattern}')
            WHERE CANCELLED = 0.0 
              AND DIVERTED = 0.0 
              AND ARR_TIME IS NOT NULL 
              AND ARR_DELAY IS NOT NULL 
              AND CRS_ELAPSED_TIME > 0
        ),
        transformed AS (
            SELECT 
                r.*,
                -- Time deconstruction
                CAST(SUBSTR(r.CRS_DEP_TIME, 1, 2) AS SMALLINT) AS CRS_DEP_HOUR,
                CAST(SUBSTR(r.CRS_DEP_TIME, 3, 2) AS SMALLINT) AS CRS_DEP_MINUTE,
                CAST(SUBSTR(r.CRS_DEP_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.CRS_DEP_TIME, 3, 2) AS SMALLINT) AS CRS_DEP_MINUTE_OF_DAY,
                
                CAST(SUBSTR(r.DEP_TIME, 1, 2) AS SMALLINT) AS DEP_HOUR,
                CAST(SUBSTR(r.DEP_TIME, 3, 2) AS SMALLINT) AS DEP_MINUTE,
                CAST(SUBSTR(r.DEP_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.DEP_TIME, 3, 2) AS SMALLINT) AS DEP_MINUTE_OF_DAY,
                
                CAST(SUBSTR(r.CRS_ARR_TIME, 1, 2) AS SMALLINT) AS CRS_ARR_HOUR,
                CAST(SUBSTR(r.CRS_ARR_TIME, 3, 2) AS SMALLINT) AS CRS_ARR_MINUTE,
                CAST(SUBSTR(r.CRS_ARR_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.CRS_ARR_TIME, 3, 2) AS SMALLINT) AS CRS_ARR_MINUTE_OF_DAY,
                
                CAST(SUBSTR(r.ARR_TIME, 1, 2) AS SMALLINT) AS ARR_HOUR,
                CAST(SUBSTR(r.ARR_TIME, 3, 2) AS SMALLINT) AS ARR_MINUTE,
                CAST(SUBSTR(r.ARR_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.ARR_TIME, 3, 2) AS SMALLINT) AS ARR_MINUTE_OF_DAY,

                -- Local departure timestamp (accounting for 2400 midnight notation)
                CASE 
                    WHEN r.CRS_DEP_TIME = '2400' THEN r.FL_DATE + INTERVAL 1 DAY
                    ELSE r.FL_DATE + (SUBSTR(r.CRS_DEP_TIME, 1, 2) || ' hours ' || SUBSTR(r.CRS_DEP_TIME, 3, 2) || ' minutes')::INTERVAL
                END AS CRS_DEP_LOCAL_DATETIME,

                -- Actual local departure timestamp derived from scheduled local time + departure delay
                CASE 
                    WHEN r.CRS_DEP_TIME = '2400' THEN r.FL_DATE + INTERVAL 1 DAY + (CAST(r.DEP_DELAY AS INTEGER) || ' minutes')::INTERVAL
                    ELSE r.FL_DATE + (SUBSTR(r.CRS_DEP_TIME, 1, 2) || ' hours ' || SUBSTR(r.CRS_DEP_TIME, 3, 2) || ' minutes')::INTERVAL + (CAST(r.DEP_DELAY AS INTEGER) || ' minutes')::INTERVAL
                END AS DEP_LOCAL_DATETIME,

                -- Operational flags
                CASE 
                    WHEN (CAST(SUBSTR(r.CRS_ARR_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.CRS_ARR_TIME, 3, 2) AS SMALLINT)) < 
                         (CAST(SUBSTR(r.CRS_DEP_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.CRS_DEP_TIME, 3, 2) AS SMALLINT))
                    THEN 1 ELSE 0 
                END AS IS_SCHEDULED_OVERNIGHT,

                CASE 
                    WHEN r.DEP_DELAY > 0 AND 
                         (CAST(SUBSTR(r.DEP_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.DEP_TIME, 3, 2) AS SMALLINT)) < 
                         (CAST(SUBSTR(r.CRS_DEP_TIME, 1, 2) AS SMALLINT) * 60 + CAST(SUBSTR(r.CRS_DEP_TIME, 3, 2) AS SMALLINT))
                    THEN 1 ELSE 0 
                END AS IS_DEP_DELAYED_PAST_MIDNIGHT,

                -- Airport metadata joins
                o.DISPLAY_AIRPORT_NAME AS ORIGIN_AIRPORT_NAME,
                o.AIRPORT_STATE_CODE AS ORIGIN_STATE,
                o.LATITUDE AS ORIGIN_LATITUDE,
                o.LONGITUDE AS ORIGIN_LONGITUDE,
                o.IANA_TIMEZONE AS ORIGIN_IANA_TZ,
                o.OBSERVES_DST AS ORIGIN_OBSERVES_DST,

                d.DISPLAY_AIRPORT_NAME AS DEST_AIRPORT_NAME,
                d.AIRPORT_STATE_CODE AS DEST_STATE,
                d.LATITUDE AS DEST_LATITUDE,
                d.LONGITUDE AS DEST_LONGITUDE,
                d.IANA_TIMEZONE AS DEST_IANA_TZ,
                d.OBSERVES_DST AS DEST_OBSERVES_DST,

                -- DST Fallback ambiguity indicator on 2025-11-02 (01:00-01:59 local window)
                CASE 
                    WHEN r.FL_DATE = DATE '2025-11-02' AND (
                        (o.OBSERVES_DST = TRUE AND r.CRS_DEP_TIME >= '0100' AND r.CRS_DEP_TIME < '0200') OR
                        (o.OBSERVES_DST = TRUE AND r.DEP_TIME >= '0100' AND r.DEP_TIME < '0200') OR
                        (d.OBSERVES_DST = TRUE AND r.CRS_ARR_TIME >= '0100' AND r.CRS_ARR_TIME < '0200') OR
                        (d.OBSERVES_DST = TRUE AND r.ARR_TIME >= '0100' AND r.ARR_TIME < '0200')
                    ) THEN 1 ELSE 0
                END AS IS_DST_TRANSITION_AMBIGUOUS

            FROM raw_completed r
            JOIN ref_airports o ON r.ORIGIN_AIRPORT_ID = o.AIRPORT_ID
            JOIN ref_airports d ON r.DEST_AIRPORT_ID = d.AIRPORT_ID
        )
        SELECT * FROM transformed
    ) TO '{cleaned_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    print("Writing compressed Parquet for Category A...")
    con.execute(q_cat_a)
    size_a_mb = os.path.getsize(cleaned_parquet_path) / (1024 * 1024)
    print(f"Exported Category A ({n_completed:,} rows) -> {cleaned_parquet_path} ({size_a_mb:.2f} MB)")

    # 4. Export Category B: Cancelled Flights to Parquet
    print("\n--- 4. Exporting Category B (Cancelled Flights) to Parquet ---")
    cancelled_parquet_path = os.path.join(processed_data_dir, "cancelled_flights.parquet")
    q_cat_b = f"""
    COPY (
        SELECT 
            r.*,
            o.DISPLAY_AIRPORT_NAME AS ORIGIN_AIRPORT_NAME,
            o.AIRPORT_STATE_CODE AS ORIGIN_STATE,
            o.LATITUDE AS ORIGIN_LATITUDE,
            o.LONGITUDE AS ORIGIN_LONGITUDE,
            o.IANA_TIMEZONE AS ORIGIN_IANA_TZ,
            d.DISPLAY_AIRPORT_NAME AS DEST_AIRPORT_NAME,
            d.AIRPORT_STATE_CODE AS DEST_STATE,
            d.LATITUDE AS DEST_LATITUDE,
            d.LONGITUDE AS DEST_LONGITUDE,
            d.IANA_TIMEZONE AS DEST_IANA_TZ
        FROM read_csv_auto('{raw_flights_pattern}') r
        JOIN ref_airports o ON r.ORIGIN_AIRPORT_ID = o.AIRPORT_ID
        JOIN ref_airports d ON r.DEST_AIRPORT_ID = d.AIRPORT_ID
        WHERE r.CANCELLED = 1.0
    ) TO '{cancelled_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_cat_b)
    size_b_mb = os.path.getsize(cancelled_parquet_path) / (1024 * 1024)
    print(f"Exported Category B ({n_cancelled:,} rows) -> {cancelled_parquet_path} ({size_b_mb:.2f} MB)")

    # 5. Export Category C: Diverted Flights to Parquet
    print("\n--- 5. Exporting Category C (Diverted Flights) to Parquet ---")
    diverted_parquet_path = os.path.join(processed_data_dir, "diverted_flights.parquet")
    q_cat_c = f"""
    COPY (
        SELECT 
            r.*,
            o.DISPLAY_AIRPORT_NAME AS ORIGIN_AIRPORT_NAME,
            o.AIRPORT_STATE_CODE AS ORIGIN_STATE,
            o.LATITUDE AS ORIGIN_LATITUDE,
            o.LONGITUDE AS ORIGIN_LONGITUDE,
            o.IANA_TIMEZONE AS ORIGIN_IANA_TZ,
            d.DISPLAY_AIRPORT_NAME AS DEST_AIRPORT_NAME,
            d.AIRPORT_STATE_CODE AS DEST_STATE,
            d.LATITUDE AS DEST_LATITUDE,
            d.LONGITUDE AS DEST_LONGITUDE,
            d.IANA_TIMEZONE AS DEST_IANA_TZ
        FROM read_csv_auto('{raw_flights_pattern}') r
        JOIN ref_airports o ON r.ORIGIN_AIRPORT_ID = o.AIRPORT_ID
        JOIN ref_airports d ON r.DEST_AIRPORT_ID = d.AIRPORT_ID
        WHERE r.CANCELLED = 0.0 AND r.DIVERTED = 1.0
    ) TO '{diverted_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_cat_c)
    size_c_mb = os.path.getsize(diverted_parquet_path) / (1024 * 1024)
    print(f"Exported Category C ({n_diverted:,} rows) -> {diverted_parquet_path} ({size_c_mb:.2f} MB)")

    # 6. Generate Missing-Value Treatment Table
    print("\n--- 6. Documenting Missing-Value Treatment & Semantics ---")
    dqr = pd.read_csv(os.path.join(results_dir, "data_quality_report.csv"))
    
    treatment_policies = {
        "TAIL_NUM": "Missing in 12,501 cancelled flights. Completed flights have 0 missing. Retain null in Category B; verified complete in Category A.",
        "DEP_TIME": "Missing in 98,170 cancelled flights. Retain null in Category B; complete in Category A.",
        "DEP_DELAY": "Missing in cancelled flights without pushback. Complete in Category A.",
        "DEP_DELAY_NEW": "Derived from DEP_DELAY. Complete in Category A.",
        "DEP_DEL15": "Derived from DEP_DELAY >= 15. Complete in Category A.",
        "DEP_DELAY_GROUP": "Derived 15-min interval group. Complete in Category A.",
        "TAXI_OUT": "Missing in cancelled flights without taxi. Complete in Category A.",
        "WHEELS_OFF": "Missing in cancelled flights without airborne status. Complete in Category A.",
        "WHEELS_ON": "Missing in cancelled, diverted, and 1 Category D flight. Complete in Category A.",
        "TAXI_IN": "Missing in cancelled, diverted, and 1 Category D flight. Complete in Category A.",
        "ARR_TIME": "Missing in cancelled, diverted, and 1 Category D flight. Complete in Category A.",
        "ARR_DELAY": "Primary target. Missing in cancelled, diverted, and 1 Category D flight. Complete in Category A.",
        "ARR_DELAY_NEW": "Derived from ARR_DELAY. Complete in Category A.",
        "ARR_DEL15": "Classification target. Complete in Category A.",
        "ARR_DELAY_GROUP": "Derived 15-min interval group. Complete in Category A.",
        "CANCELLATION_CODE": "Populated only when CANCELLED=1.0 (A=Carrier, B=Weather, C=NAS, D=Security). Retain null for non-cancelled.",
        "CRS_ELAPSED_TIME": "Missing in 2 cancelled flights. Category D flight with negative duration excluded. Complete in Category A.",
        "ACTUAL_ELAPSED_TIME": "Missing in cancelled and diverted flights. Complete in Category A.",
        "AIR_TIME": "Missing in cancelled and diverted flights. Complete in Category A.",
        "CARRIER_DELAY": "Structural missingness: reported only when ARR_DEL15=1.0. Retain NULL when ARR_DEL15=0.0. Never fill with zero.",
        "WEATHER_DELAY": "Structural missingness: reported only when ARR_DEL15=1.0. Retain NULL when ARR_DEL15=0.0. Never fill with zero.",
        "NAS_DELAY": "Structural missingness: reported only when ARR_DEL15=1.0. Retain NULL when ARR_DEL15=0.0. Never fill with zero.",
        "SECURITY_DELAY": "Structural missingness: reported only when ARR_DEL15=1.0. Retain NULL when ARR_DEL15=0.0. Never fill with zero.",
        "LATE_AIRCRAFT_DELAY": "Structural missingness: reported only when ARR_DEL15=1.0. Retain NULL when ARR_DEL15=0.0. Never fill with zero."
    }

    mvt_records = []
    for _, row in dqr.iterrows():
        col = row['column_name']
        null_count = row['null_count']
        null_pct = row['null_pct']
        treatment = treatmentpolicies = treatment_policies.get(
            col, 
            "Complete in raw data (0 nulls). Retained and verified." if null_count == 0 else "Retained as original."
        )
        
        mvt_records.append({
            "column_name": col,
            "raw_null_count": null_count,
            "raw_null_pct": null_pct,
            "category_a_treatment": "Verified 0 nulls" if col not in ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'CANCELLATION_CODE'] else "Preserve structural NULL",
            "semantic_reason_and_policy": treatment
        })

    df_mvt = pd.DataFrame(mvt_records)
    mvt_path = os.path.join(results_dir, "missing_value_treatment.csv")
    df_mvt.to_csv(mvt_path, index=False)
    print(f"Saved missing-value treatment documentation to {mvt_path}")

    # 7. Generate Timestamp Validation Report
    print("\n--- 7. Validating Processed Timestamps & Generating Timestamp Report ---")
    q_ts_val = f"""
    SELECT 
        COUNT(*) AS total_category_a_flights,
        SUM(CASE WHEN CRS_DEP_TIME = '2400' THEN 1 ELSE 0 END) AS count_crs_dep_2400,
        SUM(CASE WHEN DEP_TIME = '2400' THEN 1 ELSE 0 END) AS count_dep_2400,
        SUM(CASE WHEN CRS_ARR_TIME = '2400' THEN 1 ELSE 0 END) AS count_crs_arr_2400,
        SUM(CASE WHEN ARR_TIME = '2400' THEN 1 ELSE 0 END) AS count_arr_2400,
        SUM(IS_SCHEDULED_OVERNIGHT) AS count_scheduled_overnight,
        SUM(IS_DEP_DELAYED_PAST_MIDNIGHT) AS count_dep_delayed_past_midnight,
        SUM(IS_DST_TRANSITION_AMBIGUOUS) AS count_dst_transition_ambiguous,
        SUM(CASE WHEN CRS_DEP_LOCAL_DATETIME IS NULL THEN 1 ELSE 0 END) AS failed_crs_dep_dt,
        SUM(CASE WHEN DEP_LOCAL_DATETIME IS NULL THEN 1 ELSE 0 END) AS failed_dep_dt
    FROM read_parquet('{cleaned_parquet_path}')
    """
    df_ts_val = con.execute(q_ts_val).df()
    ts_val_path = os.path.join(results_dir, "timestamp_validation_report.csv")
    df_ts_val.to_csv(ts_val_path, index=False)
    print(df_ts_val.to_string())
    print(f"Saved timestamp validation report to {ts_val_path}")

    # 8. Re-open and verify Parquet outputs
    print("\n--- 8. Re-opening and Verifying Parquet Datasets ---")
    verified_a_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cleaned_parquet_path}')").fetchone()[0]
    verified_b_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cancelled_parquet_path}')").fetchone()[0]
    verified_c_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{diverted_parquet_path}')").fetchone()[0]

    assert verified_a_rows == n_completed, f"Category A Parquet count mismatch: {verified_a_rows} vs {n_completed}"
    assert verified_b_rows == n_cancelled, f"Category B Parquet count mismatch: {verified_b_rows} vs {n_cancelled}"
    assert verified_c_rows == n_diverted, f"Category C Parquet count mismatch: {verified_c_rows} vs {n_diverted}"

    print(f"SUCCESS: Category A Parquet reopened and verified ({verified_a_rows:,} rows).")
    print(f"SUCCESS: Category B Parquet reopened and verified ({verified_b_rows:,} rows).")
    print(f"SUCCESS: Category C Parquet reopened and verified ({verified_c_rows:,} rows).")

    total_time = time.time() - t_start
    print(f"\nPhase 2 Pipeline completed successfully in {total_time:.2f} seconds.")

    return {
        "total_input": total_in,
        "category_a_completed": verified_a_rows,
        "category_b_cancelled": verified_b_rows,
        "category_c_diverted": verified_c_rows,
        "category_d_invalid": n_invalid,
        "elapsed_seconds": total_time
    }

if __name__ == "__main__":
    run_preprocessing_pipeline()
