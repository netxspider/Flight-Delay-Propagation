"""
Phase 1: Dataset Audit and Ingestion Validation
Author: Flight Delay Propagation Research Project
"""

import os
import glob
import json
import time
import duckdb
import pandas as pd
import numpy as np

def run_dataset_audit(data_dir="data/raw", results_dir="results/tables"):
    """
    Performs comprehensive audit across all 12 monthly flight datasets
    and the Master Coordinate table.
    """
    os.makedirs(results_dir, exist_ok=True)
    con = duckdb.connect()

    monthly_files = sorted(glob.glob(os.path.join(data_dir, "2025", "*.csv")))
    master_coord_file = os.path.join(data_dir, "T_MASTER_CORD.csv")

    print(f"Found {len(monthly_files)} monthly files in {os.path.join(data_dir, '2025')}.")
    print(f"Master coordinate file: {master_coord_file}")

    # Month ordering mapping
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    # 1. Inspect Schema consistency
    print("\n--- 1. Checking Schema Consistency across 12 months ---")
    schemas = {}
    file_stats = []

    for fpath in monthly_files:
        fname = os.path.basename(fpath)
        # extract month tag
        month_str = fname.replace("T_ONTIME_REPORTING-", "").replace(".csv", "")
        m_num = month_map.get(month_str, 0)
        file_size_mb = os.path.getsize(fpath) / (1024 * 1024)

        desc = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{fpath}')").df()
        schemas[fname] = list(zip(desc["column_name"], desc["column_type"]))

        # Basic counts per month
        q = f"""
        SELECT 
            COUNT(*) AS total_rows,
            MIN(FL_DATE) AS min_fl_date,
            MAX(FL_DATE) AS max_fl_date,
            COUNT(DISTINCT OP_UNIQUE_CARRIER) AS unique_carriers,
            COUNT(DISTINCT TAIL_NUM) AS unique_tails,
            SUM(CASE WHEN TAIL_NUM IS NULL OR TAIL_NUM = '' THEN 1 ELSE 0 END) AS missing_tails,
            SUM(CANCELLED) AS cancelled_flights,
            SUM(DIVERTED) AS diverted_flights,
            SUM(CASE WHEN DEP_DELAY IS NULL THEN 1 ELSE 0 END) AS missing_dep_delay,
            SUM(CASE WHEN ARR_DELAY IS NULL THEN 1 ELSE 0 END) AS missing_arr_delay,
            SUM(CASE WHEN ARR_DEL15 = 1.0 THEN 1 ELSE 0 END) AS arr_del15_flights,
            COUNT(DISTINCT ORIGIN_AIRPORT_ID) AS unique_origins,
            COUNT(DISTINCT DEST_AIRPORT_ID) AS unique_dests
        FROM read_csv_auto('{fpath}')
        """
        row_stat = con.execute(q).df().iloc[0].to_dict()
        row_stat["file_name"] = fname
        row_stat["month_name"] = month_str
        row_stat["month_num"] = m_num
        row_stat["file_size_mb"] = round(file_size_mb, 2)
        row_stat["num_columns"] = len(desc)
        file_stats.append(row_stat)

    # Verify if all schemas match January
    ref_file = list(schemas.keys())[0]
    ref_schema = schemas[ref_file]
    all_schemas_match = True
    for fname, sch in schemas.items():
        if sch != ref_schema:
            all_schemas_match = False
            print(f"WARNING: Schema mismatch in {fname} compared to {ref_file}!")

    if all_schemas_match:
        print(f"SUCCESS: All 12 monthly CSV files have 100% IDENTICAL schemas ({len(ref_schema)} columns)!")
    
    # Sort file stats chronologically
    df_monthly = pd.DataFrame(file_stats).sort_values("month_num").reset_index(drop=True)
    
    # Compute derived rate columns
    df_monthly["cancel_rate_pct"] = (df_monthly["cancelled_flights"] / df_monthly["total_rows"] * 100).round(2)
    df_monthly["divert_rate_pct"] = (df_monthly["diverted_flights"] / df_monthly["total_rows"] * 100).round(2)
    df_monthly["missing_tail_pct"] = (df_monthly["missing_tails"] / df_monthly["total_rows"] * 100).round(2)
    
    # Delay rate among completed (non-cancelled, non-diverted) flights
    df_monthly["completed_flights"] = df_monthly["total_rows"] - df_monthly["cancelled_flights"] - df_monthly["diverted_flights"]
    df_monthly["arr_del15_rate_pct"] = (df_monthly["arr_del15_flights"] / df_monthly["completed_flights"] * 100).round(2)

    monthly_summary_path = os.path.join(results_dir, "monthly_summary.csv")
    df_monthly.to_csv(monthly_summary_path, index=False)
    print(f"Saved monthly summary to {monthly_summary_path}")

    # 2. Arrival Delay Distribution across 2025 (completed flights)
    print("\n--- 2. Computing 2025 Annual Arrival Delay Distribution (Completed Flights) ---")
    delay_stats_q = f"""
    SELECT 
        COUNT(*) AS completed_count,
        ROUND(AVG(ARR_DELAY), 2) AS mean_arr_delay,
        ROUND(MEDIAN(ARR_DELAY), 2) AS median_arr_delay,
        ROUND(STDDEV(ARR_DELAY), 2) AS std_arr_delay,
        MIN(ARR_DELAY) AS min_arr_delay,
        ROUND(QUANTILE_CONT(ARR_DELAY, 0.25), 2) AS p25_arr_delay,
        ROUND(QUANTILE_CONT(ARR_DELAY, 0.75), 2) AS p75_arr_delay,
        ROUND(QUANTILE_CONT(ARR_DELAY, 0.90), 2) AS p90_arr_delay,
        ROUND(QUANTILE_CONT(ARR_DELAY, 0.95), 2) AS p95_arr_delay,
        ROUND(QUANTILE_CONT(ARR_DELAY, 0.99), 2) AS p99_arr_delay,
        MAX(ARR_DELAY) AS max_arr_delay,
        SUM(CASE WHEN ARR_DELAY <= 0 THEN 1 ELSE 0 END) AS ontime_or_early_count,
        ROUND(SUM(CASE WHEN ARR_DELAY <= 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS ontime_or_early_pct,
        SUM(CASE WHEN ARR_DEL15 = 1 THEN 1 ELSE 0 END) AS delayed_15plus_count,
        ROUND(SUM(CASE WHEN ARR_DEL15 = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS delayed_15plus_pct
    FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
    WHERE CANCELLED = 0 AND DIVERTED = 0 AND ARR_DELAY IS NOT NULL
    """
    df_arr_delay_dist = con.execute(delay_stats_q).df()
    delay_dist_path = os.path.join(results_dir, "annual_arrival_delay_distribution.csv")
    df_arr_delay_dist.to_csv(delay_dist_path, index=False)
    print(df_arr_delay_dist.to_string())

    # 3. Overall Duplicate and Overlap checks
    print("\n--- 3. Checking for Duplicates and Data Integrity ---")
    dup_q = f"""
    SELECT 
        COUNT(*) - COUNT(DISTINCT (
            FL_DATE, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, 
            ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, CRS_DEP_TIME
        )) AS potential_duplicate_keys
    FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
    """
    dup_res = con.execute(dup_q).fetchone()[0]
    print(f"Flight key duplicate count (FL_DATE, CARRIER, FL_NUM, ORIGIN, DEST, CRS_DEP_TIME): {dup_res}")

    # Exact row duplicate check
    exact_dup_q = f"""
    SELECT COUNT(*) - COUNT(DISTINCT (
        YEAR, MONTH, DAY_OF_MONTH, FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM,
        OP_CARRIER_FL_NUM, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, CRS_DEP_TIME, DEP_TIME,
        CRS_ARR_TIME, ARR_TIME, ARR_DELAY
    )) AS exact_core_duplicates
    FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
    """
    exact_dups = con.execute(exact_dup_q).fetchone()[0]
    print(f"Exact core record duplicate count: {exact_dups}")

    # 4. Master Coordinate Table Validation
    print("\n--- 4. Validating Master Coordinate Table ---")
    mc_q = f"""
    SELECT 
        COUNT(*) AS total_mc_rows,
        COUNT(DISTINCT AIRPORT_ID) AS unique_airport_ids,
        COUNT(DISTINCT AIRPORT) AS unique_airport_codes,
        SUM(CASE WHEN AIRPORT_IS_LATEST = 1 THEN 1 ELSE 0 END) AS latest_airport_count,
        SUM(CASE WHEN AIRPORT_IS_CLOSED = 1 THEN 1 ELSE 0 END) AS closed_airport_count,
        SUM(CASE WHEN LATITUDE IS NULL THEN 1 ELSE 0 END) AS missing_latitude,
        SUM(CASE WHEN LONGITUDE IS NULL THEN 1 ELSE 0 END) AS missing_longitude,
        SUM(CASE WHEN UTC_LOCAL_TIME_VARIATION IS NULL OR UTC_LOCAL_TIME_VARIATION = '' THEN 1 ELSE 0 END) AS missing_utc_variation
    FROM read_csv_auto('{master_coord_file}')
    """
    df_mc = con.execute(mc_q).df()
    print(df_mc.to_string())

    # Airport Coverage check: Do all origin/dest airports in 2025 flights exist in Master Coordinate?
    print("\n--- 5. Checking Airport ID Matching between 2025 Flights and Master Coordinates ---")
    coverage_q = f"""
    WITH flight_airports AS (
        SELECT DISTINCT ORIGIN_AIRPORT_ID AS airport_id, ORIGIN AS code FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
        UNION
        SELECT DISTINCT DEST_AIRPORT_ID AS airport_id, DEST AS code FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
    ),
    master_latest AS (
        SELECT * FROM read_csv_auto('{master_coord_file}') WHERE AIRPORT_IS_LATEST = 1
    )
    SELECT 
        f.airport_id,
        f.code AS flight_iata_code,
        m.AIRPORT AS mc_code,
        m.DISPLAY_AIRPORT_NAME,
        m.DISPLAY_AIRPORT_CITY_NAME_FULL,
        m.LATITUDE,
        m.LONGITUDE,
        m.UTC_LOCAL_TIME_VARIATION,
        CASE WHEN m.AIRPORT_ID IS NOT NULL THEN 1 ELSE 0 END AS matched_in_master
    FROM flight_airports f
    LEFT JOIN master_latest m ON f.airport_id = m.AIRPORT_ID
    ORDER BY f.code
    """
    df_coverage = con.execute(coverage_q).df()
    unmatched_airports = df_coverage[df_coverage["matched_in_master"] == 0]
    missing_coords = df_coverage[df_coverage["LATITUDE"].isna()]
    
    print(f"Total distinct active airport IDs in 2025 flights: {len(df_coverage)}")
    print(f"Airports matched in Master Coordinate (LATEST=1): {df_coverage['matched_in_master'].sum()} / {len(df_coverage)}")
    print(f"Unmatched airports: {len(unmatched_airports)}")
    print(f"Airports with missing coordinates: {len(missing_coords)}")

    coverage_path = os.path.join(results_dir, "airport_coverage_report.csv")
    df_coverage.to_csv(coverage_path, index=False)
    print(f"Saved airport coverage report to {coverage_path}")

    # 6. Detailed 46-Column Data Quality & Missingness Report across full 2025 dataset
    print("\n--- 6. Generating Full 46-Column Data Quality Report ---")
    # Build dynamic SQL query for each column
    col_names = [col[0] for col in ref_schema]
    col_types = [col[1] for col in ref_schema]

    select_clauses = []
    for col in col_names:
        select_clauses.append(f"""
            SUM(CASE WHEN "{col}" IS NULL OR CAST("{col}" AS VARCHAR) = '' THEN 1 ELSE 0 END) AS "{col}__nulls",
            COUNT(DISTINCT "{col}") AS "{col}__distinct"
        """)
    
    full_audit_sql = f"""
    SELECT 
        COUNT(*) AS total_rows,
        {",".join(select_clauses)}
    FROM read_csv_auto('{os.path.join(data_dir, "2025", "*.csv")}')
    """
    print("Executing comprehensive columnar null & distinct scan across 7+ million rows...")
    t0 = time.time()
    audit_res = con.execute(full_audit_sql).df().iloc[0].to_dict()
    print(f"Scan finished in {time.time() - t0:.2f} seconds.")

    total_rows = audit_res["total_rows"]
    col_records = []
    for col, col_type in zip(col_names, col_types):
        null_count = audit_res[f"{col}__nulls"]
        distinct_count = audit_res[f"{col}__distinct"]
        null_pct = round(null_count / total_rows * 100, 3)
        valid_count = total_rows - null_count
        
        # Sample value
        sample_val = con.execute(f"""
            SELECT "{col}" FROM read_csv_auto('{monthly_files[0]}')
            WHERE "{col}" IS NOT NULL AND CAST("{col}" AS VARCHAR) != ''
            LIMIT 1
        """).fetchone()
        sample_str = str(sample_val[0]) if sample_val else ""

        col_records.append({
            "column_name": col,
            "data_type": col_type,
            "total_rows": total_rows,
            "valid_count": valid_count,
            "null_count": null_count,
            "null_pct": null_pct,
            "distinct_count": distinct_count,
            "sample_value": sample_str
        })

    df_dqr = pd.DataFrame(col_records)
    dqr_path = os.path.join(results_dir, "data_quality_report.csv")
    df_dqr.to_csv(dqr_path, index=False)
    print(f"Saved data quality report to {dqr_path}")

    # 7. Create Schema Documentation
    print("\n--- 7. Generating Comprehensive Schema Documentation ---")
    # Add semantic roles and descriptions
    field_metadata = {
        "YEAR": ("Flight operation year (2025)", "Temporal", "Keep"),
        "QUARTER": ("Flight calendar quarter (1-4)", "Temporal", "Keep"),
        "MONTH": ("Flight calendar month (1-12)", "Temporal", "Keep"),
        "DAY_OF_MONTH": ("Day of month (1-31)", "Temporal", "Keep"),
        "DAY_OF_WEEK": ("Day of week (1=Monday, 7=Sunday)", "Temporal", "Feature Candidate"),
        "FL_DATE": ("Flight operating date timestamp", "Temporal", "Core Identifier/Sorting"),
        "OP_UNIQUE_CARRIER": ("Reporting airline 2-letter IATA code", "Categorical", "Feature Candidate"),
        "OP_CARRIER_AIRLINE_ID": ("Unique BTS DOT airline identifier", "Categorical", "Alternative Airline ID"),
        "TAIL_NUM": ("Aircraft registration tail number", "Identifier", "Core Entity for Rotation"),
        "OP_CARRIER_FL_NUM": ("Flight operational number", "Identifier", "Flight Key"),
        "ORIGIN_AIRPORT_ID": ("Origin airport DOT identifier", "Geographic", "Core Rotation Join Key"),
        "ORIGIN": ("Origin airport 3-letter IATA code", "Geographic", "Airport Identifier"),
        "ORIGIN_CITY_NAME": ("Origin airport city & state name", "Geographic", "Descriptive"),
        "DEST_AIRPORT_ID": ("Destination airport DOT identifier", "Geographic", "Core Rotation Join Key"),
        "DEST": ("Destination airport 3-letter IATA code", "Geographic", "Airport Identifier"),
        "DEST_CITY_NAME": ("Destination airport city & state name", "Geographic", "Descriptive"),
        "CRS_DEP_TIME": ("Scheduled local departure time (HHMM)", "Temporal", "Scheduled Feature"),
        "DEP_TIME": ("Actual local departure time (HHMM)", "Temporal", "Turnaround Reconstruction"),
        "DEP_DELAY": ("Departure delay in minutes", "Numerical", "Turnaround / Preceding Delay"),
        "DEP_DELAY_NEW": ("Departure delay truncated at zero (non-negative)", "Numerical", "Derived"),
        "DEP_DEL15": ("Departure delay >= 15 min indicator (1/0)", "Binary", "Derived"),
        "DEP_DELAY_GROUP": ("Departure delay grouped in 15-min intervals", "Ordinal", "Descriptive"),
        "DEP_TIME_BLK": ("Scheduled departure time block (e.g. 0600-0659)", "Categorical", "Scheduled Feature"),
        "TAXI_OUT": ("Taxi-out ground time in minutes", "Numerical", "Ground Operation"),
        "WHEELS_OFF": ("Actual local wheels-off time (HHMM)", "Temporal", "Flight Lifecycle"),
        "WHEELS_ON": ("Actual local wheels-on time (HHMM)", "Temporal", "Flight Lifecycle"),
        "TAXI_IN": ("Taxi-in ground time in minutes", "Numerical", "Ground Operation"),
        "CRS_ARR_TIME": ("Scheduled local arrival time (HHMM)", "Temporal", "Scheduled Feature"),
        "ARR_TIME": ("Actual local arrival time (HHMM)", "Temporal", "Turnaround Reconstruction"),
        "ARR_DELAY": ("Arrival delay in minutes (can be negative)", "Numerical", "Prediction Target / Prior Delay"),
        "ARR_DELAY_NEW": ("Arrival delay truncated at zero (non-negative)", "Numerical", "Derived"),
        "ARR_DEL15": ("Arrival delay >= 15 min indicator (1/0)", "Binary", "Classification Target"),
        "ARR_DELAY_GROUP": ("Arrival delay grouped in 15-min intervals", "Ordinal", "Descriptive"),
        "ARR_TIME_BLK": ("Scheduled arrival time block", "Categorical", "Scheduled Feature"),
        "CANCELLED": ("Flight cancellation indicator (1.0 = cancelled)", "Binary", "Exclusion Filter"),
        "CANCELLATION_CODE": ("Reason for cancellation (A, B, C, D)", "Categorical", "Descriptive / Nullable"),
        "DIVERTED": ("Flight diversion indicator (1.0 = diverted)", "Binary", "Exclusion Filter"),
        "CRS_ELAPSED_TIME": ("Scheduled gate-to-gate elapsed time in minutes", "Numerical", "Scheduled Feature"),
        "ACTUAL_ELAPSED_TIME": ("Actual gate-to-gate elapsed time in minutes", "Numerical", "Flight Operation"),
        "AIR_TIME": ("Airborne time in minutes (wheels-off to wheels-on)", "Numerical", "Flight Operation"),
        "DISTANCE": ("Great-circle flight distance in statute miles", "Numerical", "Scheduled Feature"),
        "CARRIER_DELAY": ("Carrier-caused delay in minutes (retrospective)", "Numerical Delay Cause", "Retrospective / Banned in Pre-dep Model"),
        "WEATHER_DELAY": ("Weather-caused delay in minutes (retrospective)", "Numerical Delay Cause", "Retrospective / Banned in Pre-dep Model"),
        "NAS_DELAY": ("National Airspace System delay minutes (retrospective)", "Numerical Delay Cause", "Retrospective / Banned in Pre-dep Model"),
        "SECURITY_DELAY": ("Security delay in minutes (retrospective)", "Numerical Delay Cause", "Retrospective / Banned in Pre-dep Model"),
        "LATE_AIRCRAFT_DELAY": ("Delay attributed to late aircraft arrival (retrospective)", "Numerical Delay Cause", "Retrospective / Banned in Pre-dep Model")
    }

    df_dqr["description"] = df_dqr["column_name"].map(lambda x: field_metadata.get(x, ("", "", ""))[0])
    df_dqr["variable_category"] = df_dqr["column_name"].map(lambda x: field_metadata.get(x, ("", "", ""))[1])
    df_dqr["pipeline_role"] = df_dqr["column_name"].map(lambda x: field_metadata.get(x, ("", "", ""))[2])

    schema_doc_path = os.path.join(results_dir, "schema_documentation.csv")
    df_dqr.to_csv(schema_doc_path, index=False)
    print(f"Saved schema documentation to {schema_doc_path}")

    print("\n================ AUDIT SUMMARY ================")
    print(f"Total Flights Ingested: {total_rows:,}")
    print(f"Total Monthly Files: {len(monthly_files)}")
    print(f"Total Columns: {len(col_names)}")
    print(f"All monthly schemas identical: {all_schemas_match}")
    print("===============================================")

    return {
        "monthly_summary": df_monthly,
        "arrival_delay_distribution": df_arr_delay_dist,
        "airport_coverage": df_coverage,
        "data_quality_report": df_dqr,
        "total_rows": total_rows
    }

if __name__ == "__main__":
    run_dataset_audit()

