"""
Timestamp Normalization and Aircraft Operational Timeline Construction
Phase 3: Aircraft Operational Timeline and Rotation Reconstruction
Author: Flight Delay Propagation Research Project
"""

import os
import sys
import time
import duckdb
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def build_aircraft_timeline(
    cleaned_parquet="data/processed/flights_cleaned.parquet",
    diverted_parquet="data/processed/diverted_flights.parquet",
    dst_res_csv="results/tables/dst_ambiguity_resolution.csv",
    output_timeline_parquet="data/processed/aircraft_timeline.parquet"
):
    """
    Normalizes local flight timestamps to UTC and unifies completed flights
    and diverted operations into a single chronological aircraft operational timeline.
    """
    t_start = time.time()
    os.makedirs(os.path.dirname(output_timeline_parquet), exist_ok=True)
    con = duckdb.connect()

    print("=========================================================")
    print("PHASE 3: AIRCRAFT OPERATIONAL TIMELINE NORMALIZATION")
    print("=========================================================")

    # 1. Load DST resolution mappings to flag unresolved records
    print("Loading DST resolution status...")
    df_dst = pd.read_csv(dst_res_csv)
    unresolved_dst = df_dst[df_dst['resolution_status'] == 'PRESERVED_AMBIGUOUS_FLAG']
    con.register("dst_unresolved", unresolved_dst)
    print(f"Registered {len(unresolved_dst)} preserved ambiguous DST records.")

    # 2. Build unified operational timeline
    print("Constructing unified operational timeline in DuckDB...")
    q_timeline = f"""
    COPY (
        WITH completed_utc AS (
            SELECT 
                c.YEAR,
                c.QUARTER,
                c.MONTH,
                c.DAY_OF_MONTH,
                c.DAY_OF_WEEK,
                c.FL_DATE,
                c.OP_UNIQUE_CARRIER,
                c.OP_CARRIER_AIRLINE_ID,
                c.TAIL_NUM,
                c.OP_CARRIER_FL_NUM,
                c.ORIGIN_AIRPORT_ID,
                c.ORIGIN,
                c.ORIGIN_CITY_NAME,
                c.DEST_AIRPORT_ID,
                c.DEST,
                c.DEST_CITY_NAME,
                c.CRS_DEP_TIME,
                c.DEP_TIME,
                c.DEP_DELAY,
                c.DEP_DEL15,
                c.CRS_ARR_TIME,
                c.ARR_TIME,
                c.ARR_DELAY,
                c.ARR_DEL15,
                c.CRS_ELAPSED_TIME,
                c.ACTUAL_ELAPSED_TIME,
                c.AIR_TIME,
                c.DISTANCE,
                c.ORIGIN_AIRPORT_NAME,
                c.ORIGIN_STATE,
                c.ORIGIN_LATITUDE,
                c.ORIGIN_LONGITUDE,
                c.ORIGIN_IANA_TZ,
                c.DEST_AIRPORT_NAME,
                c.DEST_STATE,
                c.DEST_LATITUDE,
                c.DEST_LONGITUDE,
                c.DEST_IANA_TZ,
                c.CRS_DEP_LOCAL_DATETIME,
                c.DEP_LOCAL_DATETIME,
                -- UTC Timestamps
                timezone(c.ORIGIN_IANA_TZ, c.CRS_DEP_LOCAL_DATETIME) AT TIME ZONE 'UTC' AS CRS_DEP_UTC,
                timezone(c.ORIGIN_IANA_TZ, c.CRS_DEP_LOCAL_DATETIME) AT TIME ZONE 'UTC' + (CAST(c.DEP_DELAY AS INT) || ' minutes')::INTERVAL AS DEP_UTC,
                timezone(c.ORIGIN_IANA_TZ, c.CRS_DEP_LOCAL_DATETIME) AT TIME ZONE 'UTC' + (CAST(c.CRS_ELAPSED_TIME AS INT) || ' minutes')::INTERVAL AS CRS_ARR_UTC,
                timezone(c.ORIGIN_IANA_TZ, c.CRS_DEP_LOCAL_DATETIME) AT TIME ZONE 'UTC' + (CAST(c.CRS_ELAPSED_TIME AS INT) || ' minutes')::INTERVAL + (CAST(c.ARR_DELAY AS INT) || ' minutes')::INTERVAL AS ARR_UTC,
                0 AS IS_DIVERTED,
                0 AS IS_CANCELLED,
                -- Calendar Boundary Flags
                CASE WHEN c.MONTH = 1 AND c.DAY_OF_MONTH <= 2 THEN 1 ELSE 0 END AS IS_JANUARY_BOUNDARY,
                CASE WHEN c.MONTH = 12 AND c.DAY_OF_MONTH >= 30 THEN 1 ELSE 0 END AS IS_DECEMBER_BOUNDARY,
                -- DST Ambiguity Flag
                CASE WHEN u.FL_DATE IS NOT NULL THEN 1 ELSE 0 END AS IS_DST_AMBIGUOUS
            FROM read_parquet('{cleaned_parquet}') c
            LEFT JOIN dst_unresolved u 
                   ON c.FL_DATE = u.FL_DATE 
                  AND c.TAIL_NUM = u.TAIL_NUM 
                  AND c.OP_CARRIER_FL_NUM = u.FL_NUM 
                  AND c.ORIGIN = u.ORIGIN
        ),
        diverted_utc AS (
            SELECT 
                d.YEAR,
                d.QUARTER,
                d.MONTH,
                d.DAY_OF_MONTH,
                d.DAY_OF_WEEK,
                CAST(d.FL_DATE AS DATE) AS FL_DATE,
                d.OP_UNIQUE_CARRIER,
                d.OP_CARRIER_AIRLINE_ID,
                d.TAIL_NUM,
                d.OP_CARRIER_FL_NUM,
                d.ORIGIN_AIRPORT_ID,
                d.ORIGIN,
                d.ORIGIN_CITY_NAME,
                d.DEST_AIRPORT_ID,
                d.DEST,
                d.DEST_CITY_NAME,
                LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0') AS CRS_DEP_TIME,
                LPAD(CAST(d.DEP_TIME AS VARCHAR), 4, '0') AS DEP_TIME,
                CAST(d.DEP_DELAY AS FLOAT) AS DEP_DELAY,
                CAST(d.DEP_DEL15 AS UTINYINT) AS DEP_DEL15,
                LPAD(CAST(d.CRS_ARR_TIME AS VARCHAR), 4, '0') AS CRS_ARR_TIME,
                LPAD(CAST(d.ARR_TIME AS VARCHAR), 4, '0') AS ARR_TIME,
                CAST(NULL AS FLOAT) AS ARR_DELAY,
                CAST(NULL AS UTINYINT) AS ARR_DEL15,
                CAST(d.CRS_ELAPSED_TIME AS FLOAT) AS CRS_ELAPSED_TIME,
                CAST(d.ACTUAL_ELAPSED_TIME AS FLOAT) AS ACTUAL_ELAPSED_TIME,
                CAST(d.AIR_TIME AS FLOAT) AS AIR_TIME,
                CAST(d.DISTANCE AS FLOAT) AS DISTANCE,
                d.ORIGIN_AIRPORT_NAME,
                d.ORIGIN_STATE,
                d.ORIGIN_LATITUDE,
                d.ORIGIN_LONGITUDE,
                d.ORIGIN_IANA_TZ,
                d.DEST_AIRPORT_NAME,
                d.DEST_STATE,
                d.DEST_LATITUDE,
                d.DEST_LONGITUDE,
                d.DEST_IANA_TZ,
                CASE 
                    WHEN d.CRS_DEP_TIME = '2400' THEN CAST(d.FL_DATE AS DATE) + INTERVAL 1 DAY
                    ELSE CAST(d.FL_DATE AS DATE) + (SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 1, 2) || ' hours ' || SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 3, 2) || ' minutes')::INTERVAL
                END AS CRS_DEP_LOCAL_DATETIME,
                CASE 
                    WHEN d.CRS_DEP_TIME = '2400' THEN CAST(d.FL_DATE AS DATE) + INTERVAL 1 DAY + (CAST(COALESCE(d.DEP_DELAY, 0) AS INT) || ' minutes')::INTERVAL
                    ELSE CAST(d.FL_DATE AS DATE) + (SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 1, 2) || ' hours ' || SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 3, 2) || ' minutes')::INTERVAL + (CAST(COALESCE(d.DEP_DELAY, 0) AS INT) || ' minutes')::INTERVAL
                END AS DEP_LOCAL_DATETIME,
                -- Diverted UTC timestamps
                timezone(d.ORIGIN_IANA_TZ, CASE 
                    WHEN d.CRS_DEP_TIME = '2400' THEN CAST(d.FL_DATE AS DATE) + INTERVAL 1 DAY
                    ELSE CAST(d.FL_DATE AS DATE) + (SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 1, 2) || ' hours ' || SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 3, 2) || ' minutes')::INTERVAL
                END) AT TIME ZONE 'UTC' AS CRS_DEP_UTC,
                timezone(d.ORIGIN_IANA_TZ, CASE 
                    WHEN d.CRS_DEP_TIME = '2400' THEN CAST(d.FL_DATE AS DATE) + INTERVAL 1 DAY
                    ELSE CAST(d.FL_DATE AS DATE) + (SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 1, 2) || ' hours ' || SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 3, 2) || ' minutes')::INTERVAL
                END) AT TIME ZONE 'UTC' + (CAST(COALESCE(d.DEP_DELAY, 0) AS INT) || ' minutes')::INTERVAL AS DEP_UTC,
                timezone(d.ORIGIN_IANA_TZ, CASE 
                    WHEN d.CRS_DEP_TIME = '2400' THEN CAST(d.FL_DATE AS DATE) + INTERVAL 1 DAY
                    ELSE CAST(d.FL_DATE AS DATE) + (SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 1, 2) || ' hours ' || SUBSTR(LPAD(CAST(d.CRS_DEP_TIME AS VARCHAR), 4, '0'), 3, 2) || ' minutes')::INTERVAL
                END) AT TIME ZONE 'UTC' + (CAST(d.CRS_ELAPSED_TIME AS INT) || ' minutes')::INTERVAL AS CRS_ARR_UTC,
                CAST(NULL AS TIMESTAMP) AS ARR_UTC, -- No arrival timestamp at scheduled destination
                1 AS IS_DIVERTED,
                0 AS IS_CANCELLED,
                CASE WHEN d.MONTH = 1 AND d.DAY_OF_MONTH <= 2 THEN 1 ELSE 0 END AS IS_JANUARY_BOUNDARY,
                CASE WHEN d.MONTH = 12 AND d.DAY_OF_MONTH >= 30 THEN 1 ELSE 0 END AS IS_DECEMBER_BOUNDARY,
                0 AS IS_DST_AMBIGUOUS
            FROM read_parquet('{diverted_parquet}') d
        ),
        unified_timeline AS (
            SELECT * FROM completed_utc
            UNION ALL
            SELECT * FROM diverted_utc
        )
        SELECT * FROM unified_timeline
        ORDER BY TAIL_NUM, DEP_UTC
    ) TO '{output_timeline_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    print("Writing aircraft_timeline.parquet...")
    con.execute(q_timeline)
    
    size_mb = os.path.getsize(output_timeline_parquet) / (1024 * 1024)
    total_ops = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_timeline_parquet}')").fetchone()[0]
    n_completed = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_timeline_parquet}') WHERE IS_DIVERTED = 0").fetchone()[0]
    n_diverted = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_timeline_parquet}') WHERE IS_DIVERTED = 1").fetchone()[0]

    print(f"SUCCESS: Exported {total_ops:,} operations ({n_completed:,} completed, {n_diverted:,} diverted)")
    print(f"Timeline Parquet size: {size_mb:.2f} MB")
    print(f"Normalization completed in {time.time() - t_start:.2f} seconds.")

    return {
        "total_operations": total_ops,
        "completed_operations": n_completed,
        "diverted_operations": n_diverted,
        "timeline_parquet": output_timeline_parquet,
        "size_mb": size_mb
    }

if __name__ == "__main__":
    build_aircraft_timeline()
