"""
Airport Enrichment and IANA Timezone Resolution
Phase 2: Preprocessing and Data Cleaning
Author: Flight Delay Propagation Research Project
"""

import os
import duckdb
import pandas as pd
from zoneinfo import ZoneInfo

def get_iana_timezone_and_dst(row):
    """
    Maps an airport record to its authoritative IANA timezone identifier
    and determines whether the jurisdiction observes Daylight Saving Time (DST).
    """
    iata = row['AIRPORT']
    state = row['AIRPORT_STATE_CODE']
    utc_var = str(row['UTC_LOCAL_TIME_VARIATION']).strip()

    # Specific territorial and single-airport overrides
    if iata == 'ADK':
        return 'America/Adak', True
    if state == 'AK':
        return 'America/Anchorage', True
    if state == 'HI':
        return 'Pacific/Honolulu', False  # Hawaii does not observe DST
    if state == 'AZ':
        return 'America/Phoenix', False   # Arizona does not observe DST (except Navajo Nation, no commercial hubs)
    if state in ('PR', 'VI'):
        return 'America/Puerto_Rico', False  # Puerto Rico & US Virgin Islands do not observe DST
    if iata in ('GUM', 'SPN'):
        return 'Pacific/Guam', False      # Guam / Northern Mariana Islands do not observe DST
    if iata == 'PPG':
        return 'Pacific/Pago_Pago', False # American Samoa does not observe DST

    # Standard contiguous US time zones based on verified standard offset
    if utc_var == '-0800':
        return 'America/Los_Angeles', True
    elif utc_var == '-0700':
        return 'America/Denver', True
    elif utc_var == '-0600':
        return 'America/Chicago', True
    elif utc_var == '-0500':
        return 'America/New_York', True
    elif utc_var == '-0400':
        return 'America/Puerto_Rico', False
    elif utc_var == '+1000':
        return 'Pacific/Guam', False
    elif utc_var == '-1100':
        return 'Pacific/Pago_Pago', False
    elif utc_var == '-1000':
        return 'Pacific/Honolulu', False
    else:
        raise ValueError(f"Unresolved airport timezone for IATA: {iata}, State: {state}, Offset: {utc_var}")

def build_airport_reference(raw_data_dir="data/raw", results_dir="results/tables", con=None):
    """
    Builds and validates the authoritative 352-airport enrichment reference table
    with coordinates, IANA time zones, and DST observation status.
    """
    if con is None:
        con = duckdb.connect()

    os.makedirs(results_dir, exist_ok=True)
    master_cord_path = os.path.join(raw_data_dir, "T_MASTER_CORD.csv")
    flights_pattern = os.path.join(raw_data_dir, "2025", "*.csv")

    # 1. Identify active commercial airport IDs operating in 2025 flights
    q_active = f"""
    WITH flight_airports AS (
        SELECT DISTINCT ORIGIN_AIRPORT_ID as airport_id, ORIGIN as iata_code FROM read_csv_auto('{flights_pattern}')
        UNION
        SELECT DISTINCT DEST_AIRPORT_ID as airport_id, DEST as iata_code FROM read_csv_auto('{flights_pattern}')
    ),
    master_latest AS (
        SELECT * FROM read_csv_auto('{master_cord_path}') WHERE AIRPORT_IS_LATEST = 1
    )
    SELECT 
        m.AIRPORT_ID,
        m.AIRPORT_SEQ_ID,
        m.AIRPORT,
        m.DISPLAY_AIRPORT_NAME,
        m.DISPLAY_AIRPORT_CITY_NAME_FULL,
        m.AIRPORT_COUNTRY_CODE_ISO,
        m.AIRPORT_STATE_CODE,
        m.CITY_MARKET_ID,
        m.LATITUDE,
        m.LONGITUDE,
        m.UTC_LOCAL_TIME_VARIATION
    FROM flight_airports f
    JOIN master_latest m ON f.airport_id = m.AIRPORT_ID
    ORDER BY m.AIRPORT_STATE_CODE, m.AIRPORT
    """
    df_airports = con.execute(q_active).df()

    # 2. Enrich with IANA timezone and DST observation flag
    tz_info = df_airports.apply(get_iana_timezone_and_dst, axis=1)
    df_airports['IANA_TIMEZONE'] = [t[0] for t in tz_info]
    df_airports['OBSERVES_DST'] = [t[1] for t in tz_info]

    # 3. Validation checks
    assert len(df_airports) == 352, f"Expected 352 active airports, found {len(df_airports)}"
    assert df_airports['AIRPORT_ID'].nunique() == 352, "Airport IDs must be strictly unique"
    assert df_airports['LATITUDE'].isna().sum() == 0, "Latitude must have zero missing values"
    assert df_airports['LONGITUDE'].isna().sum() == 0, "Longitude must have zero missing values"
    assert (df_airports['LATITUDE'] >= -90.0).all() and (df_airports['LATITUDE'] <= 90.0).all(), "Invalid latitude range"
    assert (df_airports['LONGITUDE'] >= -180.0).all() and (df_airports['LONGITUDE'] <= 180.0).all(), "Invalid longitude range"

    # Verify each IANA timezone against Python's ZoneInfo database
    for tz in df_airports['IANA_TIMEZONE'].unique():
        ZoneInfo(tz)  # will raise error if invalid

    # 4. Save validation table
    val_table_path = os.path.join(results_dir, "airport_enrichment_validation.csv")
    df_airports.to_csv(val_table_path, index=False)
    print(f"Validated all {len(df_airports)} active commercial airports.")
    print(f"Saved airport enrichment validation to {val_table_path}")

    # Register table inside DuckDB connection for direct SQL joins
    con.register("ref_airports", df_airports)
    return df_airports

if __name__ == "__main__":
    df = build_airport_reference()
    print("\nSample enriched airport metadata:")
    print(df[['AIRPORT_ID', 'AIRPORT', 'DISPLAY_AIRPORT_NAME', 'AIRPORT_STATE_CODE', 'LATITUDE', 'LONGITUDE', 'IANA_TIMEZONE', 'OBSERVES_DST']].head(10).to_string())
