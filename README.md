# Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations

**Research Period:** January – December 2025  
**Data Source:** United States Bureau of Transportation Statistics (BTS), Department of Transportation (DOT)  
**Research Focus:** Aircraft rotation reconstruction, downstream delay propagation, and pre-departure delay prediction.

---

## 1. Project Directory Structure

```text
flight-delay-propagation/
│
├── data/
│   ├── raw/
│   │   ├── 2025/
│   │   │   ├── T_ONTIME_REPORTING-JAN.csv
│   │   │   ├── T_ONTIME_REPORTING-FEB.csv
│   │   │   ├── ...
│   │   │   └── T_ONTIME_REPORTING-DEC.csv
│   │   └── T_MASTER_CORD.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_aircraft_rotation_reconstruction.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── audit.py
│   │   └── generate_audit_figures.py
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── models/
├── results/
│   ├── figures/
│   │   ├── airport_geographic_distribution.png
│   │   ├── annual_arrival_delay_distribution.png
│   │   ├── monthly_delay_rate_trend.png
│   │   └── monthly_flight_volume_cancellations.png
│   ├── tables/
│   │   ├── airport_coverage_report.csv
│   │   ├── annual_arrival_delay_distribution.csv
│   │   ├── data_quality_report.csv
│   │   ├── monthly_summary.csv
│   │   └── schema_documentation.csv
│   └── metrics/
│
├── paper/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Phase 1 Audit Summary

- **Total Ingested Flight Records:** 7,001,619 (12 calendar months, 2025).
- **Monthly Schema Consistency:** 100% identical 46-column schema across all 12 files.
- **Reporting Airlines:** 14 unique carriers reporting in all 12 months.
- **Unique Commercial Airports:** 352 active origin and destination airports.
- **Airport Metadata Match:** 100% (352/352) successfully resolved in `T_MASTER_CORD.csv` with complete geographic coordinates.
- **Completed Flights:** 6,879,484 (98.26%).
- **Cancellations:** 102,876 (1.47%).
- **Diversions:** 19,259 (0.27%).
- **Arrival Delay Distribution:** Mean = +8.50 min, Median = -6.00 min, Std = 59.74 min, 90th percentile = +48.0 min, 99th percentile = +225.0 min.
- **Flights Delayed ≥ 15 Minutes (`ARR_DEL15` = 1):** 1,534,638 (22.31% of completed flights).
- **Duplicate Records:** 0 exact duplicates; 0 duplicate operational flight keys.
- **Structural Missingness:** Missing tail numbers (12,501) and missing arrival delays (122,135) correspond exclusively to cancelled and diverted flights. Completed operations have 100% observed tail numbers and arrival delays.

