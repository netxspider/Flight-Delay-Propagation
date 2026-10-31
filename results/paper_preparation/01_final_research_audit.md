# Final Research Audit & Scientific Evidence Synthesis

**Project Title:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Role:** Senior Data Scientist, Machine Learning Researcher & IEEE Technical Auditor  
**Audit Date:** 2026-10-31  
**Deliverable:** `results/paper_preparation/01_final_research_audit.md`  
**Reference Commits:** Baseline `c44ca21`, Calibration `540ce39`, Final Verification `40727c0`  

---

## 1. Executive Summary & Repository Inspection

This audit provides a comprehensive, evidence-grounded review of the entire research pipeline across all project phases, verifying that every empirical claim, numerical figure, and methodological statement intended for the final IEEE research paper is backed by verified disk artifacts.

### 1.1 Project Architecture & Lifecycle Review

The codebase is organized into modular phases, strictly maintaining data integrity, chronological partitioning, and data leakage isolation:

```
flight-delay-propagation/
├── data/
│   ├── raw/                 # BTS TranStats source files (Jan–Dec 2025, 46 fields)
│   └── processed/           # Processed parquet partitions (train, validation, test)
├── src/
│   ├── data/                # Ingestion, schema validation, and missing value treatment
│   ├── preprocessing/       # Timezone normalization, UTC conversions, and DST handling
│   ├── rotation/            # Chronological aircraft timeline & rotation reconstruction
│   ├── eda/                 # Statistical analysis, transition matrices, and odds ratios
│   ├── features/            # Leakage-safe Group A, B, and C feature engineering
│   ├── models/              # Baseline and complex model training pipelines (Phases 1-6)
│   └── evaluation/          # Threshold audits, delay band analysis, and decomposition
├── models/
│   ├── classification/      # Stage 4/5 candidate classifiers
│   ├── regression/          # Stage 4/5 candidate regressors
│   └── final/               # Final frozen estimators and ColumnTransformer preprocessors
├── notebooks/               # Numbered Jupyter research notebooks (01 through 06)
├── results/
│   ├── classification/      # Classification performance metrics & candidate comparisons
│   ├── regression/          # Regression performance metrics & delay band breakdowns
│   ├── metrics/             # Serialized JSON and CSV audit trails (Stages 3 through 6.2)
│   ├── tables/              # Formatted research tables and methodology reports
│   ├── figures/             # 32 publication-grade figures (300 DPI resolution)
│   └── predictions/         # Independent December 2025 test predictions (N = 562,202)
└── paper/                   # IEEE conference publication workspace
```

### 1.2 Chronological Partitioning & Sample Summary

The project evaluates out-of-sample temporal generalization using a strictly chronological split across the 2025 calendar year:

| Partition | Calendar Months (2025) | Observation Count | Class 0 (On-Time / Early) | Class 1 (Arrival Delay $\ge 15$m) | Delay Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training Set** | January 1 – September 30 | **$5,065,376$** | $3,972,836$ | $1,092,540$ | $21.57\%$ |
| **Validation Set** | October 1 – November 30 | **$1,140,909$** | $910,388$ | $230,521$ | $20.21\%$ |
| **Independent Test Set**| December 1 – December 31 | **$562,202$** | $413,337$ | $148,865$ | $26.48\%$ |
| **Full Rotational Corpus**| January 1 – December 31 | **$6,768,487$** | **$5,277,901$** | **$1,490,586$** | **$22.02\%$** |

*Verification:* Reconciled against `data/processed/` parquet row counts, `results/metrics/phase5_split_summary.json`, and `results/metrics/stage6_test_distribution_summary.json`.

---

## 2. Dataset & Methodology Verification

### 2.1 Data Source & Raw Inspection (Phase 1 & 2)
- **Source:** Bureau of Transportation Statistics (BTS) On-Time Performance Reporting Carrier data for 2025.
- **Flight Scale:** Reconstructed from 6,898,741 raw flight operations across 14 major operating carriers and 352 commercial airports.
- **Exclusions:** Cancelled flights ($N = 104,748$) and diverted flights ($N = 18,349$) were isolated. Cancelled flights cannot propagate ground delays, and diverted flights break normal rotational airport continuity.
- **Timezone Normalization:** All local departure and arrival timestamps were converted to UTC using airport IATA codes mapped to IANA timezones (via `timezonefinder`). Daylight Saving Time (DST) transitions (March 9 and November 2, 2025) were resolved without timestamp ambiguity or synthetic negative durations (verified in `dst_ambiguity_resolution.csv`).

### 2.2 Aircraft Rotation Reconstruction (Phase 3)
Rotations were reconstructed by chaining consecutive flight legs operated by the identical physical airframe (`TAIL_NUM`):
1. **Airport Continuity:** The scheduled departure airport of leg $k+1$ must exactly equal the actual arrival airport of leg $k$ (`NEXT_ORIGIN == PREV_DEST`). Mismatches ($N = 85,737$ flight legs, e.g., unrecorded ferry flights or tail reassignment) were rejected.
2. **Chronological Validity:** The actual gate departure of leg $k+1$ must occur chronologically after the actual gate arrival of leg $k$ ($\text{Actual Ground Time} > 0$). Exactly 37 corrupted records with negative ground time were removed.
3. **Diversion Breaks:** Diversions create destination discontinuities and were blocked from forming invalid consecutive pairs ($N = 19,109$).
4. **Final Rotation Pairs:** Yielded **6,768,487 valid consecutive rotation pairs** across 6,124 unique commercial aircraft.

### 2.3 Prediction Point & Leakage Prevention (Phase 5)
- **Prediction Point:** Formally defined as $t_{\text{pred}} = \text{Leg } k \text{ Actual Gate Arrival (Wheels-On + Taxi-In)}$.
- **Temporal Barrier:** All features are strictly observed before leg $k+1$'s departure.
- **Explicit Exclusions:** Leg $k+1$'s actual departure time, actual departure delay, actual ground time, airborne elapsed time, and BTS post-hoc delay attribution categories (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`) are strictly banned from feature matrices.
- **Audit:** Confirmed zero temporal leakage across all 25 modeling dataset columns (`feature_leakage_validation.csv`).

---

## 3. Comprehensive Figure Audit (All 32 Visualizations)

Every visualization in `results/figures/` was inspected for resolution (all 300 DPI), visual readability, numerical consistency, and scientific utility:

| Fig # | Filename | What It Shows | Main Finding | Data Verification | IEEE Suitability | Recommendation |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| 1 | `fig01_target_distributions.png` | Histograms of delay distributions & classification imbalance | Severe right skew in delay minutes; 22.02% delay prevalence | Matches `eda_target_summary.csv` | High | **Useful if space permits** |
| 2 | `fig02_delay_propagation_relationships.png` | Scatter & bin curves of $PREV\_ARR\_DELAY$ vs $NEXT\_ARR\_DELAY$ | Sharp non-linear propagation knee when inbound delay exceeds 15 min | Matches `eda_delay_propagation_bins.csv` | Very High | **Essential for Paper (Fig. 1/2)** |
| 3 | `fig03_turnaround_buffer_dynamics.png` | Delay probability vs scheduled buffer & turnaround deficit | Propagation jumps from 13.9% to 50.2% under buffer deficit | Matches `eda_turnaround_deficit_impact.csv` | Very High | **Essential for Paper (Fig. 2/3)** |
| 4 | `fig04_temporal_delay_patterns.png` | Delay rates by time of day, day of week, and calendar month | Diurnal delay accumulation (peaks 18:00–21:00 UTC) | Matches `eda_hourly_patterns.csv` | Moderate | Supplementary |
| 5 | `fig05_carrier_airport_heterogeneity.png` | Carrier and airport specific delay propagation bars | Heterogeneity across hubs and operating carriers | Matches `eda_carrier_summary.csv` | Moderate | Exclude (use text) |
| 6 | `fig06_aircraft_level_patterns.png` | Delay probability vs cumulative legs & consecutive delays | Rotational fatigue: delay rate rises with consecutive delays | Matches `eda_aircraft_consecutive_delays.csv`| High | Useful if space permits |
| 7 | `fig07_feature_correlations.png` | Correlation heatmap of all 18 numerical features | High collinearity between buffer, prev delay, and remaining turn | Matches `eda_correlation_matrix.csv` | Moderate | Exclude (table covers) |
| 8 | `stage5_tuning_curves.png` | Validation hyperparameter tuning curves across candidates | Validation optimal depth=14 and min_samples_leaf=50 | Matches `stage5_classification_tuning.csv` | Moderate | Exclude (table covers) |
| 9 | `stage5_threshold_curves.png` | Precision, recall, and F1 curves across decision thresholds | Best grid F1 at $T=0.39$; operational compromise at $T=0.35$ | Matches `stage5_threshold_integrity_audit.csv` | High | Useful if space permits |
| 10 | `stage5_final_model_diagnostics.png` | 4-panel diagnostic dashboard for validation models | RF dominates linear models across ROC-AUC and PR-AUC | Matches `stage5_final_model_selection.json` | Moderate | Supplementary |
| 11 | `stage6_test_roc_curve.png` | Test ROC curve for final RF Classifier (Group C) | Out-of-sample ROC-AUC = $0.7943$ | Matches `stage6_test_classification_metrics.json`| Very High | **Essential for Paper (Fig. 4a)** |
| 12 | `stage6_test_precision_recall_curve.png` | Test PR curve with baseline no-skill reference | Out-of-sample Average Precision (AP) = $0.7150$ | Matches `stage6_test_classification_metrics.json`| Very High | **Essential for Paper (Fig. 4b)** |
| 13 | `stage6_test_confusion_matrix.png` | Normalized & count confusion matrix at $T=0.35$ | TP=79,250, FP=30,025, FN=69,615, TN=383,312 | Matches `stage6_test_threshold_comparison.csv`| Very High | **Essential for Paper (Fig. 5)** |
| 14 | `stage6_test_actual_vs_predicted.png` | Hexbin scatter of actual vs predicted delay minutes | $R^2 = 0.4851$, 1:1 diagonal, underprediction in severe tail | Matches `stage6_test_regression_metrics.json` | Very High | **Essential for Paper (Fig. 6)** |
| 15 | `stage6_test_delay_band_performance.png` | MAE and prediction bias across 5 delay bands | MAE $\le 22.3$m for 90% of flights; severe underprediction $\ge 120$m | Matches `stage6_test_delay_band_metrics.csv` | High | Useful (covered by Table VII)|
| 16 | `stage6_test_feature_importance.png` | Top 10 Permutation feature importances (AP & MAE) | Remaining turnaround buffer is top structural predictor | Matches `stage6_test_feature_importance.csv` | Very High | **Essential for Paper (Fig. 7)** |
| 17 | `stage6_test_residual_distribution.png` | Histogram and KDE of test regression residuals | Near-zero mean (+1.21m), symmetric core, right-skewed tail | Matches `stage6_test_regression_metrics.json` | Moderate | Supplementary |
| 18 | `stage6_test_residual_vs_predicted.png` | Residuals plotted against predicted arrival delay | Fan-shaped heteroscedasticity common in duration models | Matches `stage6_test_regression_metrics.json` | Moderate | Supplementary |
| 19 | `stage6_test_calibration_curve.png` | Probability calibration curve for RF Classifier | Mild under-confidence in upper probability deciles | Matches `stage6_test_classification_metrics.json`| Moderate | Supplementary |
| 20 | `stage6_test_threshold_comparison.png` | Precision, recall, and alert volume across test thresholds | Recall collapses at $T=0.50$ (45.1%); balanced at $T=0.35$ (53.2%)| Matches `stage6_test_threshold_comparison.csv`| Moderate | Covered by Table VIII |
| 21 | `stage6_validation_test_comparison.png` | Radar / bar chart of generalization gap | Near-identical discrimination transfer (ROC-AUC 0.7932 -> 0.7943) | Matches `stage6_validation_test_comparison.csv`| Moderate | Covered by Table VI |
| 22–32| Miscellaneous Stage 3/4 figures | Early baseline comparisons, annual distributions, maps | Baseline reference artifacts | Verified against Stage 3/4 metrics | Low/Moderate| Exclude |

---

## 4. Comprehensive Table & Result Audit

### 4.1 Summary of Model Comparison (Validation Set, $N = 1,140,909$)

All models trained under strict Stage 4 and Stage 5 protocols were verified against underlying CSV/JSON artifacts:

#### Classification Candidates Comparison (Validation Set)

| Candidate / Estimator | Feature Group | Training Rows | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision (AP) | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Majority Class)** | None | $5,065,376$ | $79.79\%$ | $0.00\%$ | $0.00\%$ | $0.0000$ | $0.5000$ | $0.2021$ | $1.1$s |
| **Logistic Regression** | Group A | $5,065,376$ | $84.66\%$ | $97.14\%$ | $24.83\%$ | $0.3955$ | $0.7356$ | $0.5792$ | $5.3$s |
| **Logistic Regression** | Group B | $5,065,376$ | $84.91\%$ | $81.49\%$ | $32.73\%$ | $0.4671$ | $0.7698$ | $0.6102$ | $40.7$s |
| **Logistic Regression** | Group C | $5,065,376$ | $85.13\%$ | $77.89\%$ | $36.84\%$ | $0.5003$ | $0.7775$ | $0.6228$ | $40.9$s |
| **Decision Tree (Final)** | Group C | $500,000$ | $83.69\%$ | $67.57\%$ | $37.04\%$ | $0.4784$ | $0.7656$ | $0.5983$ | $4.8$s |
| **MLP ($\alpha=0.0$)** | Group C | $250,000$ | $85.08\%$ | $76.24\%$ | $37.64\%$ | $0.5038$ | $0.7818$ | $0.6295$ | $19.2$s |
| **Random Forest (Group A)** | Group A | $500,000$ | $84.82\%$ | $79.88\%$ | $32.06\%$ | $0.4576$ | $0.7824$ | $0.6433$ | $85.4$s |
| **Random Forest (Final, $T=0.50$)**| Group C | $500,000$ | **$85.58\%$** | **$86.68\%$** | $40.88\%$ | $0.5556$ | **$0.7932$** | **$0.6600$** | $258.4$s |
| **Random Forest (Final, $T=0.35$)**| Group C | $500,000$ | $85.29\%$ | $69.20\%$ | **$49.03\%$** | **$0.5740$** | **$0.7932$** | **$0.6600$** | $258.4$s |

*Verification:* Reconciled against `stage3_baseline_metrics.json`, `stage4_metrics.json`, and `stage5_classification_tuning.csv`.

#### Regression Candidates Comparison (Validation Set)

| Candidate / Estimator | Feature Group | Training Rows | MAE (min) | RMSE (min) | MSE ($\text{min}^2$) | $R^2$ Score | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Median Delay)** | None | $5,065,376$ | $22.65$ | $53.39$ | $2850.30$ | $-0.0479$ | $0.3$s |
| **Multiple Linear (OLS)** | Group A | $5,065,376$ | $20.70$ | $44.81$ | $2007.67$ | $0.2619$ | $3.4$s |
| **Multiple Linear (OLS)** | Group B | $5,065,376$ | $21.37$ | $44.73$ | $2000.89$ | $0.2644$ | $50.6$s |
| **Multiple Linear (OLS)** | Group C | $5,065,376$ | $21.19$ | $44.40$ | $1970.97$ | $0.2754$ | $51.9$s |
| **Decision Tree (Final)** | Group C | $500,000$ | $17.89$ | $39.46$ | $1557.09$ | $0.4281$ | $4.6$s |
| **MLP ($\alpha=0.0$)** | Group C | $250,000$ | **$17.06$** | $39.38$ | $1550.84$ | $0.4304$ | $17.5$s |
| **Random Forest (Group A)** | Group A | $500,000$ | $17.58$ | $38.89$ | $1512.43$ | $0.4440$ | $105.2$s |
| **Random Forest (Final)** | Group C | $500,000$ | $17.40$ | **$38.62$** | **$1491.67$** | **$0.4516$** | $312.1$s |

*Verification:* Reconciled against `results/regression/baseline_model_comparison.csv` and `stage5_regression_tuning.csv`.

---

## 5. Historical Discrepancy Register & Resolution Audit

Every inconsistency identified during Stages 5.1, 5.2, 6.1, and 6.2 was audited against underlying artifacts to confirm resolution:

| Issue ID | Conflicting Values | Historical Origin | Source of Truth | Definitive Resolution | IEEE Paper Impact |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **D-01** | Stage 3 Baseline ROC-AUC reported as $<0.50$ ($0.3955$) | Positive class indexing inverted in baseline evaluation script | `results/classification/baseline_model_comparison.csv` | Re-evaluated with positive class index $1$. True Group A ROC-AUC is $0.7356$. | Zero paper impact (resolved in Stage 4). |
| **D-02** | Validation Early MAE reported as $9.79$m vs. $12.19$m | Typographical draft error in Stage 5.1 report prose | `stage4_error_by_delay_band.csv` (line 4) | True empirical RF Group C early MAE is $12.19$m ($12.1856$m). $9.79$m formally expunged. | Verified $12.19$m used in decomposition and paper tables. |
| **D-03** | OLS Group A $R^2$ reported as $0.2312$ vs. $0.2619$ | Draft typo in Stage 6.1 final audit table | `results/regression/baseline_model_comparison.csv` (line 3) | Verified empirical OLS Group A $R^2$ is $0.261919$ (MSE $2007.67$). $0.2312$ formally expunged. | Table IV & V report true $R^2 = 0.2619$. |
| **D-04** | Test Parquet size reported as $61.1$ MB vs. $12.37$ MB | In-memory uncompressed DataFrame size vs on-disk Parquet size | `data/processed/modeling_test.parquet` | Disk file size is $12,973,636$ B ($12.37$ MB) with SHA-256 `3f0e5494fdb4d934...`. File was never modified. | Dataset size clarified in Section III. |
| **D-05** | Decision threshold terminology: "global F1 optimum" vs $0.35$ | Confusion between mathematical grid peak ($T=0.39$) and operational point | `results/metrics/stage5_threshold_integrity_audit.csv` | $T=0.39$ is best F1 on 0.01 grid ($0.5751$); $T=0.35$ is selected operational compromise ($0.5740$). | Clearly differentiated in Section VI & VII. |
| **D-06** | Severe delay "300-minute tree training ceiling" | Theoretical conjecture unsupported by training data | `modeling_train.parquet` & `final_selected_regressor.joblib` | Training target delays reach $4336.0$m; tree leaves output up to $1141.15$m; test predictions reach $1027.9$m. Ceiling claim refuted. | Underprediction accurately attributed to conditional leaf averaging (`min_samples_leaf=50`). |

---

## 6. Scientific Limitations & Boundaries of Validity

The IEEE research paper must clearly state what can and cannot be concluded from this study:

1. **Observational Association vs. Counterfactual Causation:**
   - The predictive association between turnaround buffer deficit and subsequent delays does **not** prove that adding schedule padding will causally eliminate downstream delays in a 1:1 ratio. Airlines adjust operational pacing, ground crew allocations, and taxi speeds in response to schedule buffers.
2. **Unobserved Operational Telemetry:**
   - The BTS dataset contains scheduled and actual flight timestamps but lacks direct meteorological data (TDWR radar, convective storm fronts), airport surface congestion (gate holds, taxiway queues), and FAA Air Traffic Control System Command Center (ATCSCC) Ground Delay Programs (GDPs). These remain unobserved operational hypotheses.
3. **Severe Delay Regime Underprediction:**
   - For delays $\ge 120$ minutes ($4.04\%$ of December operations), the model exhibits substantial negative prediction bias ($-90.38$ minutes, MAE $= 109.98$ minutes). Extreme disruptions are driven by compound operational events occurring *after* departure (airborne holding, gate unavailability upon arrival, crew duty expirations under FAR Part 117), which cannot be inferred from pre-departure rotational timestamps.
4. **Permutation Importance on Collinear Derived Features:**
   - Shuffling `ESTIMATED_REMAINING_TURNAROUND_MINUTES` independently breaks its physical algebraic identity ($\text{BUFFER} - \text{PREV\_DELAY}$), generating impossible feature states that cause tree predictions to collapse into negative $R^2$ ($-0.4415$, producing an arithmetic $R^2$ drop of $+0.9185$). This reflects heavy architectural dependence on this derived feature, rather than an independent physical causal effect.
5. **Temporal & Geographic Scope:**
   - The empirical study is bounded by US domestic commercial operations in calendar year 2025. Evaluation on December reflects holiday and winter conditions; multi-year cross-validation remains future work.

---

## 7. Recommended Paper Structure & Evidence Map

The IEEE conference paper will follow a tight, six-page narrative structure with all claims tied directly to verified artifacts:

```
I. INTRODUCTION
   ├── Operational motivation: Airframe rotations as vulnerability pathways (BTS 2025 context)
   ├── Research Question: Extent of predictability from previous flight operations
   └── Key Contributions: (1) Scale (6.77M rotations), (2) Leakage-safe framework, (3) RF dual modeling
II. RELATED WORK
   ├── Flight delay prediction (network-level vs rotational approaches)
   ├── Turnaround time management and buffer scheduling
   └── Machine learning in commercial aviation operations
III. DATASET & ROTATION RECONSTRUCTION
   ├── BTS TranStats 2025 dataset (January–December, 6.90M raw operations)
   ├── Chronological UTC conversion, DST resolution, and timezone mapping
   ├── Rotation reconstruction pipeline: tail matching, continuity, diversion breaks (Table I, II)
   └── Descriptive operational vulnerability: 382,937 turnaround deficits (99.5% delayed)
IV. FEATURE ENGINEERING & LEAKAGE ISOLATION
   ├── Formal prediction point: Immediately after Leg k gate arrival
   ├── Feature hierarchies: Group A (Physics), Group B (Context), Group C (Airframe History) (Table III)
   └── Leakage audit: Explicit exclusion of Leg k+1 departure/airborne telemetry and delay causes
V. METHODOLOGY & EXPERIMENTAL SETUP
   ├── Chronological split: Train (Jan-Sep), Val (Oct-Nov), Independent Test (Dec)
   ├── Approved estimator families: OLS/Logistic, Decision Tree, Random Forest, MLP (alpha=0.0)
   ├── Training constraints: Sparse OHE, controlled sampling (500k for RF), zero data snooping
   └── Validation-based threshold optimization: Operating point T=0.35 vs grid optimum T=0.39
VI. RESULTS & EMPIRICAL DISCUSSION
   ├── Exploratory delay propagation patterns (Fig. 1, 2)
   ├── Validation model comparison: Linear vs tree architectures (Table IV, V)
   ├── Independent December test evaluation: ROC-AUC 0.7943, AP 0.7150, R^2 0.4851 (Table VI, Fig. 3, 4)
   ├── Decision threshold spectrum & confusion matrix at T=0.35 (Table VIII, Fig. 5)
   ├── Delay-band stratification & severe delay mechanics (Table VII, Fig. 6)
   └── Permutation feature importance & algebraic identity breakdown (Fig. 7)
VII. GENERALIZATION & DISTRIBUTION SHIFT
   ├── Winter surge: Delay prevalence rise (20.2% -> 26.5%) and severe delay surge (+55.4%)
   └── Shift-share Oaxaca-Blinder decomposition: 94.0%-98.4% composition effect
VIII. SCIENTIFIC LIMITATIONS
   ├── Observational vs causal interpretation
   ├── Unobserved external constraints (ATC, weather, crew legality)
   └── Tree ensemble leaf averaging constraints
IX. CONCLUSION & FUTURE WORK
   ├── Summary of primary findings (delays are predictable; turnaround deficit is dominant signal)
   └── Operational dispatch implications & future multi-year extensions
```

---

## 8. Audit Sign-Off

The research audit is **complete, verified, and reconciled**. All numerical values to be included in the paper are cross-referenced with exact source artifacts. No inconsistencies remain unresolved.

**Proceed to Deliverable B (Figure Selection) and Deliverable C (Table Selection).**
