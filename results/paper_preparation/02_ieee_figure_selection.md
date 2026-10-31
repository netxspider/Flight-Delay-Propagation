# IEEE Conference Paper Figure Selection & Visual Evidence Plan

**Project Title:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Role:** Senior Data Scientist, Visualization Reviewer & IEEE Technical Editor  
**Date:** 2026-10-31  
**Deliverable:** `results/paper_preparation/02_ieee_figure_selection.md`  

---

## 1. Executive Summary & Selection Philosophy

In a standard six-page IEEE double-column conference paper, physical page real estate is strictly limited. Incorporating excessive visual artifacts forces unacceptable truncation of methodological details, mathematical equations, and scientific discussion. 

To maximize informational density while preserving strict page budget constraints, we apply three selection criteria:
1. **Core Research Alignment:** Does the visualization directly address the central research question regarding delay propagation across consecutive aircraft operations?
2. **Complementarity to Tables:** Does the figure convey multi-dimensional non-linear patterns, probability distributions, or ranking behaviors that cannot be effectively communicated in a tabular summary?
3. **Legibility at IEEE Column Width:** Can the visualization be rendered at standard single-column width ($3.5\text{ in} / 88\text{ mm}$) or double-column width ($7.16\text{ in} / 182\text{ mm}$) with crisp, readable labels, consistent typography, and 300 DPI resolution?

From the 32 candidate figures available in `results/figures/`, we select **5 core figures** (with 2 dual-panel compositions) for inclusion in the six-page paper, designated as **Figures 1 through 5**.

---

## 2. Complete Figure Inventory (32 Visualizations)

All figures generated across the research pipeline are preserved at 300 DPI publication resolution in `results/figures/`:

| Index | Figure Filename | Pixel Dimensions | File Size | Phase / Stage | Description / Focus |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `airport_geographic_distribution.png` | $3600 \times 1950$ | $290.1$ KB | Phase 1 | Geographic scatter map of 352 commercial airports |
| 2 | `annual_arrival_delay_distribution.png`| $4200 \times 1500$ | $257.7$ KB | Phase 1 | Full-year delay density distribution and boxplots |
| 3 | `classification_confusion_matrix.png` | $5341 \times 1545$ | $205.3$ KB | Stage 4 | Baseline validation confusion matrices (Group C) |
| 4 | `classification_precision_recall_curve.png`| $2700 \times 2100$ | $328.1$ KB | Stage 4 | Baseline validation precision-recall curves |
| 5 | `classification_roc_curve.png` | $2700 \times 2100$ | $339.6$ KB | Stage 4 | Baseline validation ROC curves |
| 6 | `fig01_target_distributions.png` | $4438 \times 3579$ | $543.8$ KB | Phase 4 | Target arrival delay histogram and binary class balance |
| 7 | `fig02_delay_propagation_relationships.png`| $4770 \times 3579$ | $1308.9$ KB| Phase 4 | Inbound delay vs outbound delay scatter & bin means |
| 8 | `fig03_turnaround_buffer_dynamics.png` | $4770 \times 3579$ | $567.9$ KB | Phase 4 | Delay probability vs scheduled buffer & turnaround deficit |
| 9 | `fig04_temporal_delay_patterns.png` | $4770 \times 3579$ | $774.7$ KB | Phase 4 | Diurnal, day-of-week, and monthly delay trends |
| 10 | `fig05_carrier_airport_heterogeneity.png`| $5070 \times 3851$ | $679.5$ KB | Phase 4 | Delay propagation rates across carriers and major hubs |
| 11 | `fig06_aircraft_level_patterns.png` | $4770 \times 3576$ | $848.9$ KB | Phase 4 | Rotational fatigue vs daily legs and consecutive delays |
| 12 | `fig07_feature_correlations.png` | $5311 \times 2688$ | $988.0$ KB | Phase 4 | Correlation matrix of 18 numerical predictors |
| 13 | `monthly_delay_rate_trend.png` | $2700 \times 1440$ | $193.9$ KB | Phase 1 | Monthly percentage of delayed flights across 2025 |
| 14 | `monthly_flight_volume_cancellations.png`| $3000 \times 1500$ | $209.6$ KB | Phase 1 | Completed flights vs cancellations by month |
| 15 | `permutation_importance_comparison.png`| $4200 \times 1800$ | $234.7$ KB | Stage 4 | Validation permutation importance across models |
| 16 | `regression_actual_vs_predicted.png` | $2700 \times 2100$ | $290.4$ KB | Stage 4 | Validation regression scatter for Group C models |
| 17 | `regression_error_by_delay_band.png` | $3900 \times 1800$ | $248.5$ KB | Stage 4 | Validation MAE across 5 delay bands |
| 18 | `regression_residual_distribution.png` | $2700 \times 1800$ | $248.4$ KB | Stage 4 | Validation residual distribution and error quantiles |
| 19 | `stage5_tuning_curves.png` | $4500 \times 1800$ | $338.5$ KB | Stage 5 | Hyperparameter search curves for RF, DT, and MLP |
| 20 | `stage5_threshold_curves.png` | $3000 \times 1800$ | $313.2$ KB | Stage 5 | Validation precision-recall curves across thresholds |
| 21 | `stage5_final_model_diagnostics.png` | $4500 \times 1800$ | $343.1$ KB | Stage 5 | 4-panel diagnostic dashboard for selected models |
| 22 | `stage6_test_roc_curve.png` | $1950 \times 1650$ | $191.0$ KB | Stage 6 | Final test ROC curve on December 2025 data |
| 23 | `stage6_test_precision_recall_curve.png`| $1950 \times 1650$ | $156.6$ KB | Stage 6 | Final test Precision-Recall curve on December data |
| 24 | `stage6_test_confusion_matrix.png` | $1800 \times 1500$ | $99.4$ KB | Stage 6 | Final test confusion matrix at operating threshold $T=0.35$ |
| 25 | `stage6_test_actual_vs_predicted.png` | $1950 \times 1800$ | $227.4$ KB | Stage 6 | Final test regression actual vs predicted hexbin scatter |
| 26 | `stage6_test_delay_band_performance.png`| $2400 \times 1440$ | $131.5$ KB | Stage 6 | Final test MAE and bias across 5 delay bands |
| 27 | `stage6_test_feature_importance.png` | $3900 \times 1650$ | $222.0$ KB | Stage 6 | Top 10 Permutation Feature Importances (AP drop & MAE) |
| 28 | `stage6_test_residual_distribution.png`| $2100 \times 1440$ | $165.1$ KB | Stage 6 | Final test residual distribution and Gaussian fit |
| 29 | `stage6_test_residual_vs_predicted.png` | $2100 \times 1440$ | $177.8$ KB | Stage 6 | Final test residual vs predicted delay scatter |
| 30 | `stage6_test_calibration_curve.png` | $1800 \times 1500$ | $149.1$ KB | Stage 6 | Final test probability calibration curve |
| 31 | `stage6_test_threshold_comparison.png`| $2129 \times 1399$ | $200.9$ KB | Stage 6 | Final test precision, recall, and F1 across threshold grid |
| 32 | `stage6_validation_test_comparison.png`| $3300 \times 1350$ | $127.8$ KB | Stage 6 | Generalization gap comparison bar chart |

---

## 3. Recommended Primary Figures for the IEEE Paper

The following 5 figures have been selected and copied to `paper/figures/` for direct compilation into `paper/main.tex`:

### Figure 1: Empirical Delay Propagation Dynamics and Turnaround Buffer Physics
- **Source File:** `paper/figures/fig_propagation_relationships.png` (or `fig_turnaround_buffer.png`)
- **Placement:** Section III / IV (Aircraft Rotation Reconstruction & Operational Patterns).
- **Proposed IEEE Caption:**  
  *Fig. 1. Empirical delay propagation dynamics across 6.77 million domestic aircraft rotations in 2025. (a) Mean subsequent arrival delay as a non-linear function of previous arrival delay, illustrating the abrupt propagation knee when inbound delay exceeds ground slack. (b) Subsequent flight delay probability ($\ge 15$ minutes) stratified across scheduled turnaround buffer durations, showing the sharp transition between unbuffered and buffered aircraft rotations.*
- **Scientific Justification:** Visually demonstrates the core physical mechanism of the research problem: ground cushion absorption limits and the threshold where inbound delays overcome scheduled buffers.

### Figure 2: Out-of-Sample Classification Discrimination Diagnostics
- **Source Files:** `paper/figures/fig_test_roc.png` and `paper/figures/fig_test_pr.png` (composed side-by-side as subfigures (a) and (b)).
- **Placement:** Section VI.B (Classification Results).
- **Proposed IEEE Caption:**  
  *Fig. 2. Out-of-sample classification discrimination on the independent December 2025 test dataset ($N = 562,202$). (a) Receiver Operating Characteristic (ROC) curve showing an area under the curve of $0.7943$ compared to the random baseline. (b) Precision-Recall curve demonstrating an Average Precision (AP) of $0.7150$ relative to the $26.48\%$ empirical baseline delay prevalence.*
- **Scientific Justification:** Establishes the primary classification evaluation criteria. Combining ROC and PR curves in a single double-column or side-by-side panel is standard in IEEE publications to illustrate prevalence-invariant ranking versus operating precision.

### Figure 3: Operational Decision Confusion Matrix at Operating Threshold ($T = 0.35$)
- **Source File:** `paper/figures/fig_test_confusion_matrix.png`
- **Placement:** Section VI.E (Threshold Analysis & Operational Dispatch).
- **Proposed IEEE Caption:**  
  *Fig. 3. Confusion matrix for the final Random Forest Classifier (Group C) evaluated on the held-out December 2025 test dataset ($N = 562,202$) at the operational decision threshold $T = 0.35$. Operating at this point identifies $53.24\%$ of delayed flights ($79,250$ true positives) while limiting false alerts on on-time flights to $7.26\%$ ($30,025$ false positives).*
- **Scientific Justification:** Directly communicates the practical utility for airline operations control centers (AOCCs), showing exact count reconciliations with zero ambiguity.

### Figure 4: Continuous Arrival Delay Prediction and Tail Behavior
- **Source File:** `paper/figures/fig_actual_vs_predicted.png`
- **Placement:** Section VI.C (Regression Performance).
- **Proposed IEEE Caption:**  
  *Fig. 4. Actual versus predicted subsequent arrival delay minutes for the final Random Forest Regressor (Group C) on the December 2025 test dataset ($N = 562,202$). The dashed diagonal indicates perfect prediction ($1:1$). The hexbin density plot illustrates high predictive fidelity within ordinary delay regimes ($\le 60$ minutes; $R^2 = 0.4851$, $\text{MAE} = 19.77$ min) alongside structural underprediction in the extreme severe-delay tail ($\ge 120$ minutes).*
- **Scientific Justification:** Essential for visualizing both the model's core explanatory power ($48.51\%$ variance explained) and its primary limitation (underpredicting multi-hour severe disruptions).

### Figure 5: Permutation Feature Importance Across Operational Predictors
- **Source File:** `paper/figures/fig_feature_importance.png`
- **Placement:** Section VI.G (Feature Importance & Attribution).
- **Proposed IEEE Caption:**  
  *Fig. 5. Permutation feature importance on the independent December 2025 test dataset ($N = 100,000$, 5 repeats) for the final Random Forest models. The left panel shows the drop in Average Precision (AP) for classification; the right panel shows the increase in Mean Absolute Error (MAE in minutes) for regression. Derived turnaround buffer features dominate structural importance, while cumulative same-day airframe history provides non-redundant predictive signal.*
- **Scientific Justification:** Directly supports the discussion on which operational signals carry predictive information and visually anchors the methodological discussion on correlated/derived predictors.

---

## 4. Conditional Secondary Figures (Space-Permitting)

If space allows during LaTeX layout tuning:
- `paper/figures/fig_delay_band_performance.png`: Bar chart of MAE and bias across the five delay bands. (Note: Table VII covers these exact numerical metrics, so this figure will only be included if column balance requires an additional graphic).

---

## 5. Rejected Figures & Exclusion Justifications

The remaining 26 figures were excluded from the final paper to ensure compliance with the strict six-page limit:

1. **Geographic and Monthly Flight Volume Maps (Figs 1, 2, 13, 14):** Descriptive dataset overviews that belong in project documentation or an extended journal version, rather than a concise conference paper.
2. **Intermediate Stage 4 Baseline Figures (Figs 3, 4, 5, 15, 16, 17, 18):** Show early baseline model evaluations that are fully captured by Table IV and Table V.
3. **Hyperparameter Tuning & Diagnostic Dashboards (Figs 8, 9, 10, 20):** Internal development artifacts. Tuning curves and threshold grids are summarized concisely in text and tables.
4. **Residual Distributions & Calibration Curves (Figs 28, 29, 30):** Useful diagnostic tools, but secondary to the main actual-vs-predicted plot (Figure 4) and confusion matrix (Figure 3).
5. **Generalization Radar / Bar Charts (Fig. 32):** The validation-to-test generalization gap is comprehensively presented in Table VI.

---

## 6. Figure Preparation Sign-Off

All 5 primary figures are verified, publication-grade, rendered at 300 DPI, and staged in `paper/figures/`.

**Proceed to Deliverable C (Table Selection & LaTeX Formatting).**
