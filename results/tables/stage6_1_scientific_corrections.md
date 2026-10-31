# Phase 6 — Stage 6.1: Scientific Interpretation, Integrity & Publication Corrections

**Research Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Audit Role:** Senior ML Engineer & IEEE Research Methodology Auditor  
**Baseline Commit:** [`c44ca21`](file:///Users/arnavraj/Downloads/flight-delay-propagation)  
**Status:** Methodological & Publication-Facing Interpretation Audit  

---

## 1. Executive Summary of Corrections

Following the completion of the Phase 6 Stage 6 independent out-of-sample test evaluation, this audit performed a rigorous verification of all scientific interpretations, generalization claims, causal statements, and publication-facing findings. 

All empirical metrics, model parameters, prediction files, and test datasets remain **100% frozen and unmodified**. The corrections documented below transition the narrative prose from informal, speculative assertions into methodologically defensible, publication-grade scientific statements suitable for an IEEE research paper.

---

## 2. Itemized Scientific Corrections Mapping

| Audit Area | Original Interpretation / Phrasing | Corrected Scientific Interpretation | Methodological Rationale |
| :--- | :--- | :--- | :--- |
| **1. Seasonal Distribution Shift** | *"Driven by winter weather and holiday air traffic volume... winter storm disruptions compound with aircraft turnarounds."* | **Described as an empirical temporal distribution shift.** Weather conditions, holiday travel demand, and airport operational congestion are identified as *plausible operational hypotheses*, not verified causal mechanisms, because real-time meteorological radar and passenger demand telemetry are absent from the BTS dataset. | The BTS dataset records only flight timestamps and carrier codes; it does not contain direct telemetry on blizzard severity, de-icing queue wait times, or holiday load factors. Observational data cannot establish causal attribution. |
| **2. Regression Generalization & MAE Increase** | *"The +2.37 minute increase in overall test MAE is almost entirely an aggregation artifact of population composition."* | **Supported by formal Oaxaca-Blinder / shift-share decomposition:** 94.0% to 98.4% ($+2.23$ to $+2.33$ minutes) of the $+2.37$ minute MAE increase is attributable to the population-composition shift (specifically, the 55% surge in severe delays $\ge 120$ min carrying higher baseline errors), while within-band model accuracy remained virtually stable ($+0.04$ to $+0.25$ minutes net effect). | Unsubstantiated claims of "aggregation artifacts" are scientifically invalid without a formal mathematical decomposition. The formal calculation now rigorously demonstrates the exact magnitude of composition vs. within-band error effects. |
| **3. Group A vs. Group C Feature Progression** | *"Cumulative same-day airframe history (Group C) further boosts PR-AUC... proving historical aircraft stress is a genuine non-redundant signal."* | **Group C vs. Group A represents the COMBINED contribution of flight context (Group B) AND aircraft operational history (Group C).** For Random Forest, Group B was not independently evaluated, so the isolated marginal contribution of aircraft history alone cannot be separated from route/carrier context. In linear baselines (where Groups A, B, and C were evaluated under identical conditions), adding aircraft history to flight context produced a $+0.0077$ ROC-AUC increase in Logistic Regression and a $+0.0110$ $R^2$ increase in OLS. | Group C is a superset containing both Group B (7 context features) and Group C (4 history features). Attributing the entire delta between Group A and Group C to aircraft history is a methodological attribution error. |
| **4. Feature Importance & Derived Predictors** | *"$ESTIMATED\_REMAINING\_TURNAROUND\_MINUTES$ caused an $R^2$ decrease of $0.9185$... proves turnaround buffer is the primary physical propagation bottleneck."* | **Permutation importance measures predictive dependence under univariate perturbation, NOT physical causation.** Because $ESTIMATED\_REMAINING\_TURNAROUND\_MINUTES$ is a derived linear combination ($SCHEDULED\_TURNAROUND\_BUFFER - PREV\_ARR\_DELAY$), permuting it independently breaks its collinear identity with its component features, generating physically impossible synthetic records that collapse model $R^2$ from $+0.4770$ to a negative value ($-0.4415$), producing an arithmetic drop of $0.9185$. | Permuting a derived feature while holding its component features constant creates out-of-distribution synthetic inputs that distort tree split logic. It reflects high predictive reliance in the fitted forest, not direct counterfactual physical causality. |
| **5. Generalization Language & Metric Characterization** | *"Flawless discrimination transfer... Captures the majority of propagated delays."* | **Replaced with precise statistical language:** "Stable discrimination on the independent December test set (ROC-AUC 0.7932 vs. 0.7943)... Identifies 53.24% of delayed flights at the selected operating threshold." Clarified that ROC-AUC measures ranking discrimination (prevalence-invariant), whereas Average Precision (AP) is prevalence-sensitive; the increase in AP ($0.6600 \rightarrow 0.7150$) reflects December's higher base delay rate (26.48% vs 20.21%), not an improvement in model quality. | Language suggesting a model "improved" after deployment is scientifically inaccurate. A model frozen in November does not learn or improve on December data; shifts in precision and AP reflect test population base-rate mechanics. |
| **6. Severe-Delay Prediction Error** | *"When delays exceed 2 hours... compound external constraints dominate—such as flight crew duty-time expirations, gate occupancy gridlocks, and FAA Ground Delay Programs."* | **Framed as unobserved operational hypotheses alongside mathematical tree ensemble limits.** Severe delays ($\ge 120$ min) exhibit large negative prediction bias ($-90.38$ minutes) primarily because tree regression leaf predictions are bounded by sample means in training partitions and cannot extrapolate into unobserved extreme tails. Compound external operational constraints are identified as unobserved contextual factors rather than proven drivers. | Observational models cannot claim to have identified the root causes of extreme delays when the causal variables (FAA ground delay programs, crew duty logs, airport gate holds) are unobserved in the modeling matrix. |

---

## 3. Formal Generalization Decomposition Findings

Using the serialized band-level error data from Validation ($N = 1,140,909$) and Test ($N = 562,202$):

$$\Delta \text{MAE} = \sum_{i=1}^5 w_i^{(T)} \cdot \text{MAE}_i^{(T)} - \sum_{i=1}^5 w_i^{(V)} \cdot \text{MAE}_i^{(V)} = 19.7693 - 17.4019 = +2.3674 \text{ minutes}$$

### Mathematical Decomposition Results:
1. **Oaxaca-Blinder (Validation Baseline):**
   - **Counterfactual Test MAE (Test weights applied to Validation within-band errors):** **$19.7316$ minutes**.
   - **Population-Composition Effect:** **$+2.3297$ minutes ($98.41\%$ of total increase)**.
   - **Within-Band Error Effect:** **$+0.0377$ minutes ($1.59\%$ of total increase)**.
2. **Symmetric Shapley Decomposition:**
   - **Composition Effect:** **$+2.2250$ minutes ($93.99\%$)**.
   - **Within-Band Error Effect:** **$+0.1424$ minutes ($6.01\%$)**.

### Scientific Conclusion:
The mathematical decomposition confirms that between **$94.0\%$ and $98.4\%$** of the macro-level $+2.37$ minute MAE increase is driven by the population-composition shift (specifically, the 55% increase in severe delays $\ge 120$ min, which have larger baseline error magnitudes). Model accuracy within specific delay bands remained remarkably constant across both seasons.

---

## 4. Feature Group Progression: Linear vs. Tree Architecture Isolation

| Model Architecture | Progression Stage | Features Included | ROC-AUC / $R^2$ | Delta Attribution |
| :--- | :--- | :---: | :---: | :--- |
| **Logistic Regression** | **Group A** (Turnaround Only) | 7 features | ROC-AUC = 0.7356 | Baseline turnaround physics |
| **Logistic Regression** | **Group B** (+ Flight / Route Context) | 14 features | ROC-AUC = 0.7698 | **+0.0341 ROC-AUC** from network context |
| **Logistic Regression** | **Group C** (+ Aircraft History) | 18 features | ROC-AUC = 0.7775 | **+0.0077 ROC-AUC** from aircraft history alone |
| **Multiple Linear (OLS)**| **Group A** (Turnaround Only) | 7 features | $R^2 = 0.2619$ | Baseline turnaround physics |
| **Multiple Linear (OLS)**| **Group B** (+ Flight / Route Context) | 14 features | $R^2 = 0.2644$ | **+0.0025 $R^2$** from network context |
| **Multiple Linear (OLS)**| **Group C** (+ Aircraft History) | 18 features | $R^2 = 0.2754$ | **+0.0110 $R^2$** from aircraft history alone |
| **Random Forest** | **Group A** (Turnaround Only) | 7 features | ROC-AUC = 0.7824, $R^2 = 0.4440$ | Baseline turnaround physics |
| **Random Forest** | **Group C** (+ Context & History) | 18 features | ROC-AUC = 0.7932, $R^2 = 0.4516$ | **+0.0108 ROC-AUC, +0.0076 $R^2$** from **combined** context + history |

*Methodological Note:* Because Group B was not trained for Random Forest in Stage 4/5, the isolated contribution of aircraft history cannot be decoupled from route context for the tree ensemble. The linear models rigorously isolate the marginal contribution of aircraft history alone ($+0.0077$ ROC-AUC, $+0.0110$ $R^2$).

---

## 5. Derived Feature Permutation Importance Mechanics

The feature `ESTIMATED_REMAINING_TURNAROUND_MINUTES` is an exact algebraic function of two other features:
$$\text{ESTIMATED\_REMAINING\_TURNAROUND\_MINUTES} = \text{SCHEDULED\_TURNAROUND\_BUFFER} - \text{PREV\_ARR\_DELAY}$$

When independently permuted across 100,000 test observations (5 repeats):
1. **Regression $R^2$ Drop of $0.9185$:**
   - Baseline sample $R^2 = +0.4770$.
   - Permuted model $R^2 = -0.4415$.
   - $\Delta R^2 = \text{base} - \text{permuted} = 0.4770 - (-0.4415) = +0.9185$.
   - The drop exceeds the baseline $R^2$ because the permuted model performs substantially worse than a horizontal mean prediction ($R^2 < 0$).
2. **Methodological Safeguard:**
   - Permuting a derived feature independently breaks its algebraic consistency with `SCHEDULED_TURNAROUND_BUFFER` and `PREV_ARR_DELAY`, creating impossible synthetic feature combinations.
   - This test measures the tree ensemble's internal structural reliance on split thresholds along that specific feature dimension. It **must not** be interpreted as a measure of physical causal influence.
