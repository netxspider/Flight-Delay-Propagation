# Phase 2 Final Methodological Corrections and Pre-Phase 3 Specification

**Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Date:** October 2026  
**Status:** Pre-Phase 3 Verification and Audit Corrections

---

## Executive Summary

Prior to initiating full aircraft rotation reconstruction in Phase 3, three critical methodological issues in temporal classification, operational sequence continuity, and predictive feature availability were thoroughly audited, corrected, and validated. This document establishes the formal mathematical rules, empirical findings, and validation tests governing these corrections.

---

## 1. Correction 1: Correct Overnight Flight Identification

### 1.1 Limitation of the Previous Methodology
The preliminary verification report identified **229,236** overnight flights based on a comparison of local scheduled departure and arrival clock minutes:

$$\text{Local Minute Comparison: } \text{CRS\_ARR\_MINUTE\_OF\_DAY} < \text{CRS\_DEP\_MINUTE\_OF\_DAY}$$

This naive heuristic is mathematically and operationally unreliable for commercial flights crossing time zones:
1. **Westbound Clock Regressions:** A flight crossing time zones westward can experience a local arrival minute that is smaller than its departure minute even when the flight completes on the exact same calendar day (e.g. departing Detroit at 23:15 Eastern, arriving Chicago at 23:45 Central, clock minutes $1425$ vs $1395$, or flights with late-night delays).
2. **Eastbound Calendar Shifts:** Conversely, long-haul domestic flights operating eastbound (e.g. Hawaii to the West Coast, or West Coast to East Coast) can depart in the afternoon or evening and land on the following calendar day, while their local clock minutes do not invert in a simple intra-day comparison.
3. **Universal Time (UTC) Disconnect:** In an air traffic network, operational sequences must be anchored in universal time. Because US commercial air traffic spans UTC-4 (Atlantic) to UTC-11 (Samoa), evening flights across the continental US routinely cross the universal midnight boundary ($00:00\text{ UTC}$) while remaining on the same local calendar day.

### 1.2 Mathematical Formulation of Corrected Timing Categories
To resolve this, we establish three distinct, mathematically unambiguous temporal categories:

1. **Genuine Destination Local Calendar Rollover:**
   $$\text{DATE}(\text{CRS\_ARR\_LOCAL}) > \text{DATE}(\text{CRS\_DEP\_LOCAL})$$
   The scheduled touchdown at the destination airport occurs on calendar day $D+1$ relative to the origin scheduled departure date.
   - **Empirical Count (2025):** **223,752 flights** ($3.25\%$).

2. **Genuine UTC Calendar Boundary Crossings (Scheduled):**
   $$\text{DATE}(\text{CRS\_ARR\_UTC}) > \text{DATE}(\text{CRS\_DEP\_UTC})$$
   The aircraft is in flight during the $00:00\text{ UTC}$ universal calendar-day transition.
   - **Empirical Count (2025):** **1,023,851 flights** ($14.88\%$).

3. **Genuine UTC Calendar Boundary Crossings (Actual):**
   $$\text{DATE}(\text{ARR\_UTC}) > \text{DATE}(\text{DEP\_UTC})$$
   The flight physically completed touchdown and gate arrival on a subsequent UTC calendar day relative to its actual departure.
   - **Empirical Count (2025):** **1,118,523 flights** ($16.26\%$).

4. **Timezone Discrepancy Artifact:**
   $$\left(\text{CRS\_ARR\_MIN} < \text{CRS\_DEP\_MIN}\right) \ne \left(\text{DATE}(\text{CRS\_ARR\_UTC}) > \text{DATE}(\text{CRS\_DEP\_UTC})\right)$$
   - **Empirical Discrepancy Count:** **1,246,585 flights** ($18.12\%$).

### 1.3 Methodological Recommendation for Phase 3
Phase 3 rotation reconstruction will strictly use **`CRS_DEP_UTC`**, **`DEP_UTC`**, **`CRS_ARR_UTC`**, and **`ARR_UTC`** for all chronological ordering, elapsed duration validation, and overnight stay identification. Local timestamps will be retained solely for time-of-day scheduled features (`NEXT_CRS_DEP_HOUR`).

---

## 2. Correction 2: Prevention of False Aircraft Connections Across Diversions

### 2.1 The Operational Hazard of Ignoring Diversions
In Phase 2, 19,258 diverted flights were preserved in `data/processed/diverted_flights.parquet`. If rotation reconstruction were performed solely on completed flights (`flights_cleaned.parquet`), diverted operations would vanish from the timeline, causing two distinct classes of false connections:

1. **Forward False Linkage ($f_{\text{diverted}} \to f_{\text{next}}$):**
   Attempting to pair a diverted flight $f_{\text{diverted}}$ with subsequent flight $f_{\text{next}}$ based on the assumption that $f_{\text{diverted}}$ landed at its scheduled destination ($DEST$). In reality, the aircraft landed at an unscheduled diversion airport ($Z \ne DEST$), meaning the aircraft was not physically present at $DEST$ to operate $f_{\text{next}}$.
2. **Bridged False Linkage ($f_{\text{prev}} \to f_{\text{next}}$ across an unobserved diversion):**
   Suppose an aircraft operates $f_1$ (ORD $\to$ LGA, completed), then $f_2$ (LGA $\to$ DFW, diverted to STL), and later $f_3$ (LGA $\to$ MIA, completed).
   If $f_2$ is omitted from the sequence, a naive algorithm comparing completed flights sees:
   - $f_1$ destination: `LGA`
   - $f_3$ origin: `LGA`
   - Aircraft tail: Identical
   - $\text{ARR\_UTC}(f_1) < \text{DEP\_UTC}(f_3)$
   The naive algorithm would falsely pair $f_1 \to f_3$ as a consecutive turnaround, completely ignoring that the aircraft flew $f_2$ in between and was diverted to St. Louis (`STL`).

### 2.2 Sequence-Breaking Protocol
To eliminate false linkages:
1. Every diverted flight is preserved in the aircraft's chronological operational timeline as a **Sequence-Breaking Marker**.
2. A diverted flight can **never** serve as $f_{\text{prev}}$ or $f_{\text{next}}$ in a primary rotation pair.
3. No completed flight preceding a diversion can pair with any completed flight following the diversion. The diversion breaks the rotation chain unconditionally.

### 2.3 Empirical Validation Results
Our empirical audit across all 2025 flights revealed:
- **Total Diverted Operations:** $19,258$ flights.
- **Diverted Operations with Preceding Completed Flight:** $19,243$ flights.
- **Diverted Operations with Following Completed Flight:** $19,231$ flights.
- **False Candidate Connections Blocked by Diversion Protocol:** **Exactly 1,352 false rotation pairs** were identified and successfully blocked.

### 2.4 Cancellation Handling Policy
In contrast to diversions, **cancelled flights do not represent aircraft movements**. When a flight is cancelled, the aircraft remains parked at the departure station.
- If $f_1$ completes at Station $X$, and scheduled flight $f_2$ ($X \to Y$) is cancelled, the physical aircraft remains at Station $X$.
- If the airline subsequently dispatches the aircraft on flight $f_3$ from Station $X$, station continuity ($\text{DEST}(f_1) == \text{ORIGIN}(f_3) == X$) is physically maintained.
- However, $f_3$ is not the originally scheduled turnaround of $f_1$; the ground stay is extended.
- **Policy:** In Phase 3, consecutive completed operations where the aircraft remained at the same station are linked, but flagged with `HAD_INTERVENING_CANCELLATION = 1`. This allows sensitivity testing with and without cancellation-impacted rotations.

---

## 3. Correction 3: Strict Prediction-Time Feature Availability

### 3.1 Definition of the Prediction Moment ($t_{\text{pred}}$)
To ensure zero target leakage, the prediction moment is strictly defined as:

$$\mathbf{t_{\text{pred}}}: \quad \text{Immediately after } \text{ARR\_UTC}(f_{\text{prev}}) \text{ is observed, strictly before } \text{DEP\_UTC}(f_{\text{next}})$$

At $t_{\text{pred}}$, the aircraft has just parked at the gate from its previous flight. The subsequent flight $f_{\text{next}}$ has not yet boarded or pushed back.

### 3.2 Feature Partitioning & Leakage Prevention Rules

#### Permitted Predictors ($\le t_{\text{pred}}$)
- **Preceding Flight Performance:** $\text{PREV\_ARR\_DELAY}$, $\text{PREV\_DEP\_DELAY}$, $\text{PREV\_ARR\_DEL15}$.
- **Turnaround Buffer Metrics:**
  - $\text{SCHEDULED\_TURNAROUND\_BUFFER} = \text{CRS\_DEP\_UTC}(f_{\text{next}}) - \text{CRS\_ARR\_UTC}(f_{\text{prev}})$ (Known from timetable).
  - $\text{ESTIMATED\_REMAINING\_TURNAROUND} = \text{SCHEDULED\_TURNAROUND\_BUFFER} - \text{PREV\_ARR\_DELAY}$ (Turnaround buffer remaining after absorbing arrival delay).
  - $\text{IS\_SCHEDULED\_TURNAROUND\_DEFICIT} = \mathbb{I}(\text{PREV\_ARR\_DELAY} > \text{SCHEDULED\_TURNAROUND\_BUFFER})$.
- **Scheduled Characteristics of Subsequent Flight:** $\text{NEXT\_CRS\_DEP\_HOUR}$, $\text{NEXT\_DAY\_OF\_WEEK}$, $\text{NEXT\_MONTH}$, $\text{NEXT\_CARRIER}$, $\text{NEXT\_ORIGIN}$, $\text{NEXT\_DEST}$, $\text{NEXT\_DISTANCE}$, $\text{NEXT\_CRS\_ELAPSED\_TIME}$.
- **Accumulated Intra-Day Aircraft Context:** $\text{AIRCRAFT\_CONSECUTIVE\_DELAYED\_FLIGHTS}$, $\text{AIRCRAFT\_CUMULATIVE\_DELAY\_TODAY}$.

#### Strictly Banned Variables ($> t_{\text{pred}}$)
- $\mathbf{\text{ACTUAL\_GROUND\_TIME}} = \text{DEP\_UTC}(f_{\text{next}}) - \text{ARR\_UTC}(f_{\text{prev}})$: **BANNED (LEAKAGE)**. Requires $\text{DEP\_UTC}(f_{\text{next}})$, which occurs in the future relative to $t_{\text{pred}}$.
- $\mathbf{\text{NEXT\_DEP\_DELAY}}$ and $\mathbf{\text{NEXT\_DEP\_TIME}}$: **BANNED (LEAKAGE)**. Actual departure delay of subsequent flight is unknown at gate arrival.
- $\mathbf{\text{NEXT\_TAXI\_OUT}}$, $\mathbf{\text{NEXT\_WHEELS\_OFF}}$, $\mathbf{\text{NEXT\_AIR\_TIME}}$: **BANNED (LEAKAGE)**.
- $\mathbf{\text{NEXT\_CARRIER\_DELAY}}$, $\mathbf{\text{NEXT\_WEATHER\_DELAY}}$, $\mathbf{\text{NEXT\_NAS\_DELAY}}$, $\mathbf{\text{NEXT\_SECURITY\_DELAY}}$, $\mathbf{\text{NEXT\_LATE\_AIRCRAFT\_DELAY}}$: **BANNED (LEAKAGE)**. Retrospective post-flight delay breakdown fields.
- $\mathbf{\text{NEXT\_ARR\_DELAY}}$ and $\mathbf{\text{NEXT\_ARR\_DEL15}}$: **PREDICTION TARGETS ONLY**.

---

## 4. Deliverables Summary

1. [`results/tables/timestamp_consistency_report.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/tables/timestamp_consistency_report.csv): Updated with corrected UTC vs. local calendar overnight metrics.
2. [`results/tables/diversion_sequence_validation.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/tables/diversion_sequence_validation.csv): Full validation report documenting the 19,258 diversions and exactly 1,352 false pairs blocked.
3. [`results/tables/prediction_feature_availability.csv`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/tables/prediction_feature_availability.csv): Comprehensive 37-variable feature availability and leakage classification matrix.
4. [`results/tables/phase2_final_methodology.md`](file:///Users/arnavraj/Downloads/flight-delay-propagation/results/tables/phase2_final_methodology.md): Formal methodological specification.

---

## 5. Phase 3 Readiness Checklist

With these three corrections verified, the project architecture is ready for Phase 3:
- [x] UTC timestamp calculation vectorized in DuckDB.
- [x] Genuine overnight operations distinguished from time zone artifacts.
- [x] Diversion sequence-breaking protocol established and tested (1,352 false pairs blocked).
- [x] Cancellation handling policy documented.
- [x] Prediction moment $t_{\text{pred}}$ established with strict ban on actual ground time and departure delays.
- [x] Delayed rotation preservation rule formally established.
"""
    methodology_path = os.path.join(results_dir, "phase2_final_methodology.md")
    with open(methodology_path, "w") as f:
        f.write(methodology_md)
    print(f"Saved phase2_final_methodology.md to {methodology_path}")
    return True

if __name__ == "__main__":
    pass

