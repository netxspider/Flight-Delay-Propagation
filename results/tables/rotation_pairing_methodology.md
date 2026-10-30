# Aircraft Rotation Reconstruction Methodology Specification

**Project:** Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations  
**Phase:** Phase 3 Methodological Formulation & Phase 2 Pre-Reconstruction Audit  
**Author:** Flight Delay Propagation Research Project

---

## 1. Fundamental Research Concept

Aircraft operate sequentially throughout each day. Delay propagation occurs when operational disruptions on Flight $i$ ($	ext{Flight}_{	ext{prev}}$) erode the scheduled turnaround buffer at the turnaround airport, forcing Flight $i+1$ ($	ext{Flight}_{	ext{next}}$) to depart late.

The primary objective of rotation reconstruction is to form valid consecutive operational flight pairs:

$$\text{Flight}_i \longrightarrow \text{Flight}_{i+1}$$

operated by the exact same physical aircraft.

---

## 2. Formal Pairing Algorithm and Sequence Conditions

For any two completed flights $f_i$ and $f_{i+1}$ to form a valid consecutive aircraft rotation, they must strictly satisfy all five operational conditions:

### Condition 1: Same Physical Airframe
$$\text{TAIL\_NUM}(f_i) == \text{TAIL\_NUM}(f_{i+1}) \quad \text{and} \quad \text{TAIL\_NUM} \ne \text{NULL}$$

Both operations must be executed by the exact same aircraft registration tail number.

### Condition 2: Spatial Continuity (Station Matching)
$$\text{DEST\_AIRPORT\_ID}(f_i) == \text{ORIGIN\_AIRPORT\_ID}(f_{i+1})$$

The arrival station of the preceding flight must be identical to the departure station of the subsequent flight. If an aircraft lands in Atlanta (`ATL`), its next operation must depart from Atlanta (`ATL`). Gaps in continuity (e.g. unrecorded repositioning or diversions) must break the chain.

### Condition 3: Chronological Sorting on Actual UTC Departure
Within each aircraft tail number, all completed operations are sorted chronologically by actual UTC departure timestamp:

$$\text{DEP\_UTC}(f_i) < \text{DEP\_UTC}(f_{i+1})$$

Sorting on actual UTC departure prevents circular dependencies and correctly orders flights that were dispatched out of scheduled sequence due to delay.

### Condition 4: Physical Feasibility (Temporal Sequence)
$$\text{ARR\_UTC}(f_i) \le \text{DEP\_UTC}(f_{i+1})$$

An aircraft cannot physically depart on its next flight before it has completed its previous flight and arrived at the gate. The actual arrival time of flight $f_i$ must precede or equal the actual departure time of flight $f_{i+1}$.

### Condition 5: Turnaround Buffer and Ground Time Computation
Two distinct temporal intervals must be calculated for every candidate pair:

1. **Actual Ground Time (Turnaround Duration):**
   $$\Delta t_{\text{ground}} = \text{DEP\_UTC}(f_{i+1}) - \text{ARR\_UTC}(f_i)$$
   This represents the actual physical elapsed time the aircraft spent on the ground between gate arrival and subsequent pushback.

2. **Scheduled Turnaround Buffer:**
   $$\Delta t_{\text{buffer}} = \text{CRS\_DEP\_UTC}(f_{i+1}) - \text{CRS\_ARR\_UTC}(f_i)$$
   This represents the planned turnaround window scheduled by the airline dispatchers.

3. **Remaining Turnaround Time (Turnaround Margin):**
   $$\text{Remaining Margin} = \Delta t_{\text{buffer}} - \text{ARR\_DELAY}(f_i)$$
   If the previous arrival delay exceeds the scheduled turnaround buffer, the remaining margin becomes negative, creating operational delay pressure.

---

## 3. Critical Methodological Rule: Preservation of Delayed Rotations

> **IMPORTANT RESEARCH MANDATE:**  
> A candidate pair must **NEVER** be rejected simply because the next scheduled departure occurs before the previous flight's actual arrival:
> 
> $$\text{CRS\_DEP\_UTC}(f_{i+1}) < \text{ARR\_UTC}(f_i)$$

### Physical and Mathematical Explanation
Consider an aircraft scheduled as follows:
- $f_1$ (SFO $\to$ LAX): Scheduled arrival 14:00. Incurred a 90-minute delay; actual arrival at 15:30.
- $f_2$ (LAX $\to$ JFK): Scheduled departure 15:00.
- Scheduled turnaround buffer: $15:00 - 14:00 = 60\text{ minutes}$.

Notice that at 15:00, the aircraft is still in the air. It lands at 15:30. The ground crew completes a rapid 45-minute turnaround, and $f_2$ departs at 16:15 (75 minutes late).

Here:
- $\text{CRS\_DEP\_UTC}(f_2) = 15:00 < \text{ARR\_UTC}(f_1) = 15:30$.
- $\text{ARR\_UTC}(f_1) = 15:30 \le \text{DEP\_UTC}(f_2) = 16:15$ (**Physically Valid!**).
- Actual Ground Time = $45\text{ minutes}$.

This scenario is **the textbook operational manifestation of delay propagation**. If a naive algorithm were to require $\text{CRS\_DEP}(f_2) \ge \text{ARR}(f_1)$, it would systematically discard the exact delayed rotations that our research seeks to model, introducing catastrophic selection bias.

The **sole physical requirement** is that actual arrival precedes actual departure:
$$\text{ARR\_UTC}(f_i) \le \text{DEP\_UTC}(f_{i+1})$$

---

## 4. Ground-Time Threshold Sensitivity Analysis

To distinguish fast commercial turnarounds from overnight maintenance stays or prolonged multi-day ground parking, candidate rotations will be evaluated across exploratory ground-time intervals:

1. **Tight Same-Day Rotation Threshold ($\Delta t_{\text{ground}} \le 6\text{ hours}$):**
   - Captures active intra-day aircraft utilization.
   - Minimizes the probability of unrecorded maintenance interventions or crew swap resets.
2. **Extended Operating Day Threshold ($\Delta t_{\text{ground}} \le 12\text{ hours}$):**
   - Captures extended ground delays, afternoon layovers, and late evening operations.
3. **Overnight / Full 24-Hour Threshold ($\Delta t_{\text{ground}} \le 18\text{ hours}$):**
   - Incorporates overnight aircraft stays at destination airports.
   - Enables empirical testing of whether arrival delays on the last flight of Day $D$ propagate to the first flight of Day $D+1$ after an overnight ground stay.

These thresholds will be evaluated comparatively in Phase 4 and Phase 5 to determine whether model feature importance and propagation coefficients remain robust across definitions.

---

## 5. Methodological Limitations & Boundary Conditions

1. **Calendar Boundary Truncation:**
   - The first observed flights on January 1, 2025 cannot identify a predecessor from December 2024.
   - The last observed flights on December 31, 2025 cannot link to successors in January 2026.
   - These boundary flights are retained as standalone operations but flagged as rotation boundaries.
2. **Repositioning Flights (Ferry Flights):**
   - Commercial airlines occasionally operate non-revenue ferry flights (Part 91) to reposition aircraft after diversions or mechanical issues. These do not appear in the passenger BTS on-time dataset.
   - Spatial continuity enforcement ($\text{DEST}_i == \text{ORIGIN}_{i+1}$) safely terminates the rotation chain when a gap occurs, preventing spurious linkages across different airports.
3. **Daylight Saving Time Boundary (November 2, 2025):**
   - The 120 flights operating near the 01:00-01:59 fallback window are evaluated under the explicit resolution policy documented in `dst_ambiguity_resolution.csv`. Models will be trained with and without these records to verify invariance.
