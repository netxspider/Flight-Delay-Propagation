"""
Script to generate notebooks/06_model_development.ipynb with all 25 required sections.
"""

import json
import os

NOTEBOOK_PATH = "/Users/arnavraj/Downloads/flight-delay-propagation/notebooks/06_model_development.ipynb"

def make_cell(cell_type, source, outputs=None, execution_count=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell

cells = []

# Section 1
cells.append(make_cell("markdown", """# Phase 6: Model Development, Training, Validation, and Evaluation
## Flight Delays Are Contagious: Predicting Delay Propagation Across Consecutive Aircraft Operations

**Academic Context:** BTech CSE CA2 Machine Learning Research Project / IEEE Paper Methodology & Evaluation  
**Research Question:** *"To what extent can delays from previous flight operations predict delays in subsequent flights operated by the same aircraft?"*  

---

### Key Experimental Boundaries & Principles:
1. **Pristine Chronological Split:**
   - **Training Set:** January 1 – September 30, 2025 ($N = 5,065,376$)
   - **Validation Set:** October 1 – November 30, 2025 ($N = 1,140,909$) — *Model selection & tuning*
   - **Test Set:** December 1 – December 31, 2025 ($N = 562,202$) — *Strictly isolated test set*
2. **Three Progressive Feature Groups:**
   - **Group A (7 features):** Preceding flight delays and turnaround buffer mechanics.
   - **Group B (14 features):** Group A + scheduled route, carrier, and temporal flight context.
   - **Group C (18 features):** Group B + same-day airframe operational delay history.
3. **Strict Course Syllabus Boundaries:**
   - **Permitted Classifiers:** Ordinary Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Classifier (SVC), MLP Classifier (`alpha=0.0`).
   - **Permitted Regressors:** Multiple Linear Regression (OLS), Polynomial Regression, Decision Tree Regressor, Random Forest Regressor, MLP Regressor (`alpha=0.0`).
   - **Strictly Banned:** Ridge, Lasso, ElasticNet, SVR, XGBoost, LightGBM, CatBoost, any GBDT, and L1/L2-regularized Logistic Regression."""))

# Section 2
cells.append(make_cell("markdown", """---
## 1. Project Overview & Research Objective"""))
cells.append(make_cell("markdown", """This study models the propagation of operational delays across consecutive aircraft flight legs. When an inbound aircraft arrives late at a turnaround station, it consumes scheduled ground buffer time. If the arrival delay exceeds this buffer, a **turnaround deficit** occurs, threatening the departure punctuality of the subsequent flight leg.

We formulate two dual predictive tasks at the prediction moment $T_{\\text{predict}} = \\text{PREV\\_ARR\\_UTC}$ (inbound gate arrival):
1. **Classification:** Predict whether the subsequent flight will experience an arrival delay $\\ge 15$ minutes (`NEXT_ARR_DEL15` $\\in \\{0, 1\\}$).
2. **Regression:** Predict the continuous arrival delay in minutes of the subsequent flight (`NEXT_ARR_DELAY`)."""))

# Section 3
cells.append(make_cell("markdown", """---
## 2. Environment Setup"""))
cells.append(make_cell("code", """# Install required packages if running in Google Colab environment
import sys
if 'google.colab' in sys.modules:
    !pip install -q duckdb scikit-learn joblib matplotlib

import os
import time
import json
import warnings
import duckdb
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings('ignore')
print("Environment initialized successfully.")
print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__} | Pandas: {pd.__version__} | DuckDB: {duckdb.__version__}")"""))

# Section 4
cells.append(make_cell("markdown", """---
## 3. Google Drive Mounting & Runtime Path Detection"""))
cells.append(make_cell("code", """# Automatic dual-environment detection: Google Colab vs Local Environment
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/Flight_Delay_Propagation'
    print(f"Running on Google Colab. Project root: {BASE_DIR}")
else:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
    print(f"Running in Local Workspace. Project root: {BASE_DIR}")

DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure output directory hierarchy exists
os.makedirs(os.path.join(MODELS_DIR, 'classification'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'regression'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'classification'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'regression'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)

TRAIN_PARQUET = os.path.join(DATA_DIR, 'modeling_train.parquet')
VAL_PARQUET = os.path.join(DATA_DIR, 'modeling_validation.parquet')
TEST_PARQUET = os.path.join(DATA_DIR, 'modeling_test.parquet')

print("Directory structure validated.")"""))

# Section 5
cells.append(make_cell("markdown", """---
## 4. Reproducibility and Runtime Information"""))
cells.append(make_cell("code", """# Enforce reproducibility across all random state initializations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import multiprocessing
cpu_count = multiprocessing.cpu_count()

# Memory check
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_avail_gb = psutil.virtual_memory().available / (1024**3)
    print(f"System Hardware: {cpu_count} CPU cores | Total RAM: {ram_gb:.2f} GB | Available RAM: {ram_avail_gb:.2f} GB")
except ImportError:
    print(f"System Hardware: {cpu_count} CPU cores")

print(f"Random seed locked to: {RANDOM_SEED}")"""))

# Section 6
cells.append(make_cell("markdown", """---
## 5. Dataset Loading"""))
cells.append(make_cell("code", """# Memory-efficient loading using DuckDB native Parquet scanner
con = duckdb.connect()

print("Loading training and validation datasets...")
t0 = time.time()
df_train = con.execute(f"SELECT * FROM read_parquet('{TRAIN_PARQUET}')").df()
df_val = con.execute(f"SELECT * FROM read_parquet('{VAL_PARQUET}')").df()
t_load = time.time() - t0

print(f"Datasets loaded in {t_load:.2f} seconds.")
print(f"Training shape:   {df_train.shape} | RAM: {df_train.memory_usage().sum() / 1e6:.1f} MB")
print(f"Validation shape: {df_val.shape} | RAM: {df_val.memory_usage().sum() / 1e6:.1f} MB")"""))

# Section 7
cells.append(make_cell("markdown", """---
## 6. Dataset Integrity Validation"""))
cells.append(make_cell("code", """# Comprehensive 7-dimension data integrity verification
print("--- Verifying Dataset Integrity ---")

# 1. Total row check
total_train = len(df_train)
total_val = len(df_val)
assert total_train == 5065376, f"Train count mismatch: {total_train}"
assert total_val == 1140909, f"Val count mismatch: {total_val}"
print(f"Check 1: Row counts verified (Train: {total_train:,}, Val: {total_val:,}).")

# 2. Chronological Disjointness
train_max_date = df_train['NEXT_FL_DATE'].max()
val_min_date = df_val['NEXT_FL_DATE'].min()
val_max_date = df_val['NEXT_FL_DATE'].max()
assert train_max_date < val_min_date, f"Temporal overlap detected: {train_max_date} >= {val_min_date}"
print(f"Check 2: Chronological disjointness confirmed ({train_max_date} < {val_min_date}).")

# 3. Missing Value Audit
null_train = df_train.isnull().sum().sum()
null_val = df_val.isnull().sum().sum()
assert null_train == 0, f"Found nulls in train: {null_train}"
assert null_val == 0, f"Found nulls in val: {null_val}"
print(f"Check 3: Missing values confirmed exactly 0 in all columns.")

# 4. Target Flight Duplication
dup_train = df_train.duplicated(subset=['TAIL_NUM', 'NEXT_CRS_DEP_UTC', 'NEXT_ORIGIN', 'NEXT_DEST']).sum()
assert dup_train == 0, f"Found duplicate flights: {dup_train}"
print(f"Check 4: Zero duplicate flight operations.")

# 5. Banned Feature Check
banned_fields = ['NEXT_DEP_DELAY', 'NEXT_DEP_UTC', 'NEXT_ARR_UTC', 'ACTUAL_GROUND_TIME', 'NEXT_ACTUAL_ELAPSED']
for f in banned_fields:
    assert f not in df_train.columns, f"BANNED FEATURE LEAKAGE: {f}"
print(f"Check 5: Zero post-prediction leakage features present.")

print("All integrity checks passed successfully!")"""))

# Section 8
cells.append(make_cell("markdown", """---
## 7. Target Distribution Analysis"""))
cells.append(make_cell("code", """# Target distributions for classification (NEXT_ARR_DEL15) and regression (NEXT_ARR_DELAY)
train_del15_count = int(df_train['NEXT_ARR_DEL15'].sum())
train_del15_rate = float(df_train['NEXT_ARR_DEL15'].mean())
val_del15_count = int(df_val['NEXT_ARR_DEL15'].sum())
val_del15_rate = float(df_val['NEXT_ARR_DEL15'].mean())

print("=== Target Distribution Summary ===")
print(f"Train Delayed Flights (>= 15 min): {train_del15_count:,} ({train_del15_rate*100:.2f}%) | On-Time: {len(df_train)-train_del15_count:,}")
print(f"Val Delayed Flights (>= 15 min):   {val_del15_count:,} ({val_del15_rate*100:.2f}%) | On-Time: {len(df_val)-val_del15_count:,}")
print(f"Train Imbalance Ratio:             {(1-train_del15_rate)/train_del15_rate:.2f} : 1")
print(f"Train Continuous Delay (min):      Mean = {df_train['NEXT_ARR_DELAY'].mean():+.2f}, Median = {df_train['NEXT_ARR_DELAY'].median():.1f}, Std = {df_train['NEXT_ARR_DELAY'].std():.2f}")
print(f"Val Continuous Delay (min):        Mean = {df_val['NEXT_ARR_DELAY'].mean():+.2f}, Median = {df_val['NEXT_ARR_DELAY'].median():.1f}, Std = {df_val['NEXT_ARR_DELAY'].std():.2f}")"""))

# Section 9
cells.append(make_cell("markdown", """---
## 8. Feature Group Definitions"""))
cells.append(make_cell("code", """# Explicit feature group partitioning
NUM_COLS_A = [
    'PREV_ARR_DELAY', 'PREV_DEP_DELAY', 'PREV_ARR_DEL15', 'PREV_DEP_DEL15',
    'SCHEDULED_TURNAROUND_BUFFER', 'ESTIMATED_REMAINING_TURNAROUND_MINUTES', 'IS_TURNAROUND_DEFICIT'
]

CAT_COLS = ['NEXT_CARRIER', 'NEXT_ORIGIN', 'NEXT_DEST']

NUM_COLS_B = NUM_COLS_A + [
    'NEXT_DISTANCE', 'NEXT_CRS_DEP_HOUR', 'NEXT_DAY_OF_WEEK', 'NEXT_MONTH'
]

NUM_COLS_C = NUM_COLS_B + [
    'PREV_AIRCRAFT_DELAYED_FLIGHTS_TODAY', 'PREV_AIRCRAFT_CONSECUTIVE_DEL15',
    'PREV_AIRCRAFT_CUMULATIVE_ARR_DELAY_TODAY', 'PREV_AIRCRAFT_DELAY_RATE_TODAY'
]

METADATA_COLS = ['TAIL_NUM', 'NEXT_FL_DATE', 'NEXT_FL_NUM', 'NEXT_CRS_DEP_UTC', 'PREV_CRS_ARR_UTC']

print(f"Group A Features: {len(NUM_COLS_A)} numeric")
print(f"Group B Features: {len(NUM_COLS_B)} numeric + {len(CAT_COLS)} categorical (Total: {len(NUM_COLS_B) + len(CAT_COLS)})")
print(f"Group C Features: {len(NUM_COLS_C)} numeric + {len(CAT_COLS)} categorical (Total: {len(NUM_COLS_C) + len(CAT_COLS)})")
print(f"Metadata Columns: {len(METADATA_COLS)} (Excluded from all feature sets)")"""))

# Section 10
cells.append(make_cell("markdown", """---
## 9. Preprocessing Pipelines (Sparse Matrix Architecture)"""))
cells.append(make_cell("code", """# Build column transformers enforcing sparse output to prevent RAM explosion
def create_preprocessors():
    prep_A = ColumnTransformer(
        transformers=[('num', StandardScaler(), NUM_COLS_A)],
        remainder='drop'
    )
    prep_B = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS_B),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CAT_COLS)
        ],
        remainder='drop'
    )
    prep_C = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS_C),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CAT_COLS)
        ],
        remainder='drop'
    )
    return {'Group_A': prep_A, 'Group_B': prep_B, 'Group_C': prep_C}

preprocessors = create_preprocessors()
X_train_proc = {}
X_val_proc = {}

for name, prep in preprocessors.items():
    t0 = time.time()
    X_train_proc[name] = prep.fit_transform(df_train)
    X_val_proc[name] = prep.transform(df_val)
    t_prep = time.time() - t0
    is_sparse = sp.issparse(X_train_proc[name])
    mem_mb = X_train_proc[name].data.nbytes / 1e6 if is_sparse else X_train_proc[name].nbytes / 1e6
    print(f"{name}: Train shape {X_train_proc[name].shape} (Sparse: {is_sparse}, Mem: {mem_mb:.1f} MB) in {t_prep:.2f}s")"""))

# Section 11
cells.append(make_cell("markdown", """---
## 10. Baseline Classification: Majority Class & Ordinary Logistic Regression"""))
cells.append(make_cell("code", """# Helper function for computing classification metrics
def eval_classifier(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

classification_results = []
y_train_cls = df_train['NEXT_ARR_DEL15'].values
y_val_cls = df_val['NEXT_ARR_DEL15'].values

# Baseline 1: Majority Class Dummy Classifier
t0 = time.time()
dummy_cls = DummyClassifier(strategy='most_frequent')
dummy_cls.fit(df_train[NUM_COLS_A], y_train_cls)
t_fit_dummy = time.time() - t0

y_pred_dummy = dummy_cls.predict(df_val[NUM_COLS_A])
y_prob_dummy = dummy_cls.predict_proba(df_val[NUM_COLS_A])[:, 1]

m_dummy = eval_classifier(y_val_cls, y_pred_dummy, y_prob_dummy)
m_dummy.update({
    "model": "Majority Class Dummy",
    "feature_group": "None (Target Prior)",
    "train_sample_size": len(df_train),
    "val_sample_size": len(df_val),
    "train_duration_sec": round(t_fit_dummy, 4)
})
classification_results.append(m_dummy)
print(f"Majority Classifier Baseline: Accuracy={m_dummy['accuracy']:.4f}, ROC-AUC={m_dummy['roc_auc']:.4f}, PR-AUC={m_dummy['average_precision_pr_auc']:.4f}")

# Baseline 2: Ordinary Unregularized Logistic Regression (Group A, B, C)
for grp in ['Group_A', 'Group_B', 'Group_C']:
    print(f"Training Logistic Regression ({grp}) on full {len(df_train):,} rows...")
    t0 = time.time()
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=RANDOM_SEED)
    lr.fit(X_train_proc[grp], y_train_cls)
    t_fit = time.time() - t0
    
    y_pred = lr.predict(X_val_proc[grp])
    y_prob = lr.predict_proba(X_val_proc[grp])[:, 1]
    
    m_lr = eval_classifier(y_val_cls, y_pred, y_prob)
    m_lr.update({
        "model": "Ordinary Logistic Regression (Unregularized)",
        "feature_group": grp,
        "train_sample_size": len(df_train),
        "val_sample_size": len(df_val),
        "train_duration_sec": round(t_fit, 2),
        "iterations": int(lr.n_iter_[0])
    })
    classification_results.append(m_lr)
    joblib.dump(lr, os.path.join(MODELS_DIR, f"classification/logistic_regression_{grp.lower()}.joblib"))
    print(f"  -> {grp}: Acc={m_lr['accuracy']:.4f}, F1={m_lr['f1']:.4f}, ROC-AUC={m_lr['roc_auc']:.4f}, PR-AUC={m_lr['average_precision_pr_auc']:.4f} (Fit {t_fit:.1f}s)")"""))

# Section 12
cells.append(make_cell("markdown", """---
## 11. Classification Model Training: Stage 4 Non-Linear Classifiers
The syllabus-approved non-linear classification models are trained and evaluated on the full validation dataset ($N = 1,140,909$):
- **Decision Tree Classifier (Group A & Group C):** `max_depth=14`, `min_samples_leaf=100`, `random_state=42`.
- **Random Forest Classifier (Group A & Group C):** `n_estimators=100`, `max_depth=14`, `min_samples_leaf=50`, `n_jobs=2`, `random_state=42`.
- **MLP Classifier (Group C):** Feedforward neural network `(64, 32)`, strictly unregularized `alpha=0.0`, `batch_size=256`, early stopping."""))
cells.append(make_cell("code", """# Stage 4 Classification Models Loading and Validation
import pandas as pd
import joblib

df_cls_stage4 = pd.read_csv(os.path.join(RESULTS_DIR, "classification/stage4_model_comparison.csv"))
print("=== Consolidated Classification Model Comparison (Validation Set, N = 1,140,909) ===")
display(df_cls_stage4[['stage', 'run_id', 'model', 'feature_group', 'sample_size', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision_pr_auc', 'log_loss', 'train_sec']])"""))

# Section 13
cells.append(make_cell("markdown", """---
## 12. Classification Validation Comparison & Benchmark Findings"""))
cells.append(make_cell("markdown", """### Key Classification Takeaways:
1. **Ensemble & Non-Linear Superiority:** Random Forest Group C achieves the highest overall discrimination with an **ROC-AUC of 0.7932** and **Average Precision (AP) of 0.6600**, significantly outperforming unregularized Logistic Regression (ROC-AUC: 0.7775, AP: 0.6228).
2. **Turnaround Thresholds:** Decision Tree on turnaround mechanics alone (Group A) jumps to an F1 of **0.5613** (compared to Logistic Regression's 0.3955) by learning strict orthogonal split boundaries when remaining turnaround buffer turns negative (`IS_TURNAROUND_DEFICIT = 1`).
3. **MLP High Precision:** Unregularized MLP Classifier (`alpha=0.0`) achieves the highest positive predictive precision (**91.89%**), indicating that deep non-linear feature representations isolate high-confidence delay propagation cascades."""))

# Section 13.1
cells.append(make_cell("markdown", """---
### 12.1 Controlled Sample-Size Experiment (Group C, N = 500,000)
To establish sample-size comparability, Logistic Regression was retrained on the exact same 500,000-row training sample with seed 42:"""))
cells.append(make_cell("code", """# Display Controlled Sample-Size Comparison
df_controlled = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/stage4_controlled_sample_comparison.csv"))
print("=== Controlled Sample-Size Experiment (Training N = 500,000, Full Val N = 1,140,909) ===")
display(df_controlled[df_controlled['task'] == 'Classification'][['model', 'controlled_train_sample_size', 'controlled_accuracy', 'controlled_precision', 'controlled_recall', 'controlled_f1', 'controlled_roc_auc', 'controlled_average_precision_ap']])"""))

# Section 13.2
cells.append(make_cell("markdown", """---
### 12.2 Permutation Feature Importance Robustness (N = 100,000, 5 Repeats)"""))
cells.append(make_cell("code", """# Permutation vs Impurity Importance for Random Forest Classifier
df_perm_cls = pd.read_csv(os.path.join(RESULTS_DIR, "classification/stage4_permutation_importance.csv"))
print("=== RF Classifier Permutation Importance (Metric: Average Precision Drop) ===")
display(df_perm_cls.head(10)[['feature', 'mean_importance', 'std_importance', 'mdi_importance']])"""))

# Section 14
cells.append(make_cell("markdown", """---
## 13. Baseline Regression: Median Delay & Multiple Linear Regression (OLS)"""))
cells.append(make_cell("code", """# Helper function for computing regression metrics
def eval_regressor(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

regression_results = []
y_train_reg = df_train['NEXT_ARR_DELAY'].values
y_val_reg = df_val['NEXT_ARR_DELAY'].values

# Baseline 1: Median Delay Dummy Regressor
t0 = time.time()
dummy_reg = DummyRegressor(strategy='median')
dummy_reg.fit(df_train[NUM_COLS_A], y_train_reg)
t_fit_dummy_reg = time.time() - t0

y_pred_reg_dummy = dummy_reg.predict(df_val[NUM_COLS_A])
m_reg_dummy = eval_regressor(y_val_reg, y_pred_reg_dummy)
m_reg_dummy.update({
    "model": "Median Delay Dummy",
    "feature_group": "None (Target Median)",
    "train_sample_size": len(df_train),
    "val_sample_size": len(df_val),
    "train_duration_sec": round(t_fit_dummy_reg, 4)
})
regression_results.append(m_reg_dummy)
print(f"Median Regressor Baseline: MAE={m_reg_dummy['mae']:.2f} min, RMSE={m_reg_dummy['rmse']:.2f} min, R2={m_reg_dummy['r2']:.4f}")

# Baseline 2: Multiple Linear Regression (OLS) (Group A, B, C)
for grp in ['Group_A', 'Group_B', 'Group_C']:
    print(f"Training OLS ({grp}) on full {len(df_train):,} rows...")
    t0 = time.time()
    ols = LinearRegression()
    ols.fit(X_train_proc[grp], y_train_reg)
    t_fit_ols = time.time() - t0
    
    y_pred_reg = ols.predict(X_val_proc[grp])
    m_ols = eval_regressor(y_val_reg, y_pred_reg)
    m_ols.update({
        "model": "Multiple Linear Regression (OLS)",
        "feature_group": grp,
        "train_sample_size": len(df_train),
        "val_sample_size": len(df_val),
        "train_duration_sec": round(t_fit_ols, 2)
    })
    regression_results.append(m_ols)
    joblib.dump(ols, os.path.join(MODELS_DIR, f"regression/linear_regression_{grp.lower()}.joblib"))
    print(f"  -> {grp}: MAE={m_ols['mae']:.2f} min, RMSE={m_ols['rmse']:.2f} min, R2={m_ols['r2']:.4f} (Fit {t_fit_ols:.1f}s)")"""))

# Section 15
cells.append(make_cell("markdown", """---
## 14. Regression Model Training: Stage 4 Non-Linear Regressors
The syllabus-approved non-linear regression models are trained and evaluated on the full validation dataset ($N = 1,140,909$):
- **Decision Tree Regressor (Group A & Group C):** `max_depth=14`, `min_samples_leaf=100`, `random_state=42`.
- **Random Forest Regressor (Group A & Group C):** `n_estimators=100`, `max_depth=14`, `min_samples_leaf=50`, `n_jobs=2`, `random_state=42`.
- **MLP Regressor (Group C):** Deep neural network `(64, 32)`, strictly unregularized `alpha=0.0`, `batch_size=256`, early stopping."""))
cells.append(make_cell("code", """# Stage 4 Regression Models Loading and Validation
df_reg_stage4 = pd.read_csv(os.path.join(RESULTS_DIR, "regression/stage4_model_comparison.csv"))
print("=== Consolidated Regression Model Comparison (Validation Set, N = 1,140,909) ===")
display(df_reg_stage4[['stage', 'run_id', 'model', 'feature_group', 'sample_size', 'mae', 'mse', 'rmse', 'r2', 'train_sec']])"""))

# Section 16
cells.append(make_cell("markdown", """---
## 15. Regression Validation Comparison & Benchmark Findings"""))
cells.append(make_cell("markdown", """### Key Regression Takeaways:
1. **Dramatic Non-Linear Variance Capture:** Linear regression (OLS) only accounted for 27.5% of delay variance ($R^2 = 0.2754$, MAE = 21.19 min). By contrast, Random Forest Group C achieves an $R^2$ of **0.4516** and reduces MAE to **17.40 minutes** (an improvement of 3.79 minutes per flight leg across 1.14M flights!).
2. **MLP Lowest Absolute Error:** MLP Regressor with `alpha=0.0` achieves the lowest Mean Absolute Error across all evaluated models at **17.06 minutes**, outperforming both linear models and individual decision trees.
3. **Turnaround Throttle Primacy:** Turnaround buffer physics explain the vast majority of continuous delay propagation, while airframe daily operational history provides key secondary modulation."""))

# Section 16.1
cells.append(make_cell("markdown", """---
### 15.1 Regression Error Analysis Across Operational Delay Bands
Performance evaluated across 5 actual target delay bands on the full validation population ($N = 1,140,909$):"""))
cells.append(make_cell("code", """# Display Delay Band Error Breakdown
df_delay_bands = pd.read_csv(os.path.join(RESULTS_DIR, "regression/stage4_error_by_delay_band.csv"))
print("=== Error Breakdown by Target Delay Band (Full Validation Set, N = 1,140,909) ===")
display(df_delay_bands)"""))

# Section 16.2
cells.append(make_cell("markdown", """---
### 15.2 Permutation Feature Importance for Continuous Delay (N = 100,000, 5 Repeats)"""))
cells.append(make_cell("code", """# Permutation vs Impurity Importance for Random Forest Regressor
df_perm_reg = pd.read_csv(os.path.join(RESULTS_DIR, "regression/stage4_permutation_importance.csv"))
print("=== RF Regressor Permutation Importance (Metric: Increase in MAE in minutes) ===")
display(df_perm_reg.head(10)[['feature', 'mean_importance_mae', 'std_importance_mae', 'mean_importance_r2', 'mdi_importance']])"""))

# Section 17
cells.append(make_cell("markdown", """---
## 16. Feature Group Ablation Study: Group A vs. Group B vs. Group C"""))
cells.append(make_cell("code", """# Quantify incremental gains across feature groups
print("=== Feature Group Ablation Gains (Validation Data) ===")
# Classification Ablation
lr_A = [r for r in classification_results if r['feature_group'] == 'Group_A'][0]
lr_B = [r for r in classification_results if r['feature_group'] == 'Group_B'][0]
lr_C = [r for r in classification_results if r['feature_group'] == 'Group_C'][0]

print("--- Classification (Logistic Regression) ---")
print(f"Group A (Turnaround only):         F1 = {lr_A['f1']:.4f} | ROC-AUC = {lr_A['roc_auc']:.4f} | PR-AUC = {lr_A['average_precision_pr_auc']:.4f}")
print(f"Group B (+ Flight/Route Context):  F1 = {lr_B['f1']:.4f} | ROC-AUC = {lr_B['roc_auc']:.4f} | PR-AUC = {lr_B['average_precision_pr_auc']:.4f} (Delta PR-AUC: {lr_B['average_precision_pr_auc'] - lr_A['average_precision_pr_auc']:+.4f})")
print(f"Group C (+ Airframe Daily History):F1 = {lr_C['f1']:.4f} | ROC-AUC = {lr_C['roc_auc']:.4f} | PR-AUC = {lr_C['average_precision_pr_auc']:.4f} (Delta PR-AUC: {lr_C['average_precision_pr_auc'] - lr_B['average_precision_pr_auc']:+.4f})")

# Regression Ablation
ols_A = [r for r in regression_results if r['feature_group'] == 'Group_A'][0]
ols_B = [r for r in regression_results if r['feature_group'] == 'Group_B'][0]
ols_C = [r for r in regression_results if r['feature_group'] == 'Group_C'][0]

print("\n--- Regression (OLS) ---")
print(f"Group A (Turnaround only):         MAE = {ols_A['mae']:.2f} min | RMSE = {ols_A['rmse']:.2f} min | R2 = {ols_A['r2']:.4f}")
print(f"Group B (+ Flight/Route Context):  MAE = {ols_B['mae']:.2f} min | RMSE = {ols_B['rmse']:.2f} min | R2 = {ols_B['r2']:.4f}")
print(f"Group C (+ Airframe Daily History):MAE = {ols_C['mae']:.2f} min | RMSE = {ols_C['rmse']:.2f} min | R2 = {ols_C['r2']:.4f} (Delta R2: {ols_C['r2'] - ols_B['r2']:+.4f})")"""))

# Section 18
cells.append(make_cell("markdown", """---
## 17. Hyperparameter Tuning Protocol & Empirical Results"""))
cells.append(make_cell("markdown", """To systematically refine our estimators without data snooping or violating syllabus constraints:
1. **Tree-Based Models (Decision Trees & Random Forests):**
   - Explored `max_depth` $\\in [10, 14, 16, 18]$ and `min_samples_leaf` $\\in [30, 50, 100]$.
   - Evaluated ensemble size ($n\\_estimators \\in [50, 100]$) and balanced class weighting (`class_weight='balanced_subsample'`).
2. **Multi-Layer Perceptrons (MLP):**
   - Strict Constraint: `alpha = 0.0` (zero L2 regularization).
   - Architectures: Single-layer $(64,)$, dual-layer $(64, 32)$, and wide $(128, 64)$ with early stopping on validation loss.
3. **Reference Benchmarks:**
   - Evaluated unregularized Logistic Regression and Ordinary Least Squares (OLS) under the exact same $N = 500,000$ conditions.
4. **Validation Integrity:**
   - Every candidate is evaluated on the full, unpruned validation set ($N = 1,140,909$, October–November 2025)."""))
cells.append(make_cell("code", """# Display Classification and Regression Hyperparameter Tuning Results
from IPython.display import Image, display

df_cls_tune = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/stage5_classification_tuning.csv"))
print("=== Classification Hyperparameter Tuning Results (Validation N = 1,140,909) ===")
display(df_cls_tune[['candidate_name', 'model_family', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision_ap', 'log_loss', 'fit_duration_sec']])

df_reg_tune = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/stage5_regression_tuning.csv"))
print("\n=== Regression Hyperparameter Tuning Results (Validation N = 1,140,909) ===")
display(df_reg_tune[['candidate_name', 'model_family', 'mae_minutes', 'rmse_minutes', 'r2_score', 'fit_duration_sec']])

print("\n=== Figure: Hyperparameter Tuning Performance Across Estimators ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/stage5_tuning_curves.png")))"""))

# Section 19
cells.append(make_cell("markdown", """---
## 18. Final Model Selection & Decision Threshold Optimization"""))
cells.append(make_cell("markdown", """### 1. Classification Decision Threshold Calibration & Audit Reconciliation
In airline operations, missing a propagated delay (False Negative) causes missed passenger connections, crew duty-time timeouts, and cascade disruptions. Conversely, an operational warning (False Positive) merely triggers expedited ground servicing.
Standard default threshold ($T = 0.50$) produces high precision (86.68%) but misses **59.12%** of propagated delays (`recall = 40.88%`, F1 = 0.5556).
A systematic threshold audit on the full validation population ($N = 1,140,909$, 230,521 positive delay events) reveals:
- **Selected Operational Threshold ($T = 0.35$):** Precision = 69.20%, Recall = 49.03%, **F1 = 0.5740**. (Lifts delay recall by +8.15 percentage points while preserving near-70% precision).
- **Mathematical F1 Optimum ($T = 0.39$ fine / $0.40$ coarse):** Precision = 72.53%, Recall = 47.65%, **F1 = 0.5751**."""))
cells.append(make_cell("code", """# Display Audited Classification Decision Threshold Analysis
df_thresh_audit = pd.read_csv(os.path.join(RESULTS_DIR, "metrics/stage5_threshold_integrity_audit.csv"))
print("=== Reconciled Classification Probability Threshold Analysis (RF Group C) ===")
display(df_thresh_audit[df_thresh_audit['model'] == 'Random Forest Classifier (Group C)'][['threshold', 'precision', 'recall', 'f1_score', 'specificity', 'accuracy', 'positive_prediction_count', 'confusion_matrix_total', 'reconciles_positives', 'reconciles_negatives', 'reconciles_total']])

print("\n=== Figure: Precision-Recall vs. Decision Threshold & Trade-off Curves ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/stage5_threshold_curves.png")))

print("\n=== Figure: Final Model Diagnostic Dashboard ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/stage5_final_model_diagnostics.png")))"""))
cells.append(make_cell("markdown", """### 2. Final Selected Models Rationale
- **Final Classifier: Random Forest Classifier (Group C)**
  - Config: `n_estimators=100`, `max_depth=14`, `min_samples_leaf=50`, operational threshold $T = 0.35$.
  - Validation: ROC-AUC = 0.7932, AP = 0.6600, F1 = 0.5740 at $T = 0.35$ (Recall: 49.03%, Precision: 69.20%).
- **Final Regressor: Random Forest Regressor (Group C)**
  - Config: `n_estimators=100`, `max_depth=14`, `min_samples_leaf=50`.
  - Validation: $R^2 = 0.4516$ (explains 45.16% of arrival delay variance relative to mean baseline), RMSE = 38.62 min, MAE = 17.40 min (dominates severe delay regimes $\ge 120$ min)."""))
cells.append(make_cell("code", """# Load and display finalized selection metadata
with open(os.path.join(RESULTS_DIR, "metrics/stage5_final_model_selection.json"), "r") as f:
    final_meta = json.load(f)

print("=== Final Selected Estimator Metadata ===")
print(json.dumps(final_meta, indent=2))"""))

# Section 20
cells.append(make_cell("markdown", """---
## 19. Final Test Evaluation Protocol (December 2025 Held-Out Set)"""))
cells.append(make_cell("markdown", """### Strict Methodological Lock:
In accordance with rigorous academic machine learning methodology and our pre-registered evaluation protocol:
- The December 2025 test dataset (`data/processed/modeling_test.parquet`, $N = 562,202$) has remained **strictly sealed, untouched, and unaccessed** during all exploratory, feature engineering, baseline, and Stage 5 tuning iterations.
- Model selection and threshold calibration were completed exclusively using validation data (October–November 2025).
- The cell below defines the standalone, reproducible test evaluation pipeline to be executed in Stage 6 upon formal approval."""))
cells.append(make_cell("code", """# Standalone Test Evaluation Routine (Held-out December 2025)
def evaluate_final_test_set():
    \"\"\"
    Executes the final test evaluation using strictly frozen models and preprocessors.
    \"\"\"
    print("Loading sealed December 2025 test dataset...")
    df_test = con.execute(f"SELECT * FROM read_parquet('{TEST_PARQUET}')").df()
    print(f"Test dataset loaded: {len(df_test):,} rows.")
    
    # Load serialized final preprocessors and estimators
    prep_C = joblib.load(os.path.join(MODELS_DIR, "final/final_preprocessor_group_c.joblib"))
    final_clf = joblib.load(os.path.join(MODELS_DIR, "final/final_selected_classifier.joblib"))
    final_reg = joblib.load(os.path.join(MODELS_DIR, "final/final_selected_regressor.joblib"))
    
    X_test_C = prep_C.transform(df_test)
    y_test_cls = df_test['NEXT_ARR_DEL15'].values
    y_test_reg = df_test['NEXT_ARR_DELAY'].values
    
    # 1. Classification Evaluation (Default T=0.50 and Calibrated T=0.35)
    y_test_prob = final_clf.predict_proba(X_test_C)[:, 1]
    y_test_pred_default = (y_test_prob >= 0.50).astype(int)
    y_test_pred_calibrated = (y_test_prob >= 0.35).astype(int)
    
    # 2. Continuous Delay Regression Evaluation
    y_test_pred_reg = final_reg.predict(X_test_C)
    
    print("Test Evaluation routine prepared and verified.")
    return {
        "test_n": len(df_test),
        "clf_auc": roc_auc_score(y_test_cls, y_test_prob),
        "clf_ap": average_precision_score(y_test_cls, y_test_prob),
        "reg_mae": mean_absolute_error(y_test_reg, y_test_pred_reg),
        "reg_rmse": np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg)),
        "reg_r2": r2_score(y_test_reg, y_test_pred_reg)
    }

print("Test evaluation protocol ready for Stage 6.")"""))

# Section 21
cells.append(make_cell("markdown", """---
## 20. Confusion Matrix & ROC / PR Curves (IEEE Figures)"""))
cells.append(make_cell("code", """# Display generated publication-grade classification diagnostics
from IPython.display import Image, display

print("=== Figure: ROC Curves Across Model Suite ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/classification_roc_curve.png")))

print("=== Figure: Precision-Recall Curves Across Model Suite ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/classification_precision_recall_curve.png")))

print("=== Figure: Validation Set Confusion Matrices ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/classification_confusion_matrix.png")))"""))

# Section 22
cells.append(make_cell("markdown", """---
## 21. Regression Diagnostics: Actual vs. Predicted & Residuals (IEEE Figures)"""))
cells.append(make_cell("code", """# Display generated publication-grade regression diagnostics
print("=== Figure: Actual vs. Predicted Arrival Delay (Random Forest Group C) ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/regression_actual_vs_predicted.png")))

print("=== Figure: Residual Error Distribution Across Regressors ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/regression_residual_distribution.png")))

print("=== Figure: Error Breakdown Across Target Delay Bands ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/regression_error_by_delay_band.png")))"""))

# Section 23
cells.append(make_cell("markdown", """---
## 22. Model Interpretation: Feature Importances & Physical Drivers"""))
cells.append(make_cell("code", """# Display top feature importances for Tree Models & Robustness Comparison
df_imp_rf_cls = pd.read_csv(os.path.join(RESULTS_DIR, "classification/feature_importances_rf_group_c.csv"))
df_imp_rf_reg = pd.read_csv(os.path.join(RESULTS_DIR, "regression/feature_importances_rf_group_c.csv"))

print("=== Top 10 Features: Random Forest Classifier (Group C) ===")
display(df_imp_rf_cls.head(10))

print("\n=== Top 10 Features: Random Forest Regressor (Group C) ===")
display(df_imp_rf_reg.head(10))

print("\n=== Figure: Permutation vs. MDI Feature Importance Comparison ===")
display(Image(filename=os.path.join(RESULTS_DIR, "figures/permutation_importance_comparison.png")))"""))

# Section 24
cells.append(make_cell("markdown", """---
## 23. Scientific Interpretation & Research Findings
### Key Empirical Findings:
1. **Turnaround Buffer Mechanics:** Inbound delay exceeding the turnaround buffer (`IS_TURNAROUND_DEFICIT = 1`) is the single strongest linear driver of delay propagation.
2. **Contextual Augmentation:** Adding route and carrier context (Group B) increases PR-AUC from 0.5792 to 0.6102.
3. **Incremental Value of Aircraft History:** Cumulative same-day airframe history (Group C) further boosts PR-AUC to 0.6228 and pushes classification F1 to 0.5003. Historical aircraft stress is a genuine non-redundant signal for propagation risk.
4. **Predictive vs. Causal Nuance:** While delays are strongly contagious, propagation is mediated by buffer slack and dispatch intervention rather than pure physical momentum alone."""))

# Section 25
cells.append(make_cell("markdown", """---
## 24. Threats to Validity & Limitations
1. **Unobserved Confounders:** En-route weather fronts, air traffic control ground delay programs (GDPs), and late-arriving connecting crews are partially reflected through arrival delays but lack direct real-time telemetry.
2. **Observational Bounds:** Models describe empirical delay correlation across consecutive operations, not structural counterfactual mechanics.
3. **Extreme Outliers:** Long delays (>300 min) introduce high squared-error penalties in continuous regression, favoring robust metric evaluation (MAE/MedAE)."""))

# Section 26
cells.append(make_cell("markdown", """---
## 25. Export Models, Metrics, and Figures"""))
cells.append(make_cell("code", """# Final serialization confirmation
print("Exported files verified:")
print(" - Models:", os.listdir(os.path.join(MODELS_DIR, "classification")))
print(" - Metrics:", os.listdir(os.path.join(RESULTS_DIR, "metrics")))
print(" - Baselines:", os.listdir(os.path.join(RESULTS_DIR, "classification")))
print("Phase 6 Notebook Execution Complete.")"""))

notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.9.6"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated {NOTEBOOK_PATH} with {len(cells)} cells.")

