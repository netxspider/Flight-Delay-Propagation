"""
Stage 3: Baseline Model Training and Evaluation Pipeline.
Research Project: Flight Delays Are Contagious
BTech CSE CA2 / IEEE Research Paper Implementation

This module trains and evaluates the baseline models on the full training dataset
(modeling_train.parquet, 5,065,376 rows) and evaluates them on the validation dataset
(modeling_validation.parquet, 1,140,909 rows):
1. Classification Baselines:
   - Majority Class Dummy Classifier
   - Ordinary Unregularized Logistic Regression (Group A, Group B, Group C)
2. Regression Baselines:
   - Median Dummy Regressor
   - Multiple Linear Regression / OLS (Group A, Group B, Group C)
"""

import os
import time
import json
import duckdb
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TRAIN_PARQUET = os.path.join(DATA_DIR, "modeling_train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "modeling_validation.parquet")

# Feature group specifications
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


def ensure_directories():
    os.makedirs(os.path.join(MODELS_DIR, "classification"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "regression"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "classification"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "regression"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)


def load_datasets():
    print("--- 1. Loading Modeling Datasets via DuckDB ---")
    con = duckdb.connect()
    t0 = time.time()
    df_train = con.execute(f"SELECT * FROM read_parquet('{TRAIN_PARQUET}')").df()
    df_val = con.execute(f"SELECT * FROM read_parquet('{VAL_PARQUET}')").df()
    print(f"Loaded Train ({len(df_train):,} rows) and Validation ({len(df_val):,} rows) in {time.time()-t0:.2f}s")
    return df_train, df_val


def build_preprocessors():
    """Build leakage-safe preprocessors fitted exclusively on training data."""
    # Group A: numeric only
    prep_A = ColumnTransformer(
        transformers=[('num', StandardScaler(), NUM_COLS_A)],
        remainder='drop'
    )
    # Group B: numeric + sparse categorical
    prep_B = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS_B),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CAT_COLS)
        ],
        remainder='drop'
    )
    # Group C: full feature set
    prep_C = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS_C),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CAT_COLS)
        ],
        remainder='drop'
    )
    return {'Group_A': prep_A, 'Group_B': prep_B, 'Group_C': prep_C}


def evaluate_classifier(y_true, y_pred, y_prob):
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


def evaluate_regressor(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def run_stage3_baselines():
    ensure_directories()
    df_train, df_val = load_datasets()

    y_train_cls = df_train['NEXT_ARR_DEL15'].values
    y_val_cls = df_val['NEXT_ARR_DEL15'].values

    y_train_reg = df_train['NEXT_ARR_DELAY'].values
    y_val_reg = df_val['NEXT_ARR_DELAY'].values

    results_cls = []
    results_reg = []
    metadata_log = {
        "experiment": "Phase 6 Stage 3: Baseline Models",
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "random_seed": 42,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    print("\n--- 2. Fitting Preprocessing Pipelines on Training Data Strictly ---")
    preprocessors = build_preprocessors()
    X_train_proc = {}
    X_val_proc = {}

    for grp_name, prep in preprocessors.items():
        t0 = time.time()
        X_train_proc[grp_name] = prep.fit_transform(df_train)
        X_val_proc[grp_name] = prep.transform(df_val)
        print(f"Preprocessed {grp_name}: Train shape {X_train_proc[grp_name].shape}, Val shape {X_val_proc[grp_name].shape} (in {time.time()-t0:.2f}s)")

    # ---------------------------------------------------------
    # Baseline 1: Majority-Class Classifier
    # ---------------------------------------------------------
    print("\n--- 3. Training Classification Baseline: Majority Class ---")
    t0 = time.time()
    dummy_cls = DummyClassifier(strategy='most_frequent')
    dummy_cls.fit(df_train[NUM_COLS_A], y_train_cls)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_dummy = dummy_cls.predict(df_val[NUM_COLS_A])
    y_prob_dummy = dummy_cls.predict_proba(df_val[NUM_COLS_A])[:, 1]
    t_pred = time.time() - t0

    metrics_dummy = evaluate_classifier(y_val_cls, y_pred_dummy, y_prob_dummy)
    metrics_dummy.update({
        "model": "Majority Class Dummy",
        "feature_group": "None (Target Prior)",
        "train_sample_size": len(df_train),
        "val_sample_size": len(df_val),
        "train_duration_sec": round(t_train, 4),
        "pred_duration_sec": round(t_pred, 4)
    })
    results_cls.append(metrics_dummy)
    joblib.dump(dummy_cls, os.path.join(MODELS_DIR, "classification/dummy_majority_classifier.joblib"))
    print(f"Majority Classifier: Accuracy={metrics_dummy['accuracy']:.4f}, F1={metrics_dummy['f1']:.4f}, ROC-AUC={metrics_dummy['roc_auc']:.4f}, AP={metrics_dummy['average_precision_pr_auc']:.4f}")

    # ---------------------------------------------------------
    # Baseline 2: Median Regressor
    # ---------------------------------------------------------
    print("\n--- 4. Training Regression Baseline: Median Delay ---")
    t0 = time.time()
    dummy_reg = DummyRegressor(strategy='median')
    dummy_reg.fit(df_train[NUM_COLS_A], y_train_reg)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_reg_dummy = dummy_reg.predict(df_val[NUM_COLS_A])
    t_pred = time.time() - t0

    metrics_reg_dummy = evaluate_regressor(y_val_reg, y_pred_reg_dummy)
    metrics_reg_dummy.update({
        "model": "Median Delay Dummy",
        "feature_group": "None (Target Median)",
        "train_sample_size": len(df_train),
        "val_sample_size": len(df_val),
        "train_duration_sec": round(t_train, 4),
        "pred_duration_sec": round(t_pred, 4)
    })
    results_reg.append(metrics_reg_dummy)
    joblib.dump(dummy_reg, os.path.join(MODELS_DIR, "regression/dummy_median_regressor.joblib"))
    print(f"Median Regressor: MAE={metrics_reg_dummy['mae']:.2f} min, RMSE={metrics_reg_dummy['rmse']:.2f} min, R2={metrics_reg_dummy['r2']:.4f}")

    # ---------------------------------------------------------
    # Baseline 3: Ordinary Unregularized Logistic Regression (A, B, C)
    # ---------------------------------------------------------
    print("\n--- 5. Training Ordinary Unregularized Logistic Regression ---")
    for grp_name in ['Group_A', 'Group_B', 'Group_C']:
        print(f"Training Logistic Regression ({grp_name}) on full {len(df_train):,} training records...")
        t0 = time.time()
        # Ordinary unregularized: penalty=None
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=42)
        lr.fit(X_train_proc[grp_name], y_train_cls)
        t_train = time.time() - t0

        t0 = time.time()
        y_pred = lr.predict(X_val_proc[grp_name])
        y_prob = lr.predict_proba(X_val_proc[grp_name])[:, 1]
        t_pred = time.time() - t0

        m = evaluate_classifier(y_val_cls, y_pred, y_prob)
        m.update({
            "model": "Ordinary Logistic Regression (Unregularized)",
            "feature_group": grp_name,
            "train_sample_size": len(df_train),
            "val_sample_size": len(df_val),
            "train_duration_sec": round(t_train, 2),
            "pred_duration_sec": round(t_pred, 2),
            "iterations": int(lr.n_iter_[0])
        })
        results_cls.append(m)
        joblib.dump(lr, os.path.join(MODELS_DIR, f"classification/logistic_regression_{grp_name.lower()}.joblib"))
        print(f"  -> {grp_name}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['average_precision_pr_auc']:.4f}, LogLoss={m['log_loss']:.4f} (Fit {t_train:.1f}s, Iters {lr.n_iter_[0]})")

    # ---------------------------------------------------------
    # Baseline 4: Multiple Linear Regression / OLS (A, B, C)
    # ---------------------------------------------------------
    print("\n--- 6. Training Multiple Linear Regression (OLS) ---")
    for grp_name in ['Group_A', 'Group_B', 'Group_C']:
        print(f"Training OLS ({grp_name}) on full {len(df_train):,} training records...")
        t0 = time.time()
        ols = LinearRegression()
        ols.fit(X_train_proc[grp_name], y_train_reg)
        t_train = time.time() - t0

        t0 = time.time()
        y_pred_reg = ols.predict(X_val_proc[grp_name])
        t_pred = time.time() - t0

        m_reg = evaluate_regressor(y_val_reg, y_pred_reg)
        m_reg.update({
            "model": "Multiple Linear Regression (OLS)",
            "feature_group": grp_name,
            "train_sample_size": len(df_train),
            "val_sample_size": len(df_val),
            "train_duration_sec": round(t_train, 2),
            "pred_duration_sec": round(t_pred, 2)
        })
        results_reg.append(m_reg)
        joblib.dump(ols, os.path.join(MODELS_DIR, f"regression/linear_regression_{grp_name.lower()}.joblib"))
        print(f"  -> {grp_name}: MAE={m_reg['mae']:.2f} min, RMSE={m_reg['rmse']:.2f} min, R2={m_reg['r2']:.4f} (Fit {t_train:.1f}s)")

    # ---------------------------------------------------------
    # Save Metrics & Tabular Outputs
    # ---------------------------------------------------------
    print("\n--- 7. Saving Metrics and CSV Summaries ---")
    df_cls = pd.DataFrame(results_cls)
    df_reg = pd.DataFrame(results_reg)

    cls_csv_path = os.path.join(RESULTS_DIR, "classification/baseline_model_comparison.csv")
    reg_csv_path = os.path.join(RESULTS_DIR, "regression/baseline_model_comparison.csv")
    metrics_json_path = os.path.join(RESULTS_DIR, "metrics/stage3_baseline_metrics.json")

    df_cls.to_csv(cls_csv_path, index=False)
    df_reg.to_csv(reg_csv_path, index=False)

    full_output = {
        "metadata": metadata_log,
        "classification_baselines": results_cls,
        "regression_baselines": results_reg
    }
    with open(metrics_json_path, 'w') as f:
        json.dump(full_output, f, indent=2)

    print(f"Saved: {cls_csv_path}")
    print(f"Saved: {reg_csv_path}")
    print(f"Saved: {metrics_json_path}")
    print("\nStage 3 Baselines Complete!")


if __name__ == "__main__":
    run_stage3_baselines()

