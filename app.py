import os
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 — Classification Models")

MODEL_DIR = "model/saved_models"
TARGET_COL = "label"

# Show overall comparison table (from evaluation script)
st.subheader("Comparison Table (Test Split)")
metrics_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
if os.path.exists(metrics_path):
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
else:
    st.warning("Run evaluation to generate metrics_summary.csv")

st.divider()

model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

chosen_model = st.selectbox("Select Model", list(model_map.keys()))
uploaded = st.file_uploader("Upload CSV (must include 'label' column)", type=["csv"])

def compute_metrics(y_true, y_pred, y_prob=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    out["AUC"] = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return out

if uploaded is None:
    st.info("Upload a CSV to run predictions (use data/test_upload.csv).")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET_COL not in df.columns:
    st.error("Uploaded CSV must include 'label' column.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Enforce same feature columns as training
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
X = X[feature_cols]

model = joblib.load(os.path.join(MODEL_DIR, model_map[chosen_model]))
y_pred = model.predict(X)

y_prob = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        y_prob = proba[:, 1]

st.subheader(f"Metrics for: {chosen_model}")
m = compute_metrics(y, y_pred, y_prob)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
c2.metric("Precision", f"{m['Precision']:.4f}")
c3.metric("Recall", f"{m['Recall']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("F1", f"{m['F1']:.4f}")
c5.metric("MCC", f"{m['MCC']:.4f}")
c6.metric("AUC", "N/A" if m["AUC"] is None else f"{m['AUC']:.4f}")

st.divider()
st.subheader("Classification Report")
st.text(classification_report(y, y_pred, zero_division=0))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y, y_pred))
