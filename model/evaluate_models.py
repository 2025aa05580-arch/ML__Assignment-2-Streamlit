import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

MODEL_DIR = "model/saved_models"

X_train, X_test, y_train, y_test = joblib.load(os.path.join(MODEL_DIR, "data_split.pkl"))

model_names = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost"
]

rows = []

for m in model_names:
    model = joblib.load(os.path.join(MODEL_DIR, f"{m}.pkl"))
    y_pred = model.predict(X_test)

    # AUC needs probabilities
    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

    rows.append({
        "ML Model Name": m,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)

print("✅ Metrics saved to model/saved_models/metrics_summary.csv")
print(df.to_string(index=False))
