import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = "data/dataset.csv"
TARGET_COL = "label"
MODEL_DIR = "model/saved_models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save columns and split for downstream use
joblib.dump(list(X_train.columns), os.path.join(MODEL_DIR, "feature_columns.pkl"))
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(MODEL_DIR, "data_split.pkl"))

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    print(f"✅ Saved model: {name}.pkl")

print("✅ Training done.")
