"""
train_model.py
--------------
Run this once to train and save the model.
Everything (features, labels, symptom names) is derived directly from Training.csv.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading Training.csv ...")
df = pd.read_csv("datasets/Training.csv")

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Features = all columns except 'prognosis'
X = df.drop("prognosis", axis=1)
y = df["prognosis"].str.strip()          # strip disease name whitespace too

feature_names = list(X.columns)          # exact symptom names from CSV
print(f"  Symptoms (features): {len(feature_names)}")
print(f"  Diseases (classes) : {y.nunique()}")

# ── 2. Encode target labels ───────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Diseases: {list(le.classes_)}")

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── 4. Train SVC ──────────────────────────────────────────────────────────────
print("\nTraining SVC (kernel=linear) ...")
svc = SVC(kernel="linear", C=1.0, probability=False, random_state=42)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {acc * 100:.2f}%")

# ── 5. Save everything ────────────────────────────────────────────────────────
with open("models/svc.pkl", "wb") as f:
    pickle.dump(svc, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nSaved:")
print("  models/svc.pkl")
print("  models/label_encoder.pkl")
print("  models/feature_names.pkl")
print("\nDone! Model is ready.")
