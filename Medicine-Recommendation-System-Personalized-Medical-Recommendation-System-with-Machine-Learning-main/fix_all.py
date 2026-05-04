"""
fix_all.py
----------
1. Fixes disease name mismatches between Training.csv and all lookup CSVs
2. Retrains and saves the model fresh
3. Verifies predictions work end-to-end
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Exact names in Training.csv ───────────────────────────────────────────────
train = pd.read_csv("datasets/Training.csv")
train["prognosis"] = train["prognosis"].str.strip()
diseases_in_training = sorted(train["prognosis"].unique())
print(f"Training diseases ({len(diseases_in_training)}):", diseases_in_training)

# ── Fix name mismatches in lookup CSVs ───────────────────────────────────────
# Mapping: {name in CSV -> correct name in Training.csv}
NAME_FIX = {
    "Peptic ulcer disease":                     "Peptic ulcer diseae",   # keep training typo
    "(vertigo) Paroymsal Positional Vertigo":   "(vertigo) Paroymsal  Positional Vertigo",
    # also strip trailing spaces wherever they appear
}

def fix_disease_column(df, col):
    df[col] = df[col].str.strip()
    df[col] = df[col].replace(NAME_FIX)
    return df

print("\nFixing description.csv ...")
desc = pd.read_csv("datasets/description.csv")
desc = fix_disease_column(desc, "Disease")
desc.to_csv("datasets/description.csv", index=False)

print("Fixing precautions_df.csv ...")
prec = pd.read_csv("datasets/precautions_df.csv")
prec = fix_disease_column(prec, "Disease")
prec.to_csv("datasets/precautions_df.csv", index=False)

print("Fixing medications.csv ...")
meds = pd.read_csv("datasets/medications.csv")
meds = fix_disease_column(meds, "Disease")
meds.to_csv("datasets/medications.csv", index=False)

print("Fixing diets.csv ...")
diet = pd.read_csv("datasets/diets.csv")
diet = fix_disease_column(diet, "Disease")
diet.to_csv("datasets/diets.csv", index=False)

print("Fixing workout_df.csv ...")
work = pd.read_csv("datasets/workout_df.csv")
work = fix_disease_column(work, "disease")
work.to_csv("datasets/workout_df.csv", index=False)

# ── Verify coverage ───────────────────────────────────────────────────────────
print("\n--- Coverage Check ---")
for df, name, col in [
    (pd.read_csv("datasets/description.csv"),    "description",  "Disease"),
    (pd.read_csv("datasets/precautions_df.csv"), "precautions",  "Disease"),
    (pd.read_csv("datasets/medications.csv"),    "medications",  "Disease"),
    (pd.read_csv("datasets/diets.csv"),          "diets",        "Disease"),
    (pd.read_csv("datasets/workout_df.csv"),     "workouts",     "disease"),
]:
    covered = set(df[col].str.strip())
    missing = [d for d in diseases_in_training if d not in covered]
    print(f"  {name}: missing {len(missing)} -> {missing}")

# ── Retrain model fresh ───────────────────────────────────────────────────────
print("\n--- Retraining Model ---")
X = train.drop("prognosis", axis=1)
y = train["prognosis"]
feature_names = list(X.columns)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

svc = SVC(kernel="linear", C=1.0, probability=False, random_state=42)
svc.fit(X_train, y_train)
acc = accuracy_score(y_test, svc.predict(X_test))
print(f"  Test Accuracy: {acc*100:.2f}%")

with open("models/svc.pkl", "wb") as f:
    pickle.dump(svc, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# ── End-to-end prediction tests ───────────────────────────────────────────────
print("\n--- Prediction Tests ---")
sd = {name: idx for idx, name in enumerate(feature_names)}

def predict(syms):
    vec = np.zeros(len(feature_names))
    matched = []
    for s in syms:
        key = s.strip().lower().replace(" ", "_")
        if key in sd:
            vec[sd[key]] = 1
            matched.append(key)
    df_in = pd.DataFrame([vec], columns=feature_names)
    idx = svc.predict(df_in)[0]
    return le.inverse_transform([idx])[0], matched

tests = [
    (["itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"], "Fungal infection"),
    (["continuous_sneezing", "shivering", "chills", "watering_from_eyes"], "Allergy"),
    (["pus_filled_pimples", "blackheads", "scurring", "skin_peeling"], "Acne"),
    (["chest_pain", "fast_heart_rate", "breathlessness", "sweating"], "Heart attack"),
    (["polyuria", "increased_appetite", "irregular_sugar_level", "fatigue", "weight_loss"], "Diabetes"),
    (["joint_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "movement_stiffness"], "Osteoarthristis"),
    (["high_fever", "chills", "muscle_pain", "sweating", "headache", "nausea"], "Malaria"),
    (["fatigue", "yellowish_skin", "dark_urine", "abdominal_pain", "loss_of_appetite"], "Hepatitis B"),
]

passed = 0
for syms, expected in tests:
    result, matched = predict(syms)
    ok = result == expected
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {syms[:3]}... -> {result} (expected: {expected})")

print(f"\n  {passed}/{len(tests)} tests passed")
print("\nDone! All datasets fixed and model retrained.")
