"""
main.py  –  Medicine Recommendation System
All symptoms, diseases, and model data are loaded from files — nothing is hardcoded.
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ── Load lookup datasets ──────────────────────────────────────────────────────
sym_des     = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout     = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets       = pd.read_csv("datasets/diets.csv")

# Strip whitespace from Disease column in all datasets
for _df in [precautions, description, medications, diets]:
    if "Disease" in _df.columns:
        _df["Disease"] = _df["Disease"].str.strip()
if "disease" in workout.columns:
    workout["disease"] = workout["disease"].str.strip()

# ── Load model artifacts (all derived from Training.csv at train time) ────────
svc           = pickle.load(open("models/svc.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
feature_names = pickle.load(open("models/feature_names.pkl", "rb"))   # 132 symptom column names

# Build symptoms_dict directly from feature_names so it is ALWAYS in sync with the model
symptoms_dict = {name: idx for idx, name in enumerate(feature_names)}

# Human-readable symptom list for the autocomplete UI (convert underscores to spaces)
ALL_SYMPTOMS = sorted([name.replace("_", " ") for name in feature_names])


# ── Helper: fetch disease info from lookup CSVs ───────────────────────────────
def helper(disease):
    # Description
    desc_row = description[description["Disease"] == disease]["Description"]
    desc = " ".join(desc_row.values) if not desc_row.empty else "No description available."

    # Precautions
    pre_rows = precautions[precautions["Disease"] == disease][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ]
    pre = list(pre_rows.values[0]) if not pre_rows.empty else []

    # Medications
    med_rows = medications[medications["Disease"] == disease]["Medication"]
    med = list(med_rows.values) if not med_rows.empty else []

    # Diet
    diet_rows = diets[diets["Disease"] == disease]["Diet"]
    diet = list(diet_rows.values) if not diet_rows.empty else []

    # Workout
    workout_rows = workout[workout["disease"] == disease]["workout"]
    wrkout = workout_rows if not workout_rows.empty else pd.Series([], dtype=str)

    return desc, pre, med, diet, wrkout


# ── Prediction function ───────────────────────────────────────────────────────
def get_predicted_value(patient_symptoms):
    """
    patient_symptoms: list of strings entered by the user.
    Converts each to underscore form, looks it up in symptoms_dict,
    builds a binary feature vector, and returns the predicted disease name.
    """
    input_vector = np.zeros(len(feature_names))
    matched = []
    unmatched = []

    for item in patient_symptoms:
        # Normalize: lowercase, strip extra spaces, replace spaces with underscore
        normalized = item.strip().lower().replace(" ", "_")
        if normalized in symptoms_dict:
            input_vector[symptoms_dict[normalized]] = 1
            matched.append(normalized)
        else:
            unmatched.append(item.strip())

    if not matched:
        raise ValueError(
            "None of the entered symptoms were recognized. "
            "Please select symptoms from the suggestions list."
        )

    # Predict using a DataFrame with correct column names
    df_input = pd.DataFrame([input_vector], columns=feature_names)
    predicted_index = svc.predict(df_input)[0]
    disease_name = label_encoder.inverse_transform([predicted_index])[0]

    return disease_name, matched, unmatched


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("about"))


@app.route("/predict", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms_raw = request.form.get("symptoms", "").strip()

        if not symptoms_raw or symptoms_raw.lower() == "symptoms":
            return render_template(
                "index.html",
                message="Please enter at least one symptom.",
                symptoms="",
                all_symptoms=ALL_SYMPTOMS,
            )

        try:
            user_symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]
            predicted_disease, matched, unmatched = get_predicted_value(user_symptoms)
            dis_des, my_precautions, meds, rec_diet, wrkout = helper(predicted_disease)

            warning = None
            if unmatched:
                warning = (
                    f"These symptoms were not recognized and were ignored: "
                    f"{', '.join(unmatched)}. "
                    "Please use symptom names from the suggestion list."
                )

            return render_template(
                "index.html",
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=meds,
                my_diet=rec_diet,
                workout=wrkout,
                symptoms=symptoms_raw,
                matched_symptoms=matched,
                warning=warning,
                all_symptoms=ALL_SYMPTOMS,
            )

        except ValueError as e:
            return render_template(
                "index.html",
                message=str(e),
                symptoms=symptoms_raw,
                all_symptoms=ALL_SYMPTOMS,
            )
        except Exception as e:
            return render_template(
                "index.html",
                message=f"An error occurred: {str(e)}",
                symptoms=symptoms_raw,
                all_symptoms=ALL_SYMPTOMS,
            )

    return render_template("index.html", symptoms="", all_symptoms=ALL_SYMPTOMS)


# ── API endpoint: returns all valid symptoms as JSON (for autocomplete) ───────
@app.route("/api/symptoms")
def api_symptoms():
    return jsonify(ALL_SYMPTOMS)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


<<<<<<< HEAD
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
=======
if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 197e591 (Complete project rebuild: fixed ML pipeline, modernized UI, fixed datasets)
