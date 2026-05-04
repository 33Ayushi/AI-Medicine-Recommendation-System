import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('Loading data...')
df = pd.read_csv('datasets/Training.csv')
X = df.drop('prognosis', axis=1)
y = df['prognosis']
feature_names = list(X.columns)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Voting Ensemble: SVC + Random Forest + Naive Bayes
svc = SVC(kernel='linear', C=1, probability=True, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, random_state=42)
nb  = MultinomialNB()

ensemble = VotingClassifier(
    estimators=[('svc', svc), ('rf', rf), ('nb', nb)],
    voting='soft'
)

print('Training ensemble (SVC + RF + NB)...')
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Ensemble Test Accuracy: {acc*100:.2f}%')

# Save model and label encoder
with open('models/svc.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Models saved to models/svc.pkl and models/label_encoder.pkl')

# Spot-check predictions
idx_map = {n: i for i, n in enumerate(feature_names)}

def predict(syms):
    vec = np.zeros(len(feature_names))
    for s in syms:
        s = s.lower().strip().replace(' ', '_')
        if s in idx_map:
            vec[idx_map[s]] = 1
    df_in = pd.DataFrame([vec], columns=feature_names)
    idx = ensemble.predict(df_in)[0]
    return le.inverse_transform([idx])[0]

tests = [
    (['itching', 'skin_rash', 'nodal_skin_eruptions'], 'Fungal infection'),
    (['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'], 'Allergy'),
    (['headache', 'dizziness', 'nausea', 'blurred_and_distorted_vision'], 'Migraine'),
    (['high_fever', 'sweating', 'headache', 'nausea', 'diarrhoea'], 'Typhoid'),
    (['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy'], 'Hypothyroidism'),
    (['chest_pain', 'fast_heart_rate', 'breathlessness'], 'Heart attack'),
    (['pus_filled_pimples', 'blackheads', 'skin_rash'], 'Acne'),
    (['fatigue', 'high_fever', 'chills', 'vomiting', 'toxic_look_(typhos)', 'belly_pain'], 'Typhoid'),
    (['depression', 'enlarged_thyroid', 'weight_gain', 'lethargy', 'brittle_nails'], 'Hypothyroidism'),
]

print()
passed = 0
for syms, expected in tests:
    result = predict(syms)
    ok = result == expected
    if ok:
        passed += 1
    status = 'PASS' if ok else 'FAIL'
    print('[' + status + '] ' + str(syms))
    print('       Expected: ' + expected)
    print('       Got:      ' + result)
    print()

print('Passed ' + str(passed) + '/' + str(len(tests)) + ' tests')
