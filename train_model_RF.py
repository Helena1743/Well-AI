import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

# Load dataset
df = pd.read_csv("cleaned_dataset3.csv")

# Drop irrelevant columns
df = df.drop(columns=["Name"])

# Encode categorical features consistent with run.py
df["Marital Status"] = df["Marital Status"].map({
    "Single": 0,
    "Married": 1,
    "Divorced": 2,
    "Widow": 3
})

df["Working Status"] = df["Working Status"].map({
    "Homemaker": 0,
    "Private": 1,
    "Public": 2,
    "Retired": 3,
    "Student": 4,
    "Unemployed": 5,
    "Working": 6
})

df["Alcohol"] = df["Alcohol"].map({
    "No": 0,
    "Non-drinker": 0,
    "Occasional": 1,
    "Regular": 2,
    "Yes": 2
})

# Map binary columns to 0/1
binary_cols = [
    "Smoker", "Diabetes", "High Cholesterol",
    "High Blood Pressure", "Heart Disease", "Stroke"
]

for col in binary_cols:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# Define feature sets matching run.py preprocessing

cardio_feats = [
    "Age", "Marital Status", "Working Status", "Smoker", "Alcohol",
    "Body Mass (kg)", "BMI", "Blood Pressure (SYS)", "Blood Pressure (DIA)",
    "Pulse Rate", "High Cholesterol", "High Blood Pressure"
]

stroke_feats = [
    "Age", "Working Status", "Smoker", "Alcohol", "BMI",
    "Blood Pressure (SYS)", "Blood Pressure (DIA)", "Pulse Rate",
    "Blood Glucose", "High Blood Pressure", "Heart Disease"
]

diab_feats = [
    "Age", "Marital Status", "Working Status", "Smoker", "Alcohol",
    "Body Mass (kg)", "Height (cm)", "BMI", "Blood Glucose", "High Cholesterol"
]

# NOTE:
# Your dataset currently has columns like 'Smoker', 'Alcohol', 'Exercise', etc.
# But run.py expects 'Gender', 'Glucose', 'Smoking_Status', 'Alcohol Intake', 'Physical Activity', etc.
# So we need to create/match these columns exactly in training.

# Create the columns needed:


# Map Blood Glucose levels like run.py expects
# run.py uses: '80-100 mg/dL' -> 0, '101-125 mg/dL' ->1, '126+ mg/dL' -> 2
# Your dataset probably has numerical glucose values, so we bucket them:
def glucose_bucket(glucose_val):
    if glucose_val <= 100:
        return 0
    elif glucose_val <= 125:
        return 1
    else:
        return 2

df['Glucose'] = df['Blood Glucose'].apply(glucose_bucket)

# Smoking_Status mapping: 'Never smoked'->0, 'Formerly smoked'->1, 'Smokes'->2
# If your dataset only has binary smoker yes/no, we can approximate:
# Let's map 'Smoker' column (0/1) to 2 if smoker, else 0
df['Smoking_Status'] = df['Smoker'].apply(lambda x: 2 if x == 1 else 0)

# Alcohol Intake mapping: 'Yes'/'No' or multi-level as above
# Your 'Alcohol' mapped 0-2 already matches 'No'=0, 'Occasional'=1, 'Regular'/'Yes'=2
df['Alcohol Intake'] = df['Alcohol']

# Physical Activity (Exercise): Assuming you have 'Exercise' column with 'Yes'/'No' or binary
if 'Exercise' in df.columns:
    df['Physical Activity'] = df['Exercise'].map({'No': 0, 'Yes': 1})
else:
    # Fallback to zeros if missing
    df['Physical Activity'] = 0

# HbA1c_level — ensure this column exists and is numeric
if 'HbA1c_level' not in df.columns:
    df['HbA1c_level'] = 0.0  # or fill with mean if missing

# Diabetes_now and Hypertension_now columns are binary
df['Diabetes_now'] = df['Diabetes']
df['Hypertension_now'] = df['High Blood Pressure']

# Residence_type for stroke model: map Urban=0, Rural=1
if 'Residence Type' in df.columns:
    df['Residence_type'] = df['Residence Type'].map({'Urban': 0, 'Rural': 1})
else:
    df['Residence_type'] = 0  # default urban if missing

# Training function
def run_problem(name, features, target, model_filename, scaler_filename):
    print(f"\n=== {name} Model Training ===")

    X = df[features]
    y = df[target]

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ts_s = scaler.transform(X_ts)

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'class_weight': ['balanced']
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=4,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_tr_s, y_tr)
    model = search.best_estimator_

    print(f"Best parameters for {name}: {search.best_params_}")

    y_pred = model.predict(X_ts_s)
    print("Accuracy:", accuracy_score(y_ts, y_pred))
    print(classification_report(y_ts, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_ts, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Feature importance plot
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.title(f"{name} - Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Cross-validation accuracy
    X_scaled = scaler.transform(X)
    cv_score = cross_val_score(model, X_scaled, y, cv=4)
    print(f"{name} 4-fold CV Accuracy: {cv_score.mean():.3f}")

    # Save model and scaler
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Saved model to {model_filename} and scaler to {scaler_filename}")

# Run training for each problem
run_problem("Cardio", cardio_feats, "Heart Disease", "model_cardio.joblib", "scaler_cardio.joblib")
run_problem("Stroke", stroke_feats, "Stroke", "model_stroke.joblib", "scaler_stroke.joblib")
run_problem("Diabetes", diab_feats, "Diabetes", "model_diabetes.joblib", "scaler_diabetes.joblib")
