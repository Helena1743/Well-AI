import requests
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
cardio_model = joblib.load("model_cardio.joblib")
diabetes_model = joblib.load('model_diabetes.joblib')
stroke_model = joblib.load("model_stroke.joblib")

# Define mappings for categorical variables
gender_mapping = {
    'Male': 1,
    'Female': 0
}

blood_glucose_level_mapping = {
    '80-100 mg/dL': 0,
    '101-125 mg/dL': 1,
    '126+ mg/dL': 2
}

smoking_mapping = {
    'Never smoked': 0,
    'Formerly smoked': 1,
    'Smokes': 2
}

residenceType_mapping = {
    'Urban': 0,
    'Rural': 1
}

binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / (height ** 2)

def preprocess_input_for_cardio(gender, age, blood_glucose_level, smoking, alcohol, exercise, bmi, hba1c_level, diabetesnow, hypertension):
    gender_encoded = gender_mapping[gender]
    blood_glucose_level_encoded = blood_glucose_level_mapping[blood_glucose_level]
    smoking_encoded = smoking_mapping[smoking]
    alcohol_encoded = binary_mapping[alcohol]
    exercise_encoded = binary_mapping[exercise]
    diabetesnow_encoded = binary_mapping[diabetesnow]
    hypertension_encoded = binary_mapping[hypertension]

    return pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Glucose': [blood_glucose_level_encoded],
        'Smoking_Status': [smoking_encoded],
        'Alcohol Intake': [alcohol_encoded],
        'Physical Activity': [exercise_encoded],
        'BMI': [bmi],
        'HbA1c_level': [hba1c_level],
        'Diabetes_now': [diabetesnow_encoded],
        'Hypertension_now': [hypertension_encoded]
    })

def preprocess_input_for_diabetes(gender, age, smoking, bmi, hba1c_level, blood_glucose_level, diabetesnow, hypertension):
    gender_encoded = gender_mapping[gender]
    smoking_encoded = smoking_mapping[smoking]
    blood_glucose_level_encoded = blood_glucose_level_mapping[blood_glucose_level]
    diabetesnow_encoded = binary_mapping[diabetesnow]
    hypertension_encoded = binary_mapping[hypertension]

    return pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Smoking_Status': [smoking_encoded],
        'BMI': [bmi],
        'HbA1c_level': [hba1c_level],
        'Glucose': [blood_glucose_level_encoded],
        'Diabetes_now': [diabetesnow_encoded],
        'Hypertension_now': [hypertension_encoded]
    })

def preprocess_input_for_stroke(gender, age, residenceType, blood_glucose_level, bmi, smoking, hba1c_level, diabetesnow, hypertension):
    gender_encoded = gender_mapping[gender]
    residenceType_encoded = residenceType_mapping[residenceType]
    blood_glucose_level_encoded = blood_glucose_level_mapping[blood_glucose_level]
    smoking_encoded = smoking_mapping[smoking]
    diabetesnow_encoded = binary_mapping[diabetesnow]
    hypertension_encoded = binary_mapping[hypertension]

    return pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Residence_type': [residenceType_encoded],
        'Glucose': [blood_glucose_level_encoded],
        'BMI': [bmi],
        'Smoking_Status': [smoking_encoded],
        'HbA1c_level': [hba1c_level],
        'Diabetes_now': [diabetesnow_encoded],
        'Hypertension_now': [hypertension_encoded]
    })

def make_predictions(model, input_data):
    predicted_prob = model.predict_proba(input_data)[:, 1]
    predicted_prob_percent = predicted_prob * 100
    return predicted_prob_percent[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Function to convert float32 to float
def convert_float32_to_float(data):
    for key, value in data.items():
        if isinstance(value, np.float32):  # If the value is a numpy float32
            data[key] = float(value)  # Convert it to Python float
    return data

# Route for displaying the form and handling form submission
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect form data
        print(request.form)
        name = request.form['name']
        address = request.form['address']
        contact_number = request.form['contact_number']
        # gender = request.form['gender']
        gender = 'Male'
        age = int(request.form['age'])
        weight = float(request.form['weight'])  # Get weight input
        height = float(request.form['height'])  # Get height input
        married = request.form['married']
        workType = request.form['workType']
        # residenceType = request.form.get('residenceType')
        residenceType = 'Urban'
        exercise = request.form.get('exercise')
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        diabetesnow = request.form['diabetesnow']
        blood_glucose_level = request.form['blood_glucose_level']
        # hba1c_level = float(request.form['hba1c_level'])
        hba1c_level = 5.7
        alcohol = request.form.get('alcohol') 
        smoking = request.form['smoking']
        
        # Calculate BMI
        bmi = calculate_bmi(weight, height)
        bmi = round(bmi, 1)

        # Preprocess input for prediction
        cardio_input_data = preprocess_input_for_cardio(gender, age, blood_glucose_level, smoking, alcohol, exercise, bmi, hba1c_level, diabetesnow, hypertension)
        diabetes_input_data = preprocess_input_for_diabetes(gender, age, smoking, bmi, hba1c_level, blood_glucose_level, diabetesnow, hypertension)
        stroke_input_data = preprocess_input_for_stroke(gender, age, residenceType, blood_glucose_level, bmi, smoking, hba1c_level, diabetesnow, hypertension)
        
        # Make Prediction
        cardio_prob = make_predictions(cardio_model, cardio_input_data)
        diabetes_prob = make_predictions(diabetes_model, diabetes_input_data)
        stroke_prob = make_predictions(stroke_model, stroke_input_data)

        cardio_prob = round(cardio_prob, 2)
        diabetes_prob = round(diabetes_prob, 2)
        stroke_prob = round(stroke_prob, 2)

        # Prepare the data to send to the Google Apps Script web app
        data = {
            'name': name,
            'address': address,
            'contact_number': contact_number,
            'gender': gender,
            'age': age,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'married': married,
            'workType': workType,
            'residenceType': residenceType,
            'exercise': exercise,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'diabetesnow': diabetesnow,
            'blood_glucose_level': blood_glucose_level,
            'hba1c_level': hba1c_level,
            'alcohol': alcohol,
            'smoking': smoking,
            'cardio_prob': cardio_prob,
            'diabetes_prob': diabetes_prob,
            'stroke_prob': stroke_prob
        }

        # Convert any np.float32 to native float
        data = convert_float32_to_float(data)

        # Google Apps Script web app URL
        url = 'https://script.google.com/macros/s/AKfycbx4uDdoLcWeHbGC-K0QkgzUz-h-e5VTHjl2mQKH4vtRt9GUzKWekcFtz3kurfgGv48eOg/exec'  # Replace with your URL from Apps Script deployment

        # Send the data to Google Sheets using a POST request to the Apps Script web app
        response = requests.post(url, json=data)

        # Print the response status and content
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")

        if response.status_code == 200:
            print("Data successfully sent to Google Sheets.")
            print(data)  # Where 'data' includes all form fields, including smoking status
        else:
            print("Failed to send data to Google Sheets.")

        return render_template("index.html", 
                                gender=gender,
                                age=age,
                                weight=weight,
                                height=height,
                                bmi=bmi,
                                married=married,
                                workType=workType,
                                residenceType=residenceType,
                                exercise=exercise,
                                hypertension=hypertension,
                                heart_disease=heart_disease,
                                diabetesnow=diabetesnow,
                                blood_glucose_level=blood_glucose_level,
                                hba1c_level=hba1c_level,
                                alcohol=alcohol,
                                smoking=smoking,
                                cardio_prob=cardio_prob, 
                                diabetes_prob=diabetes_prob, 
                                stroke_prob=stroke_prob)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
