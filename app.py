from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load models
model_path = 'models/churn_prediction_model.pkl'
preprocessor_path = 'models/preprocessor.pkl'
label_encoder_path = 'models/label_encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    le_target = pickle.load(f)

# Get feature names for form (simplified key features)
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
print("Model loaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
            'Contract': request.form['Contract'],
            'InternetService': request.form['InternetService']
        }
        
        # Create full df matching training (fill missing with 'No')
        full_data = {col: 'No' for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaymentMethod']}
        full_data.update(data)
        df_input = pd.DataFrame([full_data])
        
        # Preprocess
        X_input = preprocessor.transform(df_input)
        
        # Predict
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        churn_prob = probability[1] * 100
        
        churn_label = le_target.inverse_transform([prediction])[0]
        
        return render_template('result.html', 
                             churn=churn_label, 
                             probability=f"{churn_prob:.1f}%",
                             data=data)
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
