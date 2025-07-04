from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('ADA.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/why')
def why():
    return render_template('why.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/investments')
def investments():
    return render_template('investments.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form data
        features = [x for x in request.form.values()]
        
        # Create DataFrame with correct column names
        columns = ['age', 'workclass', 'education', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                  'hours-per-week', 'native-country']
        
        data = pd.DataFrame([features], columns=columns)
        
        # Make prediction
        prediction = model.predict(data)
        
        # Prepare result
        if prediction[0] == 0:
            result = "Your earns more than 50,000. Yes, you are ready for investment. Invest wisely."
            pred_value = ">50k"
        else:
            result = "Your earns less than 50,000. Better to invest your money to learn skills."
            pred_value = "<=50k"
            
        return render_template('result.html', result=result, prediction=pred_value)
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('result.html', result="An error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, port=4000)