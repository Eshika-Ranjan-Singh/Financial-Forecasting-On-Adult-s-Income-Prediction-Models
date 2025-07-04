import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(_name_)

# Load model
try:
    model = joblib.load('ADA.joblib')
except Exception as e:
    print(f"Error loading model: {e}")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=["POST"])
def submit():
    try:
        # Get user input
        input_feature = [x for x in request.form.values()]
        input_feature = [np.array(input_feature)]
        
        # Define column names (ensure these match your model's features)
        names = ['age', 'workclass', 'education', 'occupation', 'relationship', 
                 'race', 'sex', 'hours_per_week', 'mental_status', 
                 'native_country', 'capital_gain', 'capital_loss']
        
        # Create DataFrame
        data = pd.DataFrame([input_feature], columns=names)
        
        # Predict
        prediction = model.predict(data)
        
        # Prepare result
        if prediction[0] == 0:
            result = "You earn more than 50K. Ready for investment!"
            pred_text = ">50K"
        else:
            result = "You earn less than 50K. Consider upskilling."
            pred_text = "<=50K"
            
        return render_template("result.html", result=result, prediction=pred_text)
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template("result.html", result="Error: Invalid input.")

if _name_ == "_main_":
    app.run(debug=True, port=4000)