from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Expected columns from the encoded feature space
        expected_columns = [
            'year', 'figure_kind_score', 'seat_type_Delhi region', 
            'seat_type_J&K open merit', 'seat_type_Outside Delhi', 
            'seat_type_Paid Fee Course', 'seat_type_Special Fee Course', 
            'seat_type_general'
        ]
        
        # Ensure all columns exist and are in the correct order
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_figure': float(prediction)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
